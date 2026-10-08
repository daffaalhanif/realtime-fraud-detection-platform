"""Ekspor model tier 2 hasil fine-tuning ke ONNX, dengan gerbang kesetaraan terhadap PyTorch.

Graf yang diekspor adalah jalur inferensi `Tier2Model.forward`: lima tensor masukan dari
`build_batch` menjadi satu logit per transaksi yang dinilai. Logit belum dikalibrasi. Panjang
sequence dibuat tetap sesuai run fine-tuning karena serving selalu menyusun sequence sepanjang
itu dengan padding kiri, sedangkan ukuran batch dibiarkan dinamis.

Ekspor ditolak kalau ONNX Runtime tidak setara dengan PyTorch, kalau mask padding tidak lagi
berlaku di graf ONNX, atau kalau `model.onnx` tidak bisa dimuat sendirian. Pendaftaran ke model
registry hanya dilakukan dengan `--register`, tanpa alias.

Dijalankan lewat:
    uv run python -m fraud.training.tier2.export_onnx --finetune-run-id ID [--register] [--smoke]
"""

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.onnx as mlflow_onnx
import numpy as np
import onnx
import onnxruntime as ort
import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from fraud.features.offline_store import INITIAL_PARQUET_DIR
from fraud.training.tier2.dataset import SequenceBatch, build_batch, prepare_sequence_data
from fraud.training.tier2.finetune import (
    CHECKPOINT_ARTIFACT as FINETUNE_CHECKPOINT_ARTIFACT,
    ensure_same_spec,
    load_run_package,
)
from fraud.training.tier2.model import Tier2Model
from fraud.training.tier2.pretrain import VALIDATION_SEED
from fraud.training.tier2.tracking import setup_mlflow

STAGE = "export"

REGISTERED_MODEL_NAME = "fraud-tier2"
SMOKE_REGISTERED_MODEL_NAME = "fraud-tier2-smoke"

INPUT_NAMES = ("numeric", "missing", "categorical", "elapsed", "padding_mask")
OUTPUT_NAME = "logit"

CONSISTENCY_SAMPLE_SIZE = 4096
SINGLE_ROW_CHECKS = 64
# Galat float32 untuk graf yang sama jauh di bawah ini, sedangkan bug semantik seperti mask
# padding yang terabaikan menggeser logit jauh di atasnya.
MAX_LOGIT_DIFF = 1e-3
# Isi posisi padding tidak boleh memengaruhi logit sama sekali, di luar galat pembulatan.
MAX_PADDING_DIFF = 1e-6

LATENCY_WARMUP_CALLS = 200
LATENCY_CALLS = 3000


@dataclass(frozen=True)
class ExportReport:
    """Hasil pemeriksaan graf ONNX terhadap model PyTorch.

    Attributes:
        n_rows: Jumlah transaksi validasi yang dibandingkan.
        max_logit_diff: Selisih logit terbesar, batch besar.
        mean_logit_diff: Rata-rata selisih logit, batch besar.
        max_single_row_diff: Selisih logit terbesar saat dipanggil satu baris per panggilan.
        max_padding_diff: Perubahan logit terbesar saat isi posisi padding diacak; None kalau
            panjang sequence 1 sehingga tidak ada padding.
        standalone_loadable: `model.onnx` termuat dari folder yang hanya berisi berkas itu.
        size_mb: Ukuran `model.onnx`.
    """

    n_rows: int
    max_logit_diff: float
    mean_logit_diff: float
    max_single_row_diff: float
    max_padding_diff: float | None
    standalone_loadable: bool
    size_mb: float

    def failures(self) -> list[str]:
        """Gerbang yang tidak lolos, kosong kalau ekspor layak dipakai."""
        failed = []
        if max(self.max_logit_diff, self.max_single_row_diff) > MAX_LOGIT_DIFF:
            failed.append(f"selisih logit melebihi {MAX_LOGIT_DIFF}")
        if self.max_padding_diff is not None and self.max_padding_diff > MAX_PADDING_DIFF:
            failed.append(f"isi padding mengubah logit lebih dari {MAX_PADDING_DIFF}")
        if not self.standalone_loadable:
            failed.append("model.onnx tidak bisa dimuat sendirian")
        return failed


def _feeds(batch: SequenceBatch) -> dict[str, np.ndarray]:
    tensors = (batch.numeric, batch.missing, batch.categorical, batch.elapsed, batch.padding_mask)
    return {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, tensors, strict=True)}


def _logit(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(session.run([OUTPUT_NAME], feeds)[0])


def _session(model_bytes: bytes, threads: int | None = None) -> ort.InferenceSession:
    # Telemetri ORT memicu abort saat interpreter ditutup dan melakukan panggilan keluar.
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    return ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])


def export_model(model: Tier2Model, example: SequenceBatch) -> onnx.ModelProto:
    """Mengekspor jalur inferensi model ke graf ONNX dengan ukuran batch dinamis.

    Args:
        model: Model dalam mode eval di CPU.
        example: Batch contoh berisi minimal dua sampel; ukuran batch satu membuat exporter
            menganggap ukuran itu tetap.
    """
    batch = torch.export.Dim("batch")
    program = torch.onnx.export(
        model,
        tuple(getattr(example, name) for name in INPUT_NAMES),
        input_names=list(INPUT_NAMES),
        output_names=[OUTPUT_NAME],
        dynamic_shapes={name: {0: batch} for name in INPUT_NAMES},
        dynamo=True,
    )
    if program is None:
        raise RuntimeError("Exporter tidak menghasilkan graf ONNX.")
    return program.model_proto


def check_export(
    model: Tier2Model, model_proto: onnx.ModelProto, sample: SequenceBatch
) -> ExportReport:
    """Membandingkan ONNX Runtime dengan PyTorch pada sampel validasi yang sama."""
    with torch.no_grad():
        reference = model(
            sample.numeric, sample.missing, sample.categorical, sample.elapsed, sample.padding_mask
        ).numpy()
    model_bytes = model_proto.SerializeToString()
    session = _session(model_bytes)
    feeds = _feeds(sample)
    exported = _logit(session, feeds)
    diff = np.abs(exported - reference)

    single_row = [
        _logit(session, {name: value[row : row + 1] for name, value in feeds.items()})
        for row in range(min(SINGLE_ROW_CHECKS, len(reference)))
    ]
    single_row_diff = np.abs(np.concatenate(single_row) - reference[: len(single_row)])

    max_padding_diff = None
    padding = feeds["padding_mask"]
    if padding.any():
        rng = np.random.default_rng(VALIDATION_SEED)
        noisy = dict(feeds)
        noisy["numeric"] = np.where(
            padding[..., None], rng.normal(0, 5, feeds["numeric"].shape), feeds["numeric"]
        ).astype(np.float32)
        noisy["categorical"] = np.where(padding[..., None], 1, feeds["categorical"])
        noisy["elapsed"] = np.where(padding, rng.random(padding.shape), feeds["elapsed"]).astype(
            np.float32
        )
        perturbed = _logit(session, noisy)
        max_padding_diff = float(np.abs(perturbed - exported).max())

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.onnx"
        onnx.save_model(model_proto, path, save_as_external_data=False)
        standalone = sorted(item.name for item in Path(directory).iterdir()) == ["model.onnx"]
        try:
            _session(path.read_bytes())
        except Exception:  # noqa: BLE001 - setiap kegagalan muat berarti gerbang gagal
            standalone = False
        size_mb = path.stat().st_size / 2**20

    return ExportReport(
        n_rows=len(reference),
        max_logit_diff=float(diff.max()),
        mean_logit_diff=float(diff.mean()),
        max_single_row_diff=float(single_row_diff.max()),
        max_padding_diff=max_padding_diff,
        standalone_loadable=standalone,
        size_mb=size_mb,
    )


def measure_latency(model_bytes: bytes, sample: SequenceBatch) -> dict[str, float]:
    """Latensi inferensi satu transaksi pada satu thread, dalam milidetik.

    Mikro-benchmark di mesin pelatihan, bukan pengganti pengukuran beban pada jalur serving.
    """
    session = _session(model_bytes, threads=1)
    feeds = _feeds(sample)
    rng = np.random.default_rng(VALIDATION_SEED)
    rows = rng.integers(0, len(feeds["numeric"]), LATENCY_WARMUP_CALLS + LATENCY_CALLS)

    def call(row: int) -> None:
        _logit(session, {name: value[row : row + 1] for name, value in feeds.items()})

    for row in rows[:LATENCY_WARMUP_CALLS]:
        call(row)
    timings = []
    for row in rows[LATENCY_WARMUP_CALLS:]:
        start = time.perf_counter()
        call(row)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "latency_p99_ms": float(np.percentile(timings, 99)),
        "latency_max_ms": float(np.max(timings)),
    }


def _signature(seq_len: int, n_numeric: int, n_categorical: int) -> ModelSignature:
    """Tanda tangan model: lima masukan per batch sequence dan satu logit per transaksi."""
    return ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype(np.float32), (-1, seq_len, n_numeric), "numeric"),
                TensorSpec(np.dtype(np.uint8), (-1, seq_len, n_numeric), "missing"),
                TensorSpec(np.dtype(np.int64), (-1, seq_len, n_categorical), "categorical"),
                TensorSpec(np.dtype(np.float32), (-1, seq_len), "elapsed"),
                TensorSpec(np.dtype(np.bool_), (-1, seq_len), "padding_mask"),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype(np.float32), (-1,), OUTPUT_NAME)]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ekspor model tier 2 ke ONNX.")
    parser.add_argument("--finetune-run-id", required=True)
    parser.add_argument("--parquet-dir", type=Path, default=INITIAL_PARQUET_DIR)
    parser.add_argument(
        "--register", action="store_true", help="Daftarkan ke model registry, tanpa alias."
    )
    parser.add_argument("--smoke", action="store_true", help="Eksperimen dan registry smoke.")
    args = parser.parse_args()

    setup_mlflow(args.smoke)
    finetune_run, config, spec, state = load_run_package(
        args.finetune_run_id, FINETUNE_CHECKPOINT_ARTIFACT
    )
    params = finetune_run.data.params
    mode, seq_len, seed = params["mode"], int(params["seq_len"]), int(params["seed"])
    row_limit = None if params["row_limit"] == "None" else int(params["row_limit"])

    data = prepare_sequence_data(args.parquet_dir, row_limit)
    ensure_same_spec(data, spec, args.finetune_run_id)
    model = Tier2Model(data.spec, config)
    model.load_state_dict(state)
    model.eval()

    validation_rows = data.split_rows["validation"]
    chosen = torch.from_numpy(
        np.sort(
            np.random.default_rng(VALIDATION_SEED).choice(
                len(validation_rows),
                min(CONSISTENCY_SAMPLE_SIZE, len(validation_rows)),
                replace=False,
            )
        )
    )
    generator = torch.Generator().manual_seed(VALIDATION_SEED)
    sample = build_batch(data, validation_rows[chosen], seq_len, mode, generator)

    print(f"Mengekspor {params['mode']} L={seq_len} seed {seed} ...", flush=True)
    model_proto = export_model(model, sample)
    report = check_export(model, model_proto, sample)
    latency = measure_latency(model_proto.SerializeToString(), sample)
    print(json.dumps({**asdict(report), **latency}, indent=2), flush=True)

    arm = f"t2_{mode}"
    with mlflow.start_run(run_name=f"{arm}-{STAGE}-L{seq_len}-seed{seed}"):
        mlflow.set_tags(
            {
                "arm": arm,
                "stage": STAGE,
                "purpose": finetune_run.data.tags["purpose"],
                "seed": str(seed),
                "seq_len": str(seq_len),
            }
        )
        mlflow.log_params(
            {"finetune_run_id": args.finetune_run_id, "mode": mode, "seq_len": seq_len}
        )
        mlflow.log_metrics(
            {
                **{
                    name: float(value)
                    for name, value in asdict(report).items()
                    if value is not None
                },
                **latency,
            }
        )
        mlflow.log_dict(asdict(report), "export_report.json")
        failures = report.failures()
        if failures:
            raise SystemExit("Ekspor ditolak: " + "; ".join(failures))

        export_info = {"finetune_run_id": args.finetune_run_id, "mode": mode, "seq_len": seq_len}
        registered_name = SMOKE_REGISTERED_MODEL_NAME if args.smoke else REGISTERED_MODEL_NAME
        with tempfile.TemporaryDirectory() as directory:
            extra_files = []
            for name, content in (
                ("sequence_spec.json", spec),
                ("model_config.json", config.to_dict()),
                ("export_info.json", export_info),
            ):
                path = Path(directory) / name
                path.write_text(json.dumps(content, ensure_ascii=False))
                extra_files.append(str(path))
            # Default MLflow bisa memindahkan tensor kecil ke berkas .data terpisah, sehingga
            # model.onnx yang disalin sendirian ke image serving gagal dimuat.
            info = mlflow_onnx.log_model(
                model_proto,
                name="model",
                save_as_external_data=False,
                registered_model_name=registered_name if args.register else None,
                signature=_signature(
                    seq_len, len(spec["numeric_columns"]), len(spec["categorical_columns"])
                ),
                extra_files=extra_files,
                metadata=export_info,
            )
        if args.register:
            mlflow.log_param("registered_model_version", info.registered_model_version)
            print(
                f"Terdaftar: {registered_name} versi {info.registered_model_version} "
                "(tanpa alias, promosi ke produksi adalah keputusan manusia)",
                flush=True,
            )
    print("Ekspor lolos seluruh gerbang.", flush=True)


if __name__ == "__main__":
    main()
