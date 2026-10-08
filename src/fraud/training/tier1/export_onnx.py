"""Ekspor model tier 1 terkalibrasi ke ONNX dan pendaftarannya ke model registry MLflow.

Graf ONNX berisi ansambel pohon LightGBM ditambah interpolasi linear isotonic, sehingga satu
versi registry menghasilkan peluang terkalibrasi langsung. Konverter kalibrasi bawaan skl2onnx
tidak dipakai: kalibrator isotonic di-fit pada skor mentah berskala logit, sedangkan konverter itu
memasukkan probabilitas dan mencari knot terdekat alih-alih menginterpolasi. Karena itu skor
mentah diambil langsung dari pohon (tanpa transformasi logistik), lalu dikalibrasi dengan graf
yang ditulis di sini dan diverifikasi terhadap model sklearn asli.

Ekspor gagal (tanpa mencatat atau mendaftarkan apa pun) kalau hasilnya tidak konsisten dengan
model sklearn pada split validasi penuh. Versi baru di registry tidak diberi alias, promosi ke
produksi adalah keputusan manusia.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.export_onnx [--thresholds PATH] [--smoke]
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
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from onnx import TensorProto, helper, numpy_helper
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression

from fraud.training.tier1.calibration import latest_calibration_run, load_calibrated_model
from fraud.training.tier1.candidates import (
    RANDOM_SEED,
    load_training_data,
    parent_run,
    setup_mlflow,
)
from fraud.training.tier1.dataset import FeatureSpec
from fraud.training.tier1.threshold import DEFAULT_OUTPUT_PATH

STAGE_EXPORT = "export_onnx"

REGISTERED_MODEL_NAME = "fraud-tier1"
SMOKE_REGISTERED_MODEL_NAME = "fraud-tier1-smoke"

# Klien telemetri bawaan onnxruntime mengirim event lewat HTTP dari thread latar belakang: itu
# panggilan jaringan yang tidak diminta, dan pada macOS bisa membuat proses abort saat penutupan.
ort.disable_telemetry_events()

ONNX_OPSET = 17
ML_OPSET = 3

# Kolom kedua keluaran node pohon berisi skor mentah kelas fraud kalau transformasinya dimatikan.
RAW_SCORE_COLUMN = 1

# Batas kegagalan ekspor, diukur pada split validasi penuh.
MAX_CLASS_MISMATCH_RATE = 1e-4
MAX_MEAN_SCORE_DIFF = 1e-5

LATENCY_WARMUP_CALLS = 200
LATENCY_CALLS = 3000


@dataclass(frozen=True)
class ConsistencyReport:
    """Perbandingan keluaran ONNX dengan model sklearn asli pada data yang sama.

    Attributes:
        n_rows: Jumlah baris yang dibandingkan.
        raw_max_abs_diff: Selisih mutlak terbesar skor mentah (logit).
        raw_mean_abs_diff: Selisih mutlak rata-rata skor mentah.
        score_max_abs_diff: Selisih mutlak terbesar peluang terkalibrasi.
        score_mean_abs_diff: Selisih mutlak rata-rata peluang terkalibrasi.
        rows_over_1e_6: Jumlah baris dengan selisih peluang terkalibrasi di atas 1e-6.
        rows_over_1e_4: Jumlah baris dengan selisih di atas 1e-4.
        rows_over_1e_2: Jumlah baris dengan selisih di atas 1e-2.
        class_mismatches: Jumlah transaksi yang kelas keputusannya berbeda, per skenario.
    """

    n_rows: int
    raw_max_abs_diff: float
    raw_mean_abs_diff: float
    score_max_abs_diff: float
    score_mean_abs_diff: float
    rows_over_1e_6: int
    rows_over_1e_4: int
    rows_over_1e_2: int
    class_mismatches: dict[str, int]

    def failures(self) -> list[str]:
        """Pelanggaran batas ekspor, kosong kalau konsisten."""
        problems = []
        for scenario, mismatches in self.class_mismatches.items():
            if mismatches / self.n_rows > MAX_CLASS_MISMATCH_RATE:
                problems.append(
                    f"skenario {scenario}:1 punya {mismatches} dari {self.n_rows} transaksi "
                    f"dengan kelas keputusan berbeda (batas {MAX_CLASS_MISMATCH_RATE:.2%})"
                )
        if self.score_mean_abs_diff > MAX_MEAN_SCORE_DIFF:
            problems.append(
                f"rata-rata selisih skor terkalibrasi {self.score_mean_abs_diff:.2e} "
                f"melebihi batas {MAX_MEAN_SCORE_DIFF:.0e}"
            )
        return problems


def load_matching_thresholds(path: Path, calibration_run_id: str) -> dict:
    """Membaca konfigurasi ambang dan memastikan dihitung dari kalibrasi yang sedang diekspor.

    Args:
        path: Lokasi file konfigurasi ambang.
        calibration_run_id: Run kalibrasi yang modelnya diekspor.

    Returns:
        Isi konfigurasi ambang.

    Raises:
        FileNotFoundError: Kalau file konfigurasi belum ada.
        ValueError: Kalau konfigurasi dihitung dari run kalibrasi lain, sehingga ambangnya
            tidak berlaku untuk model yang diekspor.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} belum ada, jalankan penetapan ambang lebih dulu.")
    config = json.loads(path.read_text())
    if config["source_calibration_run_id"] != calibration_run_id:
        raise ValueError(
            f"Ambang di {path} dihitung dari kalibrasi {config['source_calibration_run_id'][:8]}, "
            f"bukan {calibration_run_id[:8]} yang akan diekspor. Jalankan ulang penetapan ambang."
        )
    return config


def calibrated_parts(
    calibrated: CalibratedClassifierCV,
) -> tuple[LGBMClassifier, IsotonicRegression]:
    """Estimator LightGBM dan kalibrator isotonic di dalam model terkalibrasi.

    Raises:
        NotImplementedError: Kalau estimator yang dikalibrasi bukan LightGBM.
        ValueError: Kalau kalibratornya bukan satu isotonic bertipe clip dengan knot naik ketat.
    """
    if len(calibrated.calibrated_classifiers_) != 1:
        raise ValueError("Ekspor mengharapkan tepat satu kalibrator terlatih.")
    inner = calibrated.calibrated_classifiers_[0]
    estimator = inner.estimator
    if isinstance(estimator, FrozenEstimator):
        estimator = estimator.estimator
    if not isinstance(estimator, LGBMClassifier):
        raise NotImplementedError(
            f"Ekspor ONNX baru mendukung LightGBM, bukan {type(estimator).__name__}."
        )
    isotonic = inner.calibrators[0]
    if not isinstance(isotonic, IsotonicRegression):
        raise ValueError("Ekspor mengharapkan kalibrator isotonic dengan out_of_bounds clip.")
    if isotonic.get_params()["out_of_bounds"] != "clip":
        raise ValueError("Ekspor mengharapkan kalibrator isotonic dengan out_of_bounds clip.")
    if not np.all(np.diff(isotonic.X_thresholds_) > 0):
        raise ValueError("Knot kalibrator isotonic harus naik ketat.")
    return estimator, isotonic


def lightgbm_to_onnx(estimator: LGBMClassifier, n_features: int) -> onnx.ModelProto:
    """Mengonversi LightGBM ke ONNX dengan keluaran skor mentah (logit), bukan probabilitas.

    Args:
        estimator: Model LightGBM terlatih.
        n_features: Jumlah kolom masukan.

    Returns:
        Model ONNX yang keluaran skor kelas fraud-nya berada di kolom `RAW_SCORE_COLUMN`.
    """
    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    model = convert_sklearn(
        estimator,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset={"": ONNX_OPSET, "ai.onnx.ml": ML_OPSET},
        options={id(estimator): {"zipmap": False}},
    )
    assert isinstance(model, onnx.ModelProto)
    trees = [node for node in model.graph.node if node.op_type == "TreeEnsembleClassifier"]
    if len(trees) != 1:
        raise ValueError(f"Graf hasil konversi punya {len(trees)} node pohon, harapannya satu.")
    for attribute in trees[0].attribute:
        if attribute.name == "post_transform":
            attribute.s = b"NONE"
    return model


def add_isotonic_calibration(
    model: onnx.ModelProto, isotonic: IsotonicRegression
) -> onnx.ModelProto:
    """Menambahkan interpolasi linear isotonic di atas skor mentah keluaran pohon.

    Setara `IsotonicRegression.predict` dengan out_of_bounds clip: skor dijepit ke rentang knot,
    lalu diinterpolasi linear antar knot. Dihitung dalam float64 karena kemiringan antar blok
    bisa sangat curam, dan hasil akhirnya diubah ke float32.

    Args:
        model: Model ONNX hasil `lightgbm_to_onnx`.
        isotonic: Kalibrator isotonic terlatih dengan knot naik ketat.

    Returns:
        Model yang sama dengan dua keluaran: `raw_score` dan `calibrated_probability`.
    """
    x_knots = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
    y_knots = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
    graph = model.graph
    tree = next(node for node in graph.node if node.op_type == "TreeEnsembleClassifier")

    constants = {
        "iso_x0": x_knots[:-1],
        "iso_y0": y_knots[:-1],
        "iso_slope": np.diff(y_knots) / np.diff(x_knots),
        "iso_edges": x_knots[1:-1],
        "iso_min": np.array(x_knots[0]),
        "iso_max": np.array(x_knots[-1]),
        "iso_col": np.array(RAW_SCORE_COLUMN, dtype=np.int64),
        "iso_axis1": np.array([1], dtype=np.int64),
    }
    graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in constants.items())

    make = helper.make_node
    graph.node.extend(
        [
            make("Gather", [tree.output[1], "iso_col"], ["raw_score"], axis=1, name="iso_select"),
            make("Cast", ["raw_score"], ["iso_x64"], to=TensorProto.DOUBLE, name="iso_to_double"),
            make("Clip", ["iso_x64", "iso_min", "iso_max"], ["iso_clipped"], name="iso_clip"),
            make("Unsqueeze", ["iso_clipped", "iso_axis1"], ["iso_column"], name="iso_column"),
            make("GreaterOrEqual", ["iso_column", "iso_edges"], ["iso_past"], name="iso_past"),
            make("Cast", ["iso_past"], ["iso_past_int"], to=TensorProto.INT64, name="iso_cast"),
            make("ReduceSum", ["iso_past_int", "iso_axis1"], ["iso_segment"], keepdims=0,
                 name="iso_segment"),
            make("Gather", ["iso_x0", "iso_segment"], ["iso_seg_x0"], name="iso_gather_x0"),
            make("Gather", ["iso_y0", "iso_segment"], ["iso_seg_y0"], name="iso_gather_y0"),
            make("Gather", ["iso_slope", "iso_segment"], ["iso_seg_slope"], name="iso_gather_s"),
            make("Sub", ["iso_clipped", "iso_seg_x0"], ["iso_delta"], name="iso_delta"),
            make("Mul", ["iso_seg_slope", "iso_delta"], ["iso_rise"], name="iso_rise"),
            make("Add", ["iso_seg_y0", "iso_rise"], ["iso_y64"], name="iso_add"),
            make("Cast", ["iso_y64"], ["calibrated_probability"], to=TensorProto.FLOAT,
                 name="iso_to_float"),
        ]
    )
    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info("raw_score", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("calibrated_probability", TensorProto.FLOAT, [None]),
        ]
    )
    onnx.checker.check_model(model)
    return model


def decision_class(score: np.ndarray, reject: float, review: float) -> np.ndarray:
    """Kelas keputusan per transaksi: 2 tolak, 1 setujui dan review, 0 setujui biasa."""
    return np.where(score >= reject, 2, np.where(score >= review, 1, 0))


def check_consistency(
    session: ort.InferenceSession,
    features: pd.DataFrame,
    calibrated: CalibratedClassifierCV,
    estimator: LGBMClassifier,
    scenarios: dict[str, dict],
) -> ConsistencyReport:
    """Membandingkan keluaran ONNX dengan model sklearn asli pada data yang sama.

    Args:
        session: Sesi ONNX Runtime untuk model hasil ekspor.
        features: Fitur split validasi, urutan kolom sama dengan saat pelatihan.
        calibrated: Model terkalibrasi asli sebagai acuan peluang terkalibrasi.
        estimator: Model LightGBM di dalamnya, sebagai acuan skor mentah.
        scenarios: Ambang tolak dan review tiap skenario, dari konfigurasi ambang.

    Returns:
        Statistik selisih skor dan jumlah transaksi yang kelas keputusannya berbeda.
    """
    matrix = features.to_numpy(dtype=np.float32)
    outputs = session.run(["raw_score", "calibrated_probability"], {"input": matrix})
    onnx_raw = np.asarray(outputs[0], dtype=np.float64)
    onnx_score = np.asarray(outputs[1], dtype=np.float64)
    reference_raw = np.asarray(estimator.predict(features, raw_score=True), dtype=np.float64)
    reference_score = calibrated.predict_proba(features)[:, 1]

    raw_diff = np.abs(onnx_raw - reference_raw)
    score_diff = np.abs(onnx_score - reference_score)
    mismatches = {
        ratio: int(
            (
                decision_class(onnx_score, scenario["reject"], scenario["review"])
                != decision_class(reference_score, scenario["reject"], scenario["review"])
            ).sum()
        )
        for ratio, scenario in scenarios.items()
    }
    return ConsistencyReport(
        n_rows=len(matrix),
        raw_max_abs_diff=float(raw_diff.max()),
        raw_mean_abs_diff=float(raw_diff.mean()),
        score_max_abs_diff=float(score_diff.max()),
        score_mean_abs_diff=float(score_diff.mean()),
        rows_over_1e_6=int((score_diff > 1e-6).sum()),
        rows_over_1e_4=int((score_diff > 1e-4).sum()),
        rows_over_1e_2=int((score_diff > 1e-2).sum()),
        class_mismatches=mismatches,
    )


def measure_latency(model_bytes: bytes, features: pd.DataFrame) -> dict[str, float]:
    """Latensi inferensi satu baris pada satu thread, dalam milidetik.

    Mikro-benchmark di mesin pelatihan, bukan pengganti pengukuran beban pada jalur serving.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])

    matrix = features.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(RANDOM_SEED)
    sample = matrix[rng.integers(0, len(matrix), LATENCY_WARMUP_CALLS + LATENCY_CALLS)]
    for row in sample[:LATENCY_WARMUP_CALLS]:
        session.run(None, {"input": row[None, :]})

    timings = []
    for row in sample[LATENCY_WARMUP_CALLS:]:
        start = time.perf_counter()
        session.run(None, {"input": row[None, :]})
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "latency_p99_ms": float(np.percentile(timings, 99)),
        "latency_max_ms": float(np.max(timings)),
    }


def _signature(n_features: int) -> ModelSignature:
    """Tanda tangan model: satu masukan float32 dan dua keluaran per transaksi."""
    return ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, n_features), "input")]),
        outputs=Schema(
            [
                TensorSpec(np.dtype(np.float32), (-1,), "raw_score"),
                TensorSpec(np.dtype(np.float32), (-1,), "calibrated_probability"),
            ]
        ),
    )


def build_calibrated_onnx(
    calibrated: CalibratedClassifierCV, n_features: int
) -> tuple[onnx.ModelProto, LGBMClassifier]:
    """Graf ONNX mandiri dari model terkalibrasi: pohon LightGBM lalu interpolasi isotonic.

    Returns:
        Graf ONNX dan estimator LightGBM di dalamnya, yang dipakai pemeriksaan konsistensi.
    """
    estimator, isotonic = calibrated_parts(calibrated)
    return add_isotonic_calibration(lightgbm_to_onnx(estimator, n_features), isotonic), estimator


def export_metrics(
    report: ConsistencyReport,
    latency: dict[str, float],
    model_bytes: bytes,
    conversion_seconds: float,
) -> dict[str, float]:
    """Metrik ekspor untuk dicatat di MLflow: ukuran, waktu konversi, konsistensi, latensi."""
    return {
        "onnx_size_mb": len(model_bytes) / 1e6,
        "conversion_seconds": conversion_seconds,
        "raw_max_abs_diff": report.raw_max_abs_diff,
        "raw_mean_abs_diff": report.raw_mean_abs_diff,
        "score_max_abs_diff": report.score_max_abs_diff,
        "score_mean_abs_diff": report.score_mean_abs_diff,
        "rows_over_1e_4": float(report.rows_over_1e_4),
        "max_class_mismatches": float(max(report.class_mismatches.values())),
        **latency,
    }


def run_export(experiment_id: str, smoke: bool, thresholds_path: Path) -> ConsistencyReport:
    """Mengekspor model terkalibrasi terbaru ke ONNX, memverifikasi, dan mendaftarkannya.

    Args:
        experiment_id: Eksperimen MLflow tempat run kalibrasi dibaca.
        smoke: True untuk memotong data dan mendaftar ke nama registry uji, bukan yang asli.
        thresholds_path: Konfigurasi ambang, harus dihitung dari kalibrasi yang diekspor.

    Returns:
        Laporan konsistensi ONNX terhadap model sklearn.

    Raises:
        ValueError: Kalau ambang tidak cocok dengan kalibrasi, atau ONNX tidak konsisten
            dengan model sklearn. Tidak ada yang dicatat atau didaftarkan dalam kasus ini.
        NotImplementedError: Kalau kandidat terkalibrasi bukan LightGBM.
    """
    calibration_run = latest_calibration_run(experiment_id)
    candidate = calibration_run.data.tags["candidate"]
    thresholds = load_matching_thresholds(thresholds_path, calibration_run.info.run_id)
    print(f"Kalibrasi sumber: {candidate} (run {calibration_run.info.run_id[:8]})", flush=True)

    calibrated = load_calibrated_model(calibration_run)
    train, validation, spec = load_training_data(smoke)
    n_features = len(spec["input_columns"])

    start = time.perf_counter()
    model, estimator = build_calibrated_onnx(calibrated, n_features)
    model_bytes = model.SerializeToString()
    conversion_seconds = time.perf_counter() - start
    print(f"Konversi: {conversion_seconds:.0f} detik, {len(model_bytes) / 1e6:.1f} MB", flush=True)

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    report = check_consistency(
        session, validation.features, calibrated, estimator, thresholds["scenarios"]
    )
    print(json.dumps(asdict(report), indent=2), flush=True)
    problems = report.failures()
    if problems:
        raise ValueError("ONNX tidak konsisten dengan model sklearn: " + "; ".join(problems))

    latency = measure_latency(model_bytes, validation.features)
    print("Latensi satu baris: " + ", ".join(f"{k} {v:.2f}" for k, v in latency.items()))

    registered_name = SMOKE_REGISTERED_MODEL_NAME if smoke else REGISTERED_MODEL_NAME
    with parent_run(candidate, STAGE_EXPORT, train, validation, spec):
        log_and_register(
            model, spec, thresholds, thresholds_path, calibration_run.info.run_id,
            candidate, registered_name, n_features,
        )
        mlflow.log_params(
            {
                "source_calibration_run_id": calibration_run.info.run_id,
                "source_candidate": candidate,
                "onnx_opset": ONNX_OPSET,
                "registered_model_name": registered_name,
            }
        )
        mlflow.log_metrics(export_metrics(report, latency, model_bytes, conversion_seconds))
        mlflow.log_dict(asdict(report), "consistency_report.json")
    return report


def log_and_register(
    model: onnx.ModelProto,
    spec: FeatureSpec,
    thresholds: dict,
    thresholds_path: Path,
    calibration_run_id: str,
    candidate: str,
    registered_name: str,
    n_features: int,
) -> None:
    """Mencatat model ONNX beserta berkas pendampingnya dan mendaftarkannya ke registry."""
    with tempfile.TemporaryDirectory() as directory:
        spec_path = Path(directory) / "feature_spec.json"
        spec_path.write_text(json.dumps(dict(spec), ensure_ascii=False))
        copied_thresholds = Path(directory) / "thresholds.json"
        copied_thresholds.write_text(thresholds_path.read_text())
        # Default MLflow memindahkan tabel knot kecil ke berkas .data terpisah, sehingga
        # model.onnx yang disalin sendirian (misal ke image serving) gagal dimuat
        info = mlflow_onnx.log_model(
            model,
            name="model",
            save_as_external_data=False,
            registered_model_name=registered_name,
            signature=_signature(n_features),
            extra_files=[str(spec_path), str(copied_thresholds)],
            metadata={
                "source_calibration_run_id": calibration_run_id,
                "source_candidate": candidate,
                "selected_cost_ratio": thresholds["selected_cost_ratio"],
            },
        )
    print(
        f"Terdaftar: {registered_name} versi {info.registered_model_version} "
        "(tanpa alias, promosi ke produksi adalah keputusan manusia)",
        flush=True,
    )
    mlflow.log_param("registered_model_version", info.registered_model_version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ekspor model tier 1 terkalibrasi ke ONNX.")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Konfigurasi ambang, harus dihitung dari kalibrasi yang diekspor.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data dipotong, dicatat di eksperimen terpisah dan nama registry uji.",
    )
    args = parser.parse_args()
    if args.smoke and args.thresholds == DEFAULT_OUTPUT_PATH:
        parser.error("--smoke butuh --thresholds ke konfigurasi hasil uji, bukan yang produksi.")

    experiment_id = setup_mlflow(args.smoke)
    run_export(experiment_id, args.smoke, args.thresholds)


if __name__ == "__main__":
    main()
