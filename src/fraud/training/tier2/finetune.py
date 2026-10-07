"""Fine-tuning model tier 2 untuk skor fraud, dan prosedur pemilihan konfigurasinya.

Bobot hasil pretraining dilatih lanjut menebak label fraud transaksi yang dinilai, dengan
learning rate berbeda per kelompok parameter: kepala klasifikasi yang masih baru belajar paling
cepat, tokenizer yang paling dekat dengan input paling pelan. Seluruh pengaturan lengan (mode,
seed, panjang sequence, ukuran model) dibaca dari run pretraining, sehingga satu pengulangan
selalu terdiri dari pretraining dan fine-tuning dengan seed yang sama.

Dijalankan lewat:
    uv run python -m fraud.training.tier2.finetune run --pretrain-run-id ID [--smoke]
    uv run python -m fraud.training.tier2.finetune pipeline --arm ordered --seq-len 32 [...]
    uv run python -m fraud.training.tier2.finetune select-report [--smoke]
"""

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from mlflow import artifacts as mlflow_artifacts
from mlflow.entities import Run
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.features.offline_store import INITIAL_PARQUET_DIR
from fraud.training.tier2.dataset import (
    SequenceBatch,
    SequenceData,
    build_batch,
    iterate_rows,
    prepare_sequence_data,
)
from fraud.training.tier2.model import Tier2Config, Tier2Model
from fraud.training.tier2.pretrain import (
    BATCH_SIZE,
    CHECKPOINT_ARTIFACT as PRETRAIN_CHECKPOINT_ARTIFACT,
    CONFIG_ARTIFACT,
    GRAD_CLIP_NORM,
    SMOKE_CONFIG,
    SPEC_ARTIFACT,
    VALIDATION_SEED,
    WEIGHT_DECAY,
    add_run_arguments,
    pretrain_and_log,
    run_settings,
    warmup_cosine,
)
from fraud.training.tier2.tracking import select_device, setup_mlflow

STAGE = "finetune"

# Learning rate per kelompok `Tier2Model.parameter_groups`, turun sekitar tiga kali ke arah
# input. Kepala pretraining tidak ikut dilatih.
GROUP_LEARNING_RATES = {"classifier": 1e-3, "encoder": 3e-4, "tokenizer": 1e-4}
MAX_EPOCHS = 10
PATIENCE = 2

CHECKPOINT_ARTIFACT = "finetune_checkpoint.pt"

# Kandidat pemilihan konfigurasi, masing-masing satu pipeline di lengan berurutan seed 42.
# Konfigurasi terpilih dipakai ketiga lengan supaya perbandingannya tidak tercampur ukuran model.
SELECTION_PURPOSE = "selection"
SELECTION_MODE = "ordered"
SELECTION_SEQ_LENS = (16, 32, 64)
SELECTION_SIZES = {
    "small": Tier2Config(),
    "medium": Tier2Config(d_model=128, n_layers=4, n_heads=8, d_ff=256),
}
_SIZE_PARAMS = ("d_model", "n_layers", "n_heads", "d_ff")


def _load_pretraining(run_id: str) -> tuple[Run, Tier2Config, dict, dict[str, torch.Tensor]]:
    """Run, konfigurasi, kontrak elemen, dan bobot sebuah run pretraining."""
    run = mlflow.MlflowClient().get_run(run_id)
    with tempfile.TemporaryDirectory() as directory:
        local = Path(mlflow_artifacts.download_artifacts(run_id=run_id, dst_path=directory))
        config = Tier2Config(**json.loads((local / CONFIG_ARTIFACT).read_text()))
        spec = json.loads((local / SPEC_ARTIFACT).read_text())
        state = torch.load(local / PRETRAIN_CHECKPOINT_ARTIFACT)
    return run, config, spec, state


def _logits(model: Tier2Model, batch: SequenceBatch) -> torch.Tensor:
    return model(
        batch.numeric, batch.missing, batch.categorical, batch.elapsed, batch.padding_mask
    )


def predict(
    model: Tier2Model, data: SequenceData, split: str, seq_len: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Logit fraud dan label untuk seluruh transaksi yang dinilai di satu split.

    Mode `shuffled` memakai pengacakan tetap dari `VALIDATION_SEED`, supaya skor antar epoch,
    seed, dan pemanggilan sebanding.

    Returns:
        Logit dan label, keduanya urut sesuai `data.split_rows[split]`.
    """
    generator = torch.Generator().manual_seed(VALIDATION_SEED)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for rows in iterate_rows(data.split_rows[split], BATCH_SIZE, shuffle=False):
            batch = build_batch(data, rows, seq_len, mode, generator)
            logits.append(_logits(model, batch).float().cpu())
            labels.append(batch.label.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def run_finetuning(
    model: Tier2Model, data: SequenceData, mode: str, seed: int, seq_len: int, max_epochs: int
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Melatih kepala klasifikasi dan backbone, mencatat kurvanya ke run MLflow aktif.

    Returns:
        Bobot epoch dengan PR-AUC validasi tertinggi (di CPU), nomor epoch itu, dan PR-AUC-nya.
    """
    device = data.numeric.device
    torch.manual_seed(seed)
    groups = model.parameter_groups()
    for parameter in groups["pretrain_head"]:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [{"params": groups[name], "lr": rate} for name, rate in GROUP_LEARNING_RATES.items()],
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(data.split_rows["train"]) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps_per_epoch * max_epochs)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batch_generator = torch.Generator().manual_seed(seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    best_state: dict[str, torch.Tensor] = {}
    best_epoch, best_pr_auc, stale_epochs = 0, -math.inf, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        start = time.perf_counter()
        # Dijumlah di device dan dibaca sekali per epoch, supaya tidak sinkron tiap langkah.
        loss_sum = torch.zeros((), device=device)
        n_batches = 0
        for rows in iterate_rows(data.split_rows["train"], BATCH_SIZE, True, batch_generator):
            batch = build_batch(data, rows, seq_len, mode, batch_generator)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logit = _logits(model, batch)
            loss = F.binary_cross_entropy_with_logits(logit.float(), batch.label)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach()
            n_batches += 1

        logits, labels = predict(model, data, "validation", seq_len, mode)
        validation_loss = float(
            F.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(labels))
        )
        pr_auc = float(average_precision_score(labels, logits))
        epoch_seconds = time.perf_counter() - start
        mlflow.log_metrics(
            {
                "train_loss": loss_sum.item() / n_batches,
                "validation_loss": validation_loss,
                "validation_pr_auc": pr_auc,
                "validation_roc_auc": float(roc_auc_score(labels, logits)),
                "epoch_seconds": epoch_seconds,
            },
            step=epoch,
        )
        print(
            f"  epoch {epoch}: train {loss_sum.item() / n_batches:.4f}, "
            f"PR-AUC validasi {pr_auc:.4f}, {epoch_seconds:.0f} detik",
            flush=True,
        )

        if pr_auc > best_pr_auc:
            best_pr_auc, best_epoch, stale_epochs = pr_auc, epoch, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"  berhenti: PR-AUC validasi tidak membaik {PATIENCE} epoch", flush=True)
                break
    return best_state, best_epoch, best_pr_auc


def finetune_and_log(pretrain_run_id: str, data: SequenceData, max_epochs: int) -> str:
    """Fine-tuning satu run pretraining sebagai run MLflow baru.

    Args:
        pretrain_run_id: Run pretraining sumber bobot dan seluruh pengaturan lengan.
        data: Data ter-encode di device pelatihan, dibangun dengan batas baris yang sama
            dengan pretraining.
        max_epochs: Batas atas epoch sebelum early stopping.

    Returns:
        Id run fine-tuning.

    Raises:
        ValueError: Kontrak elemen data berbeda dari kontrak saat pretraining, misalnya karena
            Parquet yang dibaca bukan salinan yang sama.
    """
    pretrain_run, config, spec, state = _load_pretraining(pretrain_run_id)
    params = pretrain_run.data.params
    # Dibandingkan lewat JSON karena kontrak pretraining sudah melewati serialisasi JSON.
    if json.loads(json.dumps(dict(data.spec))) != spec:
        raise ValueError(
            f"Kontrak elemen data berbeda dari pretraining {pretrain_run_id}; kode kategori "
            "atau statistik normalisasi tidak lagi cocok dengan bobotnya."
        )
    mode, seed, seq_len = params["mode"], int(params["seed"]), int(params["seq_len"])
    purpose = pretrain_run.data.tags["purpose"]

    model = Tier2Model(data.spec, config).to(data.numeric.device)
    model.load_state_dict(state)

    arm = f"t2_{mode}"
    with mlflow.start_run(run_name=f"{arm}-{STAGE}-L{seq_len}-seed{seed}") as run:
        mlflow.set_tags(
            {
                "arm": arm,
                "stage": STAGE,
                "purpose": purpose,
                "seed": str(seed),
                "seq_len": str(seq_len),
            }
        )
        mlflow.log_params(
            {
                **config.to_dict(),
                "pretrain_run_id": pretrain_run_id,
                "mode": mode,
                "seq_len": seq_len,
                "seed": seed,
                "row_limit": params["row_limit"],
                **{f"learning_rate_{name}": rate for name, rate in GROUP_LEARNING_RATES.items()},
                "weight_decay": WEIGHT_DECAY,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "batch_size": BATCH_SIZE,
                "max_epochs": max_epochs,
                "patience": PATIENCE,
                "device": data.numeric.device.type,
            }
        )
        mlflow.log_dict(config.to_dict(), CONFIG_ARTIFACT)
        mlflow.log_dict(spec, SPEC_ARTIFACT)

        best_state, best_epoch, best_pr_auc = run_finetuning(
            model, data, mode, seed, seq_len, max_epochs
        )
        mlflow.log_metrics({"best_validation_pr_auc": best_pr_auc, "best_epoch": best_epoch})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CHECKPOINT_ARTIFACT
            torch.save(best_state, path)
            mlflow.log_artifact(str(path))
    print(
        f"Fine-tuning selesai. Epoch terbaik {best_epoch}, PR-AUC validasi {best_pr_auc:.4f}, "
        f"run id {run.info.run_id}",
        flush=True,
    )
    return run.info.run_id


def _row_limit(value: str) -> int | None:
    return None if value == "None" else int(value)


def run_command(args: argparse.Namespace) -> None:
    setup_mlflow(args.smoke)
    params = mlflow.MlflowClient().get_run(args.pretrain_run_id).data.params
    device = select_device()
    data = prepare_sequence_data(args.parquet_dir, _row_limit(params["row_limit"])).to(device)
    max_epochs = int(params["max_epochs"]) if args.smoke else MAX_EPOCHS
    finetune_and_log(args.pretrain_run_id, data, max_epochs)


def pipeline_command(args: argparse.Namespace) -> None:
    if args.purpose == SELECTION_PURPOSE and args.arm != SELECTION_MODE:
        raise SystemExit(f"Pemilihan konfigurasi hanya dilakukan di lengan {SELECTION_MODE}.")
    config, max_epochs, row_limit = run_settings(args)
    setup_mlflow(args.smoke)
    device = select_device()
    print(f"Menyiapkan data dari {args.parquet_dir} untuk device {device} ...", flush=True)
    data = prepare_sequence_data(args.parquet_dir, row_limit).to(device)
    pretrain_run_id = pretrain_and_log(
        data, args.arm, args.seed, args.seq_len, config, max_epochs, args.purpose, row_limit
    )
    finetune_and_log(pretrain_run_id, data, max_epochs if args.smoke else MAX_EPOCHS)


def selection_candidates(smoke: bool) -> list[tuple[int, Tier2Config]]:
    """Pasangan panjang sequence dan ukuran model yang wajib selesai sebelum memilih.

    Mode smoke memakai model mini untuk semua panjang, sama dengan yang dipakai `pipeline`.
    """
    sizes = [SMOKE_CONFIG] if smoke else list(SELECTION_SIZES.values())
    return [(seq_len, config) for seq_len in SELECTION_SEQ_LENS for config in sizes]


def _candidate_key(seq_len: int | str, config: Tier2Config | dict) -> tuple[str, ...]:
    values = config.to_dict() if isinstance(config, Tier2Config) else config
    return (str(seq_len), *(str(values[name]) for name in _SIZE_PARAMS))


def select_report_command(args: argparse.Namespace) -> None:
    experiment_id = setup_mlflow(args.smoke)
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.stage = '{STAGE}' and tags.purpose = '{SELECTION_PURPOSE}' "
            f"and tags.arm = 't2_{SELECTION_MODE}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
    )
    # Kandidat yang diulang diwakili run terbarunya.
    latest: dict[tuple[str, ...], Run] = {}
    for run in runs:
        latest.setdefault(_candidate_key(run.data.params["seq_len"], run.data.params), run)

    expected = [
        _candidate_key(seq_len, config) for seq_len, config in selection_candidates(args.smoke)
    ]
    missing = [key for key in expected if key not in latest]
    ranked = sorted(
        (latest[key] for key in expected if key in latest),
        key=lambda run: run.data.metrics["best_validation_pr_auc"],
        reverse=True,
    )
    header = ("seq_len", *_SIZE_PARAMS)
    print(f"Kandidat pemilihan selesai: {len(ranked)} dari {len(expected)}")
    print("  " + " ".join(f"{name:>8s}" for name in header) + f" {'PR-AUC':>8s}  run id")
    for run in ranked:
        key = _candidate_key(run.data.params["seq_len"], run.data.params)
        print(
            "  " + " ".join(f"{value:>8s}" for value in key)
            + f" {run.data.metrics['best_validation_pr_auc']:>8.4f}  {run.info.run_id}"
        )
    if missing:
        print("PERHATIAN: kandidat belum lengkap, konfigurasi belum boleh dipilih. Belum ada:")
        for key in missing:
            print("  " + ", ".join(f"{name} {value}" for name, value in zip(header, key)))
        return
    best_params = ranked[0].data.params
    best = dict(zip(header, _candidate_key(best_params["seq_len"], best_params)))
    print("Terpilih: " + ", ".join(f"{name} {value}" for name, value in best.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning dan pemilihan model tier 2.")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Fine-tuning dari run pretraining yang ada.")
    run_parser.add_argument("--pretrain-run-id", required=True)
    run_parser.add_argument("--parquet-dir", type=Path, default=INITIAL_PARQUET_DIR)
    run_parser.add_argument("--smoke", action="store_true", help="Eksperimen smoke.")
    run_parser.set_defaults(handler=run_command)

    pipeline_parser = commands.add_parser("pipeline", help="Pretraining lalu fine-tuning.")
    add_run_arguments(pipeline_parser)
    pipeline_parser.set_defaults(handler=pipeline_command)

    report_parser = commands.add_parser("select-report", help="Ringkasan pemilihan konfigurasi.")
    report_parser.add_argument("--smoke", action="store_true", help="Eksperimen smoke.")
    report_parser.set_defaults(handler=select_report_command)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
