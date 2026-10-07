"""Pretraining self-supervised model tier 2: menebak field transaksi yang disamarkan.

Sebagian field pada setiap posisi valid disamarkan, lalu model menebak isinya dari field lain
pada transaksi yang sama dan dari transaksi lain dalam sequence. Label fraud tidak dipakai sama
sekali. Hanya transaksi yang dinilai di split train yang menjadi sampel, supaya pretraining
tidak melihat sebaran fitur periode validasi dan uji. Loss validasi memakai penyamaran tetap
sehingga sebanding antar epoch, dan bobot epoch terbaik disimpan ke MLflow untuk fine-tuning.

Dijalankan lewat:
    uv run python -m fraud.training.tier2.pretrain --arm ordered --seed 42 --seq-len 32 [--smoke]
"""

import argparse
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

from fraud.features.offline_store import INITIAL_PARQUET_DIR
from fraud.training.tier2.dataset import (
    SEQUENCE_MODES,
    SequenceBatch,
    SequenceData,
    build_batch,
    iterate_rows,
    prepare_sequence_data,
)
from fraud.training.tier2.model import Tier2Config, Tier2Model
from fraud.training.tier2.tracking import select_device, setup_mlflow

STAGE = "pretrain"

# Porsi samaran mengikuti konvensi BERT, tidak dituning pada data ini.
MASK_RATE = 0.15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 512
MAX_EPOCHS = 10
PATIENCE = 2
LOG_EVERY_STEPS = 50
# Penyamaran validasi selalu dari seed ini, supaya loss antar epoch, seed, dan lengan sebanding.
VALIDATION_SEED = 0

SMOKE_ROW_LIMIT = 30_000
SMOKE_MAX_EPOCHS = 2
SMOKE_CONFIG = Tier2Config(d_model=16, n_layers=1, n_heads=2, d_ff=32, cat_dim=4)

CHECKPOINT_ARTIFACT = "pretrain_checkpoint.pt"
CONFIG_ARTIFACT = "model_config.json"
SPEC_ARTIFACT = "sequence_spec.json"

LOSS_PARTS = ("categorical", "numeric", "missing")


@dataclass(frozen=True)
class MaskedInputs:
    """Input model setelah penyamaran, beserta letak field yang disamarkan.

    Attributes:
        numeric: Nilai numerik, 0 pada field yang disamarkan.
        missing: Indikator kosong, 0 pada field yang disamarkan.
        categorical: Kode kategori, kode MASK kolomnya pada field yang disamarkan.
        masked_numeric: True untuk field numerik yang disamarkan, bool `(B, L, kolom numerik)`.
        masked_categorical: True untuk field kategori yang disamarkan, bool `(B, L, kolom kategorikal)`.
    """

    numeric: torch.Tensor
    missing: torch.Tensor
    categorical: torch.Tensor
    masked_numeric: torch.Tensor
    masked_categorical: torch.Tensor


def mask_fields(
    batch: SequenceBatch, mask_codes: torch.Tensor, generator: torch.Generator
) -> MaskedInputs:
    """Menyamarkan tiap field pada posisi valid dengan peluang `MASK_RATE`.

    Args:
        batch: Batch hasil `build_batch`.
        mask_codes: Kode MASK per kolom kategorikal, dari `Tier2Model.tokenizer.mask_codes`.
        generator: Generator di device yang sama dengan batch.

    Returns:
        Input tersamarkan; padding tidak pernah disamarkan.
    """
    valid = ~batch.padding_mask[..., None]
    device = batch.numeric.device
    masked_numeric = (
        torch.rand(batch.numeric.shape, generator=generator, device=device) < MASK_RATE
    ) & valid
    masked_categorical = (
        torch.rand(batch.categorical.shape, generator=generator, device=device) < MASK_RATE
    ) & valid
    return MaskedInputs(
        numeric=batch.numeric.masked_fill(masked_numeric, 0.0),
        missing=batch.missing.masked_fill(masked_numeric, 0),
        categorical=torch.where(masked_categorical, mask_codes, batch.categorical),
        masked_numeric=masked_numeric,
        masked_categorical=masked_categorical,
    )


def _masked_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    # Batch kecil bisa tanpa field terpilih; dibagi minimal 1 supaya tidak menghasilkan NaN.
    return (values * selected).sum() / selected.sum().clamp(min=1)


def pretrain_losses(
    model: Tier2Model, batch: SequenceBatch, inputs: MaskedInputs
) -> dict[str, torch.Tensor]:
    """Loss tiap bagian tebakan, dihitung hanya pada field yang disamarkan.

    Returns:
        `categorical` (cross-entropy kode kategori), `numeric` (MSE nilai yang aslinya tidak
        kosong), `missing` (BCE indikator kosong), dan `total` sebagai jumlah ketiganya.
    """
    hidden = model.encode(
        inputs.numeric,
        inputs.missing,
        inputs.categorical,
        batch.elapsed,
        batch.padding_mask,
        masked=inputs.masked_numeric,
    )
    outputs = model.pretrain_outputs(hidden)

    # Loss dihitung di semua posisi lalu dikali mask, bukan dipilih lewat indeks boolean: indeks
    # boolean memaksa sinkronisasi device per kolom dan memperlambat tiap langkah.
    per_column = [
        F.cross_entropy(
            logits.float().flatten(0, 1),
            batch.categorical[..., column].flatten(),
            reduction="none",
        ).view_as(batch.categorical[..., column])
        for column, logits in enumerate(outputs.categorical_logits)
    ]
    categorical = _masked_mean(torch.stack(per_column, dim=-1), inputs.masked_categorical)

    target_missing = batch.missing.float()
    squared_error = (outputs.numeric.float() - batch.numeric) ** 2
    numeric = _masked_mean(squared_error, inputs.masked_numeric & (batch.missing == 0))
    missing_error = F.binary_cross_entropy_with_logits(
        outputs.missing_logits.float(), target_missing, reduction="none"
    )
    missing = _masked_mean(missing_error, inputs.masked_numeric)

    return {
        "categorical": categorical,
        "numeric": numeric,
        "missing": missing,
        "total": categorical + numeric + missing,
    }


def warmup_cosine(total_steps: int) -> Callable[[int], float]:
    """Pengali learning rate: naik linear selama warmup, lalu turun mengikuti kosinus ke nol."""
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def evaluate_pretraining(
    model: Tier2Model, data: SequenceData, seq_len: int, mode: str
) -> dict[str, float]:
    """Loss pretraining rata-rata di split validasi dengan penyamaran tetap."""
    device = data.numeric.device
    batch_generator = torch.Generator().manual_seed(VALIDATION_SEED)
    mask_generator = torch.Generator(device=device).manual_seed(VALIDATION_SEED)
    totals = dict.fromkeys([*LOSS_PARTS, "total"], 0.0)
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for rows in iterate_rows(data.split_rows["validation"], BATCH_SIZE, shuffle=False):
            batch = build_batch(data, rows, seq_len, mode, batch_generator)
            inputs = mask_fields(batch, model.tokenizer.mask_codes, mask_generator)
            losses = pretrain_losses(model, batch, inputs)
            for name, value in losses.items():
                totals[name] += float(value) * len(rows)
            n_samples += len(rows)
    return {name: value / n_samples for name, value in totals.items()}


def run_pretraining(
    data: SequenceData,
    mode: str,
    seed: int,
    seq_len: int,
    config: Tier2Config,
    max_epochs: int,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Melatih model pada tujuan pretraining dan mencatat kurvanya ke run MLflow aktif.

    Args:
        data: Data ter-encode, sudah di device pelatihan.
        mode: Mode sequence lengan, salah satu `SEQUENCE_MODES`.
        seed: Seed inisialisasi, urutan batch, pengacakan, dan penyamaran.
        seq_len: Panjang sequence termasuk transaksi yang dinilai.
        config: Ukuran arsitektur.
        max_epochs: Batas atas epoch sebelum early stopping.

    Returns:
        Bobot epoch dengan loss validasi terendah (di CPU), nomor epoch itu, dan loss-nya.
    """
    device = data.numeric.device
    torch.manual_seed(seed)
    model = Tier2Model(data.spec, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(data.split_rows["train"]) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps_per_epoch * max_epochs)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batch_generator = torch.Generator().manual_seed(seed)
    mask_generator = torch.Generator(device=device).manual_seed(seed)

    best_state: dict[str, torch.Tensor] = {}
    best_epoch, best_loss, stale_epochs, step = 0, math.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        start = time.perf_counter()
        # Dijumlah di device dan dibaca sekali per epoch, supaya tidak sinkron tiap langkah.
        sums = {name: torch.zeros((), device=device) for name in [*LOSS_PARTS, "total"]}
        n_batches = 0
        for rows in iterate_rows(data.split_rows["train"], BATCH_SIZE, True, batch_generator):
            batch = build_batch(data, rows, seq_len, mode, batch_generator)
            inputs = mask_fields(batch, model.tokenizer.mask_codes, mask_generator)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                losses = pretrain_losses(model, batch, inputs)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            n_batches += 1
            for name, value in losses.items():
                sums[name] += value.detach()
            if step % LOG_EVERY_STEPS == 0:
                mlflow.log_metric("train_loss_step", losses["total"].item(), step=step)

        train_means = {name: value.item() / n_batches for name, value in sums.items()}
        validation = evaluate_pretraining(model, data, seq_len, mode)
        epoch_seconds = time.perf_counter() - start
        mlflow.log_metrics(
            {
                **{f"train_{name}": value for name, value in train_means.items()},
                **{f"validation_{name}": value for name, value in validation.items()},
                "epoch_seconds": epoch_seconds,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            },
            step=epoch,
        )
        print(
            f"  epoch {epoch}: train {train_means['total']:.4f}, "
            f"validasi {validation['total']:.4f}, {epoch_seconds:.0f} detik",
            flush=True,
        )

        if validation["total"] < best_loss:
            best_loss, best_epoch, stale_epochs = validation["total"], epoch, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"  berhenti: validasi tidak membaik {PATIENCE} epoch", flush=True)
                break
    return best_state, best_epoch, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretraining self-supervised model tier 2.")
    parser.add_argument(
        "--arm", required=True, choices=SEQUENCE_MODES, help="Mode sequence lengan."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, required=True, help="Panjang sequence.")
    parser.add_argument("--parquet-dir", type=Path, default=INITIAL_PARQUET_DIR)
    defaults = Tier2Config()
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--n-layers", type=int, default=defaults.n_layers)
    parser.add_argument("--n-heads", type=int, default=defaults.n_heads)
    parser.add_argument("--d-ff", type=int, default=defaults.d_ff)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data, model, dan epoch dipotong, dicatat di eksperimen terpisah.",
    )
    args = parser.parse_args()

    if args.smoke:
        config, max_epochs, row_limit = SMOKE_CONFIG, SMOKE_MAX_EPOCHS, SMOKE_ROW_LIMIT
    else:
        config = Tier2Config(
            d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff
        )
        max_epochs, row_limit = MAX_EPOCHS, None

    setup_mlflow(args.smoke)
    device = select_device()
    print(f"Menyiapkan data dari {args.parquet_dir} untuk device {device} ...", flush=True)
    data = prepare_sequence_data(args.parquet_dir, row_limit).to(device)

    arm = f"t2_{args.arm}"
    run_name = f"{arm}-{STAGE}-L{args.seq_len}-seed{args.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {"arm": arm, "stage": STAGE, "seed": str(args.seed), "seq_len": str(args.seq_len)}
        )
        mlflow.log_params(
            {
                **config.to_dict(),
                "mode": args.arm,
                "seq_len": args.seq_len,
                "seed": args.seed,
                "mask_rate": MASK_RATE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_fraction": WARMUP_FRACTION,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "batch_size": BATCH_SIZE,
                "max_epochs": max_epochs,
                "patience": PATIENCE,
                "device": device.type,
                "n_train_samples": len(data.split_rows["train"]),
                "n_validation_samples": len(data.split_rows["validation"]),
            }
        )
        mlflow.log_dict(config.to_dict(), CONFIG_ARTIFACT)
        mlflow.log_dict(dict(data.spec), SPEC_ARTIFACT)

        best_state, best_epoch, best_loss = run_pretraining(
            data, args.arm, args.seed, args.seq_len, config, max_epochs
        )
        mlflow.log_metrics({"best_validation_total": best_loss, "best_epoch": best_epoch})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CHECKPOINT_ARTIFACT
            torch.save(best_state, path)
            mlflow.log_artifact(str(path))
    print(
        f"Selesai. Epoch terbaik {best_epoch}, loss validasi {best_loss:.4f}.\n"
        f"Run id pretraining (untuk fine-tuning): {run.info.run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
