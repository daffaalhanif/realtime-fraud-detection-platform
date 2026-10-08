"""Orkestrasi run utama pengujian tier 2: lengan berurutan, uji acak urutan, dan kontrol.

Ketiga lengan dijalankan sebagai satu paket perbandingan dengan konfigurasi hasil pemilihan dan
seed pengulangan yang sama: `ordered` (urutan asli), `shuffled` (uji acak urutan), dan
`current_only` (kontrol panjang sequence 1). Tiap pengulangan adalah satu pretraining dan satu
fine-tuning dengan seed sendiri.

Pengulangan yang sudah selesai di MLflow dilewati, sehingga perintah yang sama bisa dijalankan
ulang setelah sesi terputus. Urutannya seed di luar dan lengan di dalam, supaya ketiga lengan
selalu punya jumlah pengulangan yang sama kalau pekerjaan berhenti di tengah.

Dijalankan lewat:
    uv run python -m fraud.training.tier2.ablation_shuffle [--arms ...] [--seeds ...] [--dry-run]
"""

import argparse
import statistics
from pathlib import Path

import mlflow
from mlflow.entities import Run

from fraud.features.offline_store import INITIAL_PARQUET_DIR
from fraud.training.tier2.dataset import SEQUENCE_MODES, prepare_sequence_data
from fraud.training.tier2.finetune import (
    STAGE as FINETUNE_STAGE,
    candidate_key,
    finetune_and_log,
    selected_configuration,
)
from fraud.training.tier2.model import Tier2Config
from fraud.training.tier2.pretrain import (
    MAX_EPOCHS,
    SMOKE_MAX_EPOCHS,
    SMOKE_ROW_LIMIT,
    pretrain_and_log,
)
from fraud.training.tier2.tracking import HYPOTHESIS_SEEDS, select_device, setup_mlflow

PURPOSE = "main"
SMOKE_SEED_COUNT = 2


def arm_seq_len(mode: str, selected_seq_len: int) -> int:
    """Panjang sequence yang dicatat dan diekspor untuk satu lengan.

    Lengan `current_only` selalu dilatih dengan panjang 1; mencatat panjang terpilih untuknya
    akan membuat serving menyusun sequence lebih panjang dari yang dilihat model.
    """
    return 1 if mode == "current_only" else selected_seq_len


def finished_main_runs(experiment_id: str) -> list[Run]:
    """Run fine-tuning pengulangan utama yang sudah selesai."""
    return mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.stage = '{FINETUNE_STAGE}' and tags.purpose = '{PURPOSE}' "
            "and attributes.status = 'FINISHED'"
        ),
        max_results=1000,
    )


def latest_main_runs(
    experiment_id: str, seq_len: int, config: Tier2Config
) -> dict[tuple[str, int], Run]:
    """Run fine-tuning pengulangan utama terbaru per (mode, seed) dengan konfigurasi terpilih.

    Run dari konfigurasi lain tidak ikut, sehingga pemilihan yang diulang tidak mencampur
    pengulangan dari dua konfigurasi berbeda.
    """
    latest: dict[tuple[str, int], Run] = {}
    for run in sorted(
        finished_main_runs(experiment_id), key=lambda run: run.info.start_time, reverse=True
    ):
        params = run.data.params
        if candidate_key(params["seq_len"], params) == candidate_key(
            arm_seq_len(params["mode"], seq_len), config
        ):
            latest.setdefault((params["mode"], int(params["seed"])), run)
    return latest


def plan_runs(
    experiment_id: str,
    modes: list[str],
    seeds: list[int],
    seq_len: int,
    config: Tier2Config,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Membagi pasangan (mode, seed) menjadi yang sudah selesai dan yang masih harus dijalankan.

    Returns:
        Pasangan yang sudah selesai dan pasangan yang tersisa, urut seed lalu lengan.
    """
    done_keys = latest_main_runs(experiment_id, seq_len, config)
    ordered_pairs = [(mode, seed) for seed in seeds for mode in modes]
    done = [pair for pair in ordered_pairs if pair in done_keys]
    remaining = [pair for pair in ordered_pairs if pair not in done_keys]
    return done, remaining


def print_summary(experiment_id: str, seq_len: int, config: Tier2Config) -> None:
    """PR-AUC validasi per lengan; hanya informasi, putusan resmi dihitung di evaluasi."""
    by_mode: dict[str, list[float]] = {}
    for (mode, _), run in latest_main_runs(experiment_id, seq_len, config).items():
        by_mode.setdefault(mode, []).append(run.data.metrics["best_validation_pr_auc"])
    print("\nPR-AUC validasi per lengan (informasi, bukan putusan pengujian):")
    for mode in SEQUENCE_MODES:
        scores = by_mode.get(mode, [])
        if len(scores) >= 2:
            spread = f"{statistics.mean(scores):.4f} +/- {statistics.stdev(scores):.4f}"
        elif scores:
            spread = f"{scores[0]:.4f}"
        else:
            spread = "-"
        print(f"  {mode:13s} {spread} ({len(scores)} seed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run utama ketiga lengan pengujian tier 2.")
    parser.add_argument("--arms", nargs="+", choices=SEQUENCE_MODES, default=list(SEQUENCE_MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(HYPOTHESIS_SEEDS))
    parser.add_argument("--parquet-dir", type=Path, default=INITIAL_PARQUET_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Cetak rencana tanpa melatih.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: konfigurasi seleksi smoke, dua seed, eksperimen terpisah.",
    )
    args = parser.parse_args()

    experiment_id = setup_mlflow(args.smoke)
    selected = selected_configuration(experiment_id, args.smoke)
    if selected is None:
        raise SystemExit(
            "Pemilihan konfigurasi belum lengkap; jalankan `finetune select-report` untuk melihat "
            "kandidat yang belum selesai."
        )
    seq_len, config = selected
    seeds = args.seeds[:SMOKE_SEED_COUNT] if args.smoke else args.seeds
    max_epochs = SMOKE_MAX_EPOCHS if args.smoke else MAX_EPOCHS
    row_limit = SMOKE_ROW_LIMIT if args.smoke else None

    done, remaining = plan_runs(experiment_id, args.arms, seeds, seq_len, config)
    print(f"Konfigurasi terpilih: seq_len {seq_len}, {config.to_dict()}")
    print(f"Selesai: {len(done)}, tersisa: {len(remaining)}")
    for mode, seed in remaining:
        print(f"  akan dijalankan: {mode}, seed {seed}, seq_len {arm_seq_len(mode, seq_len)}")
    if args.dry_run or not remaining:
        print_summary(experiment_id, seq_len, config)
        return

    device = select_device()
    print(f"Menyiapkan data dari {args.parquet_dir} untuk device {device} ...", flush=True)
    data = prepare_sequence_data(args.parquet_dir, row_limit).to(device)
    for index, (mode, seed) in enumerate(remaining, start=1):
        print(f"\n[{index}/{len(remaining)}] {mode}, seed {seed}", flush=True)
        pretrain_run_id = pretrain_and_log(
            data, mode, seed, arm_seq_len(mode, seq_len), config, max_epochs, PURPOSE, row_limit
        )
        finetune_and_log(pretrain_run_id, data, max_epochs)
    print_summary(experiment_id, seq_len, config)


if __name__ == "__main__":
    main()
