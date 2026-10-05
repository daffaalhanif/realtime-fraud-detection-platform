"""Pembanding adil untuk pengujian tier 2: tier 1 tanpa dan dengan fitur jendela waktu.

Dua lengan dilatih dengan hyperparameter LightGBM yang sama persis dengan model tier 1
produksi versi 1, masing-masing dengan beberapa seed: `t1` (kontrak input versi 1) dan
`t1_plus` (ditambah fitur jendela waktu). Lengan `t1_plus` menjadi pembanding tier 2, supaya
keunggulan tier 2 tidak keliru dibaca sebagai keunggulan sekadar mengetahui aktivitas terbaru.
Lengan `t1` dilatih ulang, bukan diambil dari run konfirmasi tier 1, karena model per seed
di sana tidak tersimpan, padahal perbandingan berpasangan membutuhkan skornya.

Hanya split train dan validasi yang dipakai. Split uji dievaluasi terpisah, sekali, bersama
seluruh lengan pengujian tier 2. Model di sini tidak didaftarkan ke model registry.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.fair_comparator [--arms t1 t1_plus] [--smoke]
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import numpy as np
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from mlflow import artifacts as mlflow_artifacts
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.training.tier1.candidates import RANDOM_SEED, parent_run
from fraud.training.tier1.dataset import WINDOW_FEATURE_NAMES, Split, prepare_datasets

EXPERIMENT_NAME = "tier2-hypothesis"
SMOKE_EXPERIMENT_NAME = "tier2-hypothesis-smoke"

CANDIDATE = "lightgbm"
THRESHOLDS_PATH = Path("configs/tier1_thresholds.json")

# Lengan dan apakah fitur jendela waktu ikut menjadi input.
ARMS = {"t1": False, "t1_plus": True}

COMPARATOR_SEEDS = tuple(RANDOM_SEED + offset for offset in range(5))

SMOKE_ROW_LIMIT = 30_000
# Dua seed cukup untuk membuktikan simpangan baku bisa dihitung; pohon dikurangi supaya cepat.
SMOKE_SEED_COUNT = 2
SMOKE_N_ESTIMATORS = 50


def setup_mlflow(smoke: bool) -> str:
    """Menyambungkan ke server MLflow dan memilih eksperimen, mengembalikan id eksperimennya."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME)
    return experiment.experiment_id


def production_source_params(thresholds_path: Path = THRESHOLDS_PATH) -> tuple[str, dict]:
    """Hyperparameter LightGBM model tier 1 versi 1, ditelusuri dari konfigurasi ambangnya.

    Jejaknya: konfigurasi ambang menyimpan run kalibrasi, run kalibrasi menyimpan run
    pencarian sumbernya, dan run sumber menyimpan `best_params.json`.

    Args:
        thresholds_path: Konfigurasi ambang yang diekspor bersama model versi 1.

    Returns:
        Id run pencarian sumber dan hyperparameter lengkapnya.
    """
    calibration_run_id = json.loads(thresholds_path.read_text())["source_calibration_run_id"]
    calibration_run = mlflow.MlflowClient().get_run(calibration_run_id)
    source_run_id = calibration_run.data.params["source_run_id"]
    params = mlflow_artifacts.load_dict(f"runs:/{source_run_id}/best_params.json")
    return source_run_id, params


def _head(split: Split, rows: int) -> Split:
    return Split(
        features=split.features.iloc[:rows],
        label=split.label.iloc[:rows],
        entity_key=split.entity_key.iloc[:rows],
    )


def _window_gain_shares(model: LGBMClassifier, columns: list[str]) -> dict[str, float]:
    """Porsi total gain yang berasal dari tiap fitur jendela waktu."""
    gains = model.booster_.feature_importance(importance_type="gain")
    total = float(gains.sum())
    by_name = dict(zip(columns, gains, strict=True))
    return {
        f"gain_share_{name}": float(by_name[name]) / total if total else 0.0
        for name in WINDOW_FEATURE_NAMES
        if name in by_name
    }


def train_seed(
    arm: str, seed: int, params: dict[str, Any], train: Split, validation: Split
) -> float:
    """Melatih satu seed satu lengan sebagai run anak, lengkap dengan modelnya.

    Harus dipanggil di dalam `parent_run`.

    Returns:
        PR-AUC validasi seed ini.
    """
    with mlflow.start_run(run_name=f"{arm}-seed{seed}", nested=True):
        mlflow.set_tags(
            {"candidate": CANDIDATE, "arm": arm, "run_role": "trial", "seed": str(seed)}
        )
        model_params = {**params, "random_state": seed}
        mlflow.log_params(model_params)
        model = LGBMClassifier(**model_params)

        start = time.perf_counter()
        model.fit(train.features, train.label)
        fit_seconds = time.perf_counter() - start

        # Stub LightGBM menyertakan matriks sparse di tipe balikan; input padat selalu ndarray.
        scores = np.asarray(model.predict_proba(validation.features))[:, 1]
        pr_auc = float(average_precision_score(validation.label, scores))
        mlflow.log_metrics(
            {
                "validation_pr_auc": pr_auc,
                "validation_roc_auc": float(roc_auc_score(validation.label, scores)),
                "fit_seconds": fit_seconds,
                **_window_gain_shares(model, list(train.features.columns)),
            }
        )
        # Disimpan karena evaluasi berpasangan nanti memuat ulang model tiap seed.
        mlflow_lightgbm.log_model(model, name="model")
    print(f"  [{arm}] seed {seed}: PR-AUC {pr_auc:.4f}, {fit_seconds:.0f} detik", flush=True)
    return pr_auc


def run_arm(arm: str, smoke: bool) -> list[float]:
    """Melatih seluruh seed satu lengan di bawah satu run induk.

    Args:
        arm: Nama lengan, kunci di `ARMS`.
        smoke: True untuk memotong data, seed, dan jumlah pohon, hanya untuk uji coba cepat.

    Returns:
        PR-AUC validasi tiap seed, urut sesuai seed.
    """
    source_run_id, params = production_source_params()
    data = prepare_datasets(with_window_features=ARMS[arm])
    train, validation, spec = data.train, data.validation, data.spec
    seeds = COMPARATOR_SEEDS
    if smoke:
        train, validation = _head(train, SMOKE_ROW_LIMIT), _head(validation, SMOKE_ROW_LIMIT)
        seeds = seeds[:SMOKE_SEED_COUNT]
        params = {**params, "n_estimators": SMOKE_N_ESTIMATORS}

    scores = []
    with parent_run(CANDIDATE, arm, train, validation, spec):
        mlflow.set_tag("arm", arm)
        mlflow.log_params(
            {
                "arm": arm,
                "with_window_features": ARMS[arm],
                "source_run_id": source_run_id,
                "seeds": ",".join(str(seed) for seed in seeds),
            }
        )
        for seed in seeds:
            scores.append(train_seed(arm, seed, params, train, validation))
        mlflow.log_metrics(
            {
                "pr_auc_mean": statistics.mean(scores),
                "pr_auc_std": statistics.stdev(scores),
                "pr_auc_min": min(scores),
                "pr_auc_max": max(scores),
            }
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih pembanding adil tier 1 untuk tier 2.")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ARMS),
        default=list(ARMS),
        help="Lengan yang dilatih, default keduanya.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data, seed, dan pohon dipotong, dicatat di eksperimen terpisah.",
    )
    args = parser.parse_args()

    setup_mlflow(args.smoke)
    results = {}
    for arm in args.arms:
        print(f"Lengan {arm} ...", flush=True)
        results[arm] = run_arm(arm, args.smoke)

    print("\nPR-AUC validasi per lengan (rata-rata +/- simpangan baku):")
    for arm, scores in results.items():
        print(
            f"  {arm:8s} {statistics.mean(scores):.4f} +/- {statistics.stdev(scores):.4f} "
            f"({len(scores)} seed)"
        )
    print("Putusan promosi dihitung terpisah dengan kedua gerbang kriteria sukses.")


if __name__ == "__main__":
    main()
