"""Definisi empat kandidat model tier 1, satu percobaan pelatihan, dan baseline default.

Modul ini menyediakan bahan yang dipakai strategi pencarian (`fraud.training.tier1.search`):
cara membangun tiap kandidat, cara melatih dan menilai satu percobaan, dan cara mencatatnya
ke MLflow. Baseline melatih tiap kandidat dengan hyperparameter bawaan library sebagai acuan
sebelum tuning. Setiap percobaan menjadi run anak di bawah satu run induk per kandidat per
tahap. Hanya split train dan validasi yang dipakai, split uji tidak pernah dibaca di sini.

Dijalankan lewat (baseline):
    uv run python -m fraud.training.tier1.candidates [--candidates NAMA ...] [--smoke]
"""

import argparse
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import mlflow.sklearn as mlflow_sklearn
import mlflow.xgboost as mlflow_xgboost
import numpy as np
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud.training.tier1.dataset import FeatureSpec, Split, prepare_datasets

EXPERIMENT_NAME = "tier1-tuning"
SMOKE_EXPERIMENT_NAME = "tier1-tuning-smoke"
SMOKE_ROW_LIMIT = 30_000

STAGE_BASELINE = "baseline"
STAGE_PHASE1 = "phase1"
STAGE_PHASE2 = "phase2"
STAGE_CONFIRM = "confirm"

RANDOM_SEED = 42

# Dibatasi supaya laptop tetap responsif selama pelatihan panjang.
N_JOBS = 6

# Kode yang jarang muncul digabung, supaya kolom bernilai ribuan tidak meledakkan dimensi.
ONE_HOT_MIN_FREQUENCY = 100

# Positif dan berekor panjang, termasuk sentinel entitas baru: di-log1p khusus regresi logistik.
LOG_TRANSFORM_COLUMNS = (
    "txn_count_so_far",
    "seconds_since_last_txn",
    "amt_mean_so_far",
    "amt_std_so_far",
    "amt_max_so_far",
    "amt_ratio_to_mean",
    "TransactionAmt",
)

CANDIDATE_NAMES = ("logistic_regression", "random_forest", "lightgbm", "xgboost")

# Kandidat utama menurut rancangan, sisanya pembanding. Peran menentukan anggaran pencarian.
MAIN_CANDIDATES = ("lightgbm", "xgboost")

# Hanya pengaturan komputasi dan seed, hyperparameter lain dibiarkan bawaan library.
OPERATIONAL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {},
    "random_forest": {"n_jobs": N_JOBS, "random_state": RANDOM_SEED},
    "lightgbm": {"n_jobs": N_JOBS, "random_state": RANDOM_SEED, "verbose": -1},
    "xgboost": {"n_jobs": N_JOBS, "random_state": RANDOM_SEED},
}

_TREE_ESTIMATORS = {
    "random_forest": RandomForestClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
}

# skops menolak dua tipe ini secara default, dipercaya karena artefaknya dibuat sendiri.
_TRUSTED_SKOPS_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]
_log_sklearn_model = partial(mlflow_sklearn.log_model, skops_trusted_types=_TRUSTED_SKOPS_TYPES)

_MODEL_LOGGERS: dict[str, Callable[..., Any]] = {
    "logistic_regression": _log_sklearn_model,
    "random_forest": _log_sklearn_model,
    "lightgbm": mlflow_lightgbm.log_model,
    "xgboost": mlflow_xgboost.log_model,
}

# Kebalikan _MODEL_LOGGERS: dipakai tahap sesudah pencarian (kalibrasi, dst) untuk memuat
# model yang sudah tercatat. Publik karena dipakai lintas modul, tidak seperti _MODEL_LOGGERS.
MODEL_LOADERS: dict[str, Callable[[str], Any]] = {
    "logistic_regression": mlflow_sklearn.load_model,
    "random_forest": mlflow_sklearn.load_model,
    "lightgbm": mlflow_lightgbm.load_model,
    "xgboost": mlflow_xgboost.load_model,
}


@dataclass(frozen=True)
class TrialResult:
    """Hasil satu percobaan pelatihan."""

    run_name: str
    params: dict[str, Any]
    model: Any
    validation_pr_auc: float
    fit_seconds: float


def _logistic_regression_preprocessing(spec: FeatureSpec) -> ColumnTransformer:
    """Menyiapkan input regresi logistik: one-hot untuk nominal, skala untuk angka."""
    one_hot_columns = spec["encoded_columns"] + spec["nominal_numeric_columns"]
    special_columns = set(one_hot_columns) | set(LOG_TRANSFORM_COLUMNS)
    plain_columns = [column for column in spec["input_columns"] if column not in special_columns]

    one_hot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=ONE_HOT_MIN_FREQUENCY,
    )
    log_scaled = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    scaled = Pipeline(
        [
            # Indikator nilai kosong dipertahankan karena kosongnya banyak kolom berpola blok.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("one_hot", one_hot, one_hot_columns),
            ("log_scaled", log_scaled, list(LOG_TRANSFORM_COLUMNS)),
            ("scaled", scaled, plain_columns),
        ]
    )


def build_model(name: str, model_params: dict[str, Any], spec: FeatureSpec) -> Any:
    """Membangun model belum terlatih untuk satu kandidat dan satu kombinasi parameter."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("preprocess", _logistic_regression_preprocessing(spec)),
                ("model", LogisticRegression(**model_params)),
            ]
        )
    return _TREE_ESTIMATORS[name](**model_params)


def setup_mlflow(smoke: bool) -> str:
    """Menyambungkan ke server MLflow dan memilih eksperimen, mengembalikan id eksperimennya."""
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME)
    return experiment.experiment_id


def _head(split: Split, rows: int) -> Split:
    """Mengambil `rows` baris pertama sebuah split, untuk uji coba cepat."""
    return Split(
        features=split.features.iloc[:rows],
        label=split.label.iloc[:rows],
        entity_key=split.entity_key.iloc[:rows],
    )


def load_training_data(smoke: bool) -> tuple[Split, Split, FeatureSpec]:
    """Menyiapkan split train dan validasi beserta kontrak inputnya.

    Args:
        smoke: True untuk memotong kedua split, hanya untuk uji coba cepat.

    Returns:
        Tuple `(train, validation, spec)`. Split uji sengaja tidak dikembalikan.
    """
    data = prepare_datasets()
    train, validation = data.train, data.validation
    if smoke:
        train, validation = _head(train, SMOKE_ROW_LIMIT), _head(validation, SMOKE_ROW_LIMIT)
    return train, validation, data.spec


@contextmanager
def parent_run(
    name: str, stage: str, train: Split, validation: Split, spec: FeatureSpec
) -> Generator[mlflow.ActiveRun]:
    """Membuka run induk satu kandidat pada satu tahap, lengkap dengan fakta datanya."""
    with mlflow.start_run(run_name=f"{name}-{stage}") as run:
        mlflow.set_tags({"candidate": name, "search_stage": stage, "run_role": "parent"})
        mlflow.log_params(
            {
                "candidate": name,
                "random_seed": RANDOM_SEED,
                "n_train_rows": len(train.features),
                "n_validation_rows": len(validation.features),
                "n_features": len(spec["input_columns"]),
                "train_fraud_rate": round(float(train.label.mean()), 4),
                "validation_fraud_rate": round(float(validation.label.mean()), 4),
            }
        )
        mlflow.log_dict(dict(spec), "feature_spec.json")
        yield run


def run_trial(
    name: str,
    stage: str,
    index: int,
    model_params: dict[str, Any],
    train: Split,
    validation: Split,
    spec: FeatureSpec,
) -> TrialResult:
    """Melatih satu kombinasi parameter dan mencatatnya sebagai run anak di MLflow.

    Harus dipanggil di dalam `parent_run`.

    Args:
        name: Nama kandidat, salah satu `CANDIDATE_NAMES`.
        stage: Tahap pencarian, salah satu konstanta `STAGE_*`.
        index: Nomor urut percobaan di dalam tahap, hanya untuk penamaan run.
        model_params: Parameter lengkap model, sudah digabung dengan parameter tetapnya.
        train: Split train.
        validation: Split validasi, dipakai untuk skor.
        spec: Kontrak input model.

    Returns:
        Hasil percobaan beserta model terlatihnya.
    """
    run_name = f"{name}-{stage}-{index:02d}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tags({"candidate": name, "search_stage": stage, "run_role": "trial"})
        mlflow.log_params(model_params)
        model = build_model(name, model_params, spec)

        start = time.perf_counter()
        model.fit(train.features, train.label)
        fit_seconds = time.perf_counter() - start

        scores = model.predict_proba(validation.features)[:, 1]
        pr_auc = float(average_precision_score(validation.label, scores))
        mlflow.log_metrics(
            {
                "validation_pr_auc": pr_auc,
                "validation_roc_auc": float(roc_auc_score(validation.label, scores)),
                "fit_seconds": fit_seconds,
            }
        )
        if name == "logistic_regression":
            logistic = model.named_steps["model"]
            n_iter = int(logistic.n_iter_[0])
            # n_iter yang mencapai max_iter berarti belum konvergen, skornya diragukan.
            mlflow.log_metrics({"n_iter": n_iter, "converged": int(n_iter < logistic.max_iter)})
    print(f"  [{name}] {stage} #{index}: PR-AUC {pr_auc:.4f}, {fit_seconds:.0f} detik", flush=True)
    return TrialResult(run_name, model_params, model, pr_auc, fit_seconds)


def log_best(name: str, best: TrialResult, log_model: bool = True) -> None:
    """Mencatat percobaan terbaik di run induk yang sedang aktif.

    Args:
        name: Nama kandidat.
        best: Percobaan terbaik pada tahap ini.
        log_model: Apakah model terlatihnya ikut disimpan sebagai artefak.
    """
    mlflow.log_metric("best_validation_pr_auc", best.validation_pr_auc)
    mlflow.set_tag("best_trial", best.run_name)
    mlflow.log_dict(best.params, "best_params.json")
    if log_model:
        _MODEL_LOGGERS[name](best.model, name="model")


def run_baseline(name: str, train: Split, validation: Split, spec: FeatureSpec) -> TrialResult:
    """Melatih satu kandidat dengan hyperparameter bawaan library sebagai acuan tuning."""
    with parent_run(name, STAGE_BASELINE, train, validation, spec):
        trial = run_trial(
            name, STAGE_BASELINE, 0, OPERATIONAL_PARAMS[name], train, validation, spec
        )
        # Model baseline tidak disimpan, Random Forest bawaan bisa sangat besar.
        log_best(name, trial, log_model=False)
    return trial


def _baseline_gate_warnings(scores: dict[str, float]) -> list[str]:
    """Mencari pembanding yang mengungguli kandidat utama pada skor baseline."""
    comparators = [name for name in scores if name not in MAIN_CANDIDATES]
    return [
        f"{comparator} ({scores[comparator]:.4f}) mengungguli {main} ({scores[main]:.4f})"
        for comparator in comparators
        for main in MAIN_CANDIDATES
        if scores[comparator] > scores[main]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih baseline default empat kandidat tier 1.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATE_NAMES,
        default=list(CANDIDATE_NAMES),
        help="Kandidat yang dilatih, default semuanya.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data dipotong dan dicatat di eksperimen terpisah.",
    )
    args = parser.parse_args()

    setup_mlflow(args.smoke)
    train, validation, spec = load_training_data(args.smoke)

    scores = {}
    for name in args.candidates:
        print(f"Baseline {name} ...", flush=True)
        scores[name] = run_baseline(name, train, validation, spec).validation_pr_auc

    print("\nPeringkat PR-AUC validasi (hyperparameter bawaan):")
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:22s} {score:.4f}")

    if set(scores) == set(CANDIDATE_NAMES):
        warnings = _baseline_gate_warnings(scores)
        if warnings:
            print("\nPERHATIAN, gerbang baseline tidak lolos, diskusikan sebelum phase 1:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("\nGerbang baseline lolos: kedua kandidat utama mengungguli pembanding.")


if __name__ == "__main__":
    main()
