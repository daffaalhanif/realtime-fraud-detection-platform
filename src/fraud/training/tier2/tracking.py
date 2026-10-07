"""Pengaturan bersama untuk seluruh lengan pengujian hipotesis tier 2: eksperimen MLflow dan device.

Pembanding tier 1, pretraining, fine-tuning, uji acak urutan, dan evaluasi mencatat ke satu
eksperimen yang sama, supaya evaluasi cukup membaca satu tempat dan membedakan lengan lewat tag.
"""

import os

import mlflow
import torch
from dotenv import load_dotenv

EXPERIMENT_NAME = "tier2-hypothesis"
SMOKE_EXPERIMENT_NAME = "tier2-hypothesis-smoke"


def setup_mlflow(smoke: bool) -> str:
    """Menyambungkan ke server MLflow dan memilih eksperimen, mengembalikan id eksperimennya.

    Membaca alamat server dari environment variable `MLFLOW_TRACKING_URI`, baik dari `.env`
    di mesin pengembangan maupun yang disetel langsung di notebook Colab.

    Raises:
        KeyError: `MLFLOW_TRACKING_URI` tidak diset.
    """
    load_dotenv()
    # Tanpa URI eksplisit MLflow diam-diam membuat penyimpanan lokal baru di folder kerja.
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment(SMOKE_EXPERIMENT_NAME if smoke else EXPERIMENT_NAME)
    return experiment.experiment_id


def select_device() -> torch.device:
    """Device pelatihan terbaik yang tersedia: CUDA, lalu MPS, lalu CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
