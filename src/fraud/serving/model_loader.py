"""Pemuatan model tier 1 ONNX dari model registry untuk API scoring.

Model yang dimuat selalu versi yang ditandai alias produksi. Alias hanya dipindahkan
lewat persetujuan manusia di registry, jadi API scoring tidak pernah memilih versi
model sendiri.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
from mlflow import artifacts as mlflow_artifacts

from fraud.features.encoding import FeatureSpec

TIER1_MODEL_NAME = "fraud-tier1"
PRODUCTION_ALIAS = "production"

_INPUT_NAME = "input"
_SCORE_OUTPUT_NAME = "calibrated_probability"


@dataclass(frozen=True)
class Tier1Thresholds:
    """Ambang keputusan tier 1 yang tersimpan bersama versi model.

    Attributes:
        reject: Skor sama dengan atau di atas ini ditolak otomatis.
        review: Skor sama dengan atau di atas ini (dan di bawah `reject`) disetujui lalu
            ditandai review.
        lower: Batas bawah zona abu-abu pemanggil tier 2. `None` selama tier 2 belum ada.
    """

    reject: float
    review: float
    lower: float | None


@dataclass(frozen=True)
class Tier1Model:
    """Satu versi model tier 1 yang siap dipakai scoring.

    Objek ini tidak diubah setelah dibuat, sehingga penggantian model cukup dengan menukar
    referensinya: request yang sedang berjalan tetap memakai versi lama sampai selesai.

    Attributes:
        name: Nama model di registry.
        version: Nomor versi model di registry.
        session: Sesi ONNX Runtime yang memuat model.
        thresholds: Ambang keputusan dari paket versi ini.
        feature_spec: Kontrak input model (urutan kolom dan pemetaan kategorikal).
    """

    name: str
    version: str
    session: ort.InferenceSession
    thresholds: Tier1Thresholds
    feature_spec: FeatureSpec

    @property
    def model_version(self) -> str:
        """Penanda versi untuk dicatat di tiap keputusan, format `nama/versi`."""
        return f"{self.name}/{self.version}"

    def score(self, features: np.ndarray) -> float:
        """Menghitung peluang fraud terkalibrasi untuk satu transaksi.

        Args:
            features: Vektor input satu transaksi berbentuk `(1, n_fitur)` bertipe float32,
                urutan kolom mengikuti `feature_spec["input_columns"]`.

        Returns:
            Peluang fraud terkalibrasi.
        """
        outputs = self.session.run([_SCORE_OUTPUT_NAME], {_INPUT_NAME: features})
        return float(np.asarray(outputs[0])[0])


def _create_session(model_bytes: bytes) -> ort.InferenceSession:
    """Membuat sesi ONNX Runtime untuk inferensi satu baris per panggilan."""
    # Telemetri ORT memicu abort saat interpreter ditutup dan melakukan panggilan keluar.
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    # Satu baris per request tidak untung dari paralelisme, dan API berjalan dengan banyak
    # worker; thread default per sesi hanya saling berebut core antar worker.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _validate(
    session: ort.InferenceSession, thresholds: Tier1Thresholds, spec: FeatureSpec
) -> None:
    """Menolak paket model yang tidak konsisten, supaya gagal saat start-up, bukan per request."""
    n_model_inputs = session.get_inputs()[0].shape[1]
    n_spec_columns = len(spec["input_columns"])
    if n_model_inputs != n_spec_columns:
        raise ValueError(
            f"Model menerima {n_model_inputs} kolom, feature_spec memuat {n_spec_columns}."
        )
    if not 0.0 < thresholds.review <= thresholds.reject < 1.0:
        raise ValueError(
            f"Ambang tidak valid: review {thresholds.review}, reject {thresholds.reject}."
        )


def load_production_tier1() -> Tier1Model:
    """Memuat versi model tier 1 yang sedang ditandai alias produksi.

    Membaca alamat registry dari environment variable `MLFLOW_TRACKING_URI`.

    Returns:
        Model tier 1 siap scoring beserta ambang dan kontrak inputnya.

    Raises:
        KeyError: `MLFLOW_TRACKING_URI` tidak diset.
        mlflow.exceptions.MlflowException: Alias produksi belum ditempelkan ke versi mana pun.
        ValueError: Isi paket model tidak konsisten.
    """
    # Tanpa URI eksplisit MLflow diam-diam memakai penyimpanan lokal kosong, sehingga
    # kesalahan konfigurasi baru terlihat sebagai pesan alias tidak ditemukan.
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(TIER1_MODEL_NAME, PRODUCTION_ALIAS).version

    # Diunduh lewat nomor versi hasil resolusi, bukan lewat alias, supaya versi yang dicatat
    # tetap sama dengan isi paket walau alias berpindah di tengah proses unduh.
    with tempfile.TemporaryDirectory() as directory:
        package = Path(
            mlflow_artifacts.download_artifacts(
                f"models:/{TIER1_MODEL_NAME}/{version}",
                dst_path=directory,
                tracking_uri=tracking_uri,
            )
        )
        thresholds_file = json.loads((package / "extra_files" / "thresholds.json").read_text())
        feature_spec = json.loads((package / "extra_files" / "feature_spec.json").read_text())
        session = _create_session((package / "model.onnx").read_bytes())

    thresholds = Tier1Thresholds(**thresholds_file["thresholds"])
    _validate(session, thresholds, feature_spec)
    return Tier1Model(
        name=TIER1_MODEL_NAME,
        version=str(version),
        session=session,
        thresholds=thresholds,
        feature_spec=feature_spec,
    )
