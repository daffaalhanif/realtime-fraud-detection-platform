"""Pemuatan model ONNX dari model registry untuk API scoring.

Model tier 1 yang dimuat selalu versi yang ditandai alias produksi. Model tier 2 dimuat hanya
kalau sebuah versinya ditandai alias shadow: skornya dicatat tanpa ikut menentukan keputusan.
Alias hanya dipindahkan lewat persetujuan manusia di registry, jadi API scoring tidak pernah
memilih versi model sendiri.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
from mlflow import artifacts as mlflow_artifacts
from mlflow.exceptions import MlflowException

from fraud.features.encoding import FeatureSpec
from fraud.features.sequence import SequenceInput, SequenceSpec

TIER1_MODEL_NAME = "fraud-tier1"
PRODUCTION_ALIAS = "production"

TIER2_MODEL_NAME = "fraud-tier2"
SHADOW_ALIAS = "shadow"
GRAY_ZONE_CONFIG_PATH = Path("configs/gray_zone.json")

# Mode sequence yang bisa disusun serving; urutan acak hanya ada sebagai lengan pengujian.
_SERVABLE_TIER2_MODES = ("ordered", "current_only")
_TIER2_INPUT_NAMES = ("numeric", "missing", "categorical", "elapsed", "padding_mask")
_TIER2_OUTPUT_NAME = "logit"

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class Tier2Model:
    """Satu versi model tier 2 yang dipanggil dalam mode shadow.

    Attributes:
        name: Nama model di registry.
        version: Nomor versi model di registry.
        session: Sesi ONNX Runtime yang memuat model.
        sequence_spec: Kontrak elemen sequence dari paket versi ini.
        seq_len: Panjang sequence yang diterima graf, termasuk transaksi yang dinilai.
    """

    name: str
    version: str
    session: ort.InferenceSession
    sequence_spec: SequenceSpec
    seq_len: int

    @property
    def model_version(self) -> str:
        """Penanda versi untuk dicatat bersama skor shadow, format `nama/versi`."""
        return f"{self.name}/{self.version}"

    def logit(self, sequence: SequenceInput) -> float:
        """Logit fraud (belum dikalibrasi) untuk satu transaksi yang dinilai."""
        feeds = {name: getattr(sequence, name) for name in _TIER2_INPUT_NAMES}
        outputs = self.session.run([_TIER2_OUTPUT_NAME], feeds)
        return float(np.asarray(outputs[0])[0])


def _validate_tier2(session: ort.InferenceSession, seq_len: int, mode: str) -> None:
    """Menolak paket tier 2 yang tidak bisa dilayani, supaya gagal saat start-up."""
    if mode not in _SERVABLE_TIER2_MODES:
        raise ValueError(f"Mode sequence {mode!r} tidak bisa dilayani serving.")
    inputs = {item.name: item.shape for item in session.get_inputs()}
    if set(inputs) != set(_TIER2_INPUT_NAMES):
        raise ValueError(f"Input graf tier 2 tidak sesuai: {sorted(inputs)}.")
    if inputs["elapsed"][1] != seq_len:
        raise ValueError(f"Graf tier 2 menerima panjang {inputs['elapsed'][1]}, bukan {seq_len}.")


def load_shadow_tier2() -> Tier2Model | None:
    """Memuat versi model tier 2 yang ditandai alias shadow, kalau ada.

    Membaca alamat registry dari environment variable `MLFLOW_TRACKING_URI`.

    Returns:
        Model tier 2 siap dipanggil, atau None kalau model tier 2 belum terdaftar atau tidak
        ada versi yang ditandai shadow.

    Raises:
        ValueError: Alias produksi ditempelkan ke tier 2 (keputusan oleh tier 2 belum didukung),
            atau isi paket tidak bisa dilayani.
    """
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    try:
        aliases = client.get_registered_model(TIER2_MODEL_NAME).aliases
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    if PRODUCTION_ALIAS in aliases:
        raise ValueError(
            f"Alias {PRODUCTION_ALIAS!r} pada {TIER2_MODEL_NAME} berarti tier 2 ikut memutuskan, "
            "yang butuh kalibrasi dan ambang tier 2 yang belum dibangun. Pakai alias "
            f"{SHADOW_ALIAS!r}."
        )
    version = aliases.get(SHADOW_ALIAS)
    if version is None:
        return None

    with tempfile.TemporaryDirectory() as directory:
        package = Path(
            mlflow_artifacts.download_artifacts(
                f"models:/{TIER2_MODEL_NAME}/{version}",
                dst_path=directory,
                tracking_uri=tracking_uri,
            )
        )
        sequence_spec = json.loads((package / "extra_files" / "sequence_spec.json").read_text())
        export_info = json.loads((package / "extra_files" / "export_info.json").read_text())
        session = _create_session((package / "model.onnx").read_bytes())

    _validate_tier2(session, export_info["seq_len"], export_info["mode"])
    return Tier2Model(
        name=TIER2_MODEL_NAME,
        version=str(version),
        session=session,
        sequence_spec=sequence_spec,
        seq_len=export_info["seq_len"],
    )


def load_gray_zone_lower(
    tier1_model_version: str, path: Path = GRAY_ZONE_CONFIG_PATH
) -> float | None:
    """Ambang bawah zona abu-abu untuk satu versi tier 1.

    Ambang ini melekat pada sebaran skor versi tier 1 tertentu, sehingga versi yang belum
    dianalisis tidak memanggil tier 2 sama sekali.

    Returns:
        Ambang bawah, atau None kalau berkas atau entri versi itu belum ada.
    """
    if not path.exists():
        return None
    entry = json.loads(path.read_text()).get(tier1_model_version)
    return None if entry is None else float(entry["lower"])


@dataclass(frozen=True)
class ShadowTier2:
    """Tier 2 mode shadow beserta batas bawah zona abu-abu untuk versi tier 1 yang aktif.

    Attributes:
        model: Model tier 2 yang dipanggil.
        lower: Skor tier 1 sama dengan atau di atas ini (dan di bawah ambang tolak) memanggil
            tier 2.
    """

    model: Tier2Model
    lower: float


def load_shadow(tier1_model_version: str) -> ShadowTier2 | None:
    """Menyiapkan tier 2 mode shadow untuk satu versi tier 1, kalau keduanya tersedia.

    Returns:
        Tier 2 beserta ambang bawahnya, atau None kalau tidak ada versi tier 2 beralias shadow
        atau ambang bawah untuk versi tier 1 ini belum dianalisis.
    """
    model = load_shadow_tier2()
    if model is None:
        return None
    lower = load_gray_zone_lower(tier1_model_version)
    if lower is None:
        logger.warning(
            "%s beralias shadow, tetapi ambang bawah zona abu-abu untuk %s belum ada; "
            "tier 2 tidak dipanggil",
            model.model_version,
            tier1_model_version,
        )
        return None
    return ShadowTier2(model=model, lower=lower)
