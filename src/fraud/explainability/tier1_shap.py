"""Penjelasan TreeSHAP tier 1 untuk satu transaksi, dihitung di consumer narasi.

Model yang melayani scoring berupa graf ONNX, sedangkan TreeSHAP butuh struktur pohon aslinya.
Booster LightGBM dimuat dari model sumber versi `fraud-tier1` yang tercatat pesan, lewat
metadata di registry: versi model, run kalibrasi sumbernya, run model sumbernya, lalu model
LightGBM yang tercatat di run itu. Input disusun dari transaksi dan fitur rekayasa di pesan
dengan `build_model_input`, fungsi yang sama dengan scoring, sehingga baris yang dijelaskan
identik dengan baris yang dinilai.

Kontribusi dihitung pada skala logit skor mentah model, sebelum kalibrasi isotonic. Kolom V yang
saling berkorelasi tinggi dijumlahkan per kelompok supaya pengaruh gabungannya tidak tampak
terpecah menjadi baris-baris kecil. Jumlah kontribusi tidak persis sama dengan skor mentah, jadi
selisihnya dilaporkan sebagai bagian tersendiri, bukan disembunyikan.
"""

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# lightgbm wajib diimpor sebelum sklearn (dimuat skops saat model dibuka). Urutan terbalik
# membuat proses segfault karena dua runtime OpenMP saling bentrok.
import lightgbm
import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import numpy as np
from mlflow import artifacts as mlflow_artifacts
from mlflow.models import Model

from fraud.features.encoding import FeatureSpec, build_model_input
from fraud.schemas.scored_message import ScoredMessage
from fraud.schemas.transaction import Transaction

FEATURE_GROUPS_PATH = Path("configs/feature_groups.json")

# Metadata yang ditulis ekspor ONNX ke tiap versi `fraud-tier1`.
CALIBRATION_RUN_METADATA_KEY = "source_calibration_run_id"

# Jumlah baris kontribusi terbesar yang disimpan; sisanya dijumlahkan jadi satu bagian.
TOP_CONTRIBUTIONS = 10

# Model sebesar ini lambat per transaksi, jadi beberapa thread dipakai supaya satu batch
# tidak mendekati batas waktu poll consumer.
SHAP_THREADS = 4

UNAVAILABLE_REASON = "Kontribusi fitur tidak tersedia untuk item ini."

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureGroups:
    """Kelompok fitur yang dilaporkan sebagai satu baris.

    Attributes:
        group_of: Id kelompok untuk tiap kolom anggota kelompok.
        members: Daftar kolom anggota untuk tiap id kelompok.
    """

    group_of: dict[str, str]
    members: dict[str, list[str]]


@dataclass(frozen=True)
class Tier1ExplainerModel:
    """Model LightGBM satu versi `fraud-tier1` beserta kontrak inputnya.

    Attributes:
        model_version: Penanda versi format `nama/versi`, sama dengan di pesan.
        booster: Booster LightGBM yang menjadi sumber model di dalam graf ONNX.
        feature_spec: Kontrak input versi ini.
    """

    model_version: str
    booster: lightgbm.Booster
    feature_spec: FeatureSpec


def load_feature_groups(path: Path = FEATURE_GROUPS_PATH) -> FeatureGroups:
    """Membaca kelompok fitur dari berkas hasil `fraud.training.tier1.explainer`."""
    document = json.loads(path.read_text())
    group_of: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for group in document["groups"]:
        members[group["id"]] = list(group["members"])
        for column in group["members"]:
            group_of[column] = group["id"]
    return FeatureGroups(group_of=group_of, members=members)


def load_explainer_model(model_version: str) -> Tier1ExplainerModel:
    """Memuat booster LightGBM sumber dari satu versi `fraud-tier1`.

    Membaca alamat registry dari environment variable `MLFLOW_TRACKING_URI`.

    Args:
        model_version: Penanda versi format `nama/versi`, misal `fraud-tier1/1`.

    Raises:
        KeyError: `MLFLOW_TRACKING_URI` tidak diset, atau metadata sumber tidak ada.
        ValueError: Kontrak input tidak cocok dengan jumlah fitur booster.
        mlflow.exceptions.MlflowException: Versi, run, atau model sumber tidak ditemukan.
    """
    name, _, version = model_version.partition("/")
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    # Pencarian dan pemuatan model tercatat memakai tracking URI global, bukan klien di atas.
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    # Hanya dua berkas kecil yang diunduh; paket lengkapnya memuat graf ONNX ratusan megabyte.
    with tempfile.TemporaryDirectory() as directory:
        package_model = Model.load(
            mlflow_artifacts.download_artifacts(
                f"models:/{name}/{version}/MLmodel", dst_path=directory
            )
        )
        feature_spec: FeatureSpec = json.loads(
            Path(
                mlflow_artifacts.download_artifacts(
                    f"models:/{name}/{version}/extra_files/feature_spec.json", dst_path=directory
                )
            ).read_text()
        )

    calibration_run = client.get_run((package_model.metadata or {})[CALIBRATION_RUN_METADATA_KEY])
    source_run = client.get_run(calibration_run.data.params["source_run_id"])
    logged_models = mlflow.search_logged_models(
        experiment_ids=[source_run.info.experiment_id],
        filter_string=f"source_run_id = '{source_run.info.run_id}'",
        output_format="list",
    )
    if not logged_models:
        raise RuntimeError(f"Run model sumber {source_run.info.run_id} tidak mencatat model.")
    estimator = mlflow_lightgbm.load_model(f"models:/{logged_models[0].model_id}")
    if not isinstance(estimator, lightgbm.LGBMClassifier):
        raise TypeError(f"Model sumber {model_version} bukan LGBMClassifier.")

    booster = estimator.booster_
    if booster.num_feature() != len(feature_spec["input_columns"]):
        raise ValueError(
            f"Booster {model_version} menerima {booster.num_feature()} fitur, kontrak input "
            f"memuat {len(feature_spec['input_columns'])}."
        )
    return Tier1ExplainerModel(model_version, booster, feature_spec)


def _json_value(value: Any) -> Any:
    """Nilai fitur yang aman ditulis ke JSON: nilai tak berhingga dianggap kosong."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ensure_finite(explanation: dict[str, Any]) -> None:
    """Menolak dokumen berisi angka tak berhingga atau NaN.

    JSON standar tidak punya representasi untuk nilai itu, sehingga penulisan ke JSONB gagal dan
    consumer berhenti di item yang sama berulang kali.

    Raises:
        ValueError: Skor, nilai dasar, selisih, atau kontribusi mana pun tidak berhingga.
    """
    numbers = [
        explanation["raw_score"],
        explanation["base_value"],
        explanation["unexplained"],
        explanation["other"]["contribution"],
        *(entry["contribution"] for entry in explanation["contributions"]),
    ]
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError("Penjelasan memuat angka tak berhingga atau NaN.")


def unavailable_explanation() -> dict[str, Any]:
    """Penjelasan pengganti saat kontribusi tidak bisa dihitung untuk sebuah item."""
    return {"available": False, "reason": UNAVAILABLE_REASON}


def build_explanation(
    model: Tier1ExplainerModel,
    groups: FeatureGroups,
    transaction: Transaction,
    engineered_features: dict[str, float],
) -> dict[str, Any]:
    """Menghitung penjelasan TreeSHAP untuk satu transaksi.

    Args:
        model: Booster dan kontrak input versi model yang menilai transaksi.
        groups: Kelompok kolom yang dilaporkan sebagai satu baris.
        transaction: Transaksi mentah dari pesan.
        engineered_features: Fitur rekayasa dari pesan, persis yang masuk ke model.

    Returns:
        Dokumen JSON penjelasan: skor mentah (logit), nilai dasar, kontribusi terbesar berupa
        fitur tunggal atau kelompok, jumlah kontribusi sisanya, dan selisih yang tidak
        terjelaskan. Nilai dasar, kontribusi seluruhnya, dan selisih berjumlah skor mentah.

    Raises:
        ValueError: Pesan tidak memuat salah satu kolom input model, atau hasil perhitungan
            memuat angka tak berhingga atau NaN.
    """
    spec = model.feature_spec
    values: dict[str, Any] = transaction.model_dump()
    values.update(engineered_features)
    missing = [name for name in spec["input_columns"] if name not in values]
    if missing:
        raise ValueError(f"Kolom input tidak ada di pesan: {missing[:5]}")

    matrix = build_model_input({name: [values[name]] for name in spec["input_columns"]}, spec)
    contributions = np.asarray(
        model.booster.predict(matrix, pred_contrib=True, num_threads=SHAP_THREADS)
    )
    raw_score = float(np.asarray(model.booster.predict(matrix, raw_score=True))[0])
    base_value = float(contributions[0, -1])

    totals: dict[str, float] = {}
    for name, contribution in zip(spec["input_columns"], contributions[0, :-1], strict=True):
        unit = groups.group_of.get(name, name)
        totals[unit] = totals.get(unit, 0.0) + float(contribution)
    ranked = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)
    top, rest = ranked[:TOP_CONTRIBUTIONS], ranked[TOP_CONTRIBUTIONS:]

    entries: list[dict[str, Any]] = []
    for unit, contribution in top:
        if unit in groups.members:
            entries.append(
                {
                    "name": unit,
                    "kind": "group",
                    "members": groups.members[unit],
                    "contribution": contribution,
                }
            )
        else:
            entries.append(
                {
                    "name": unit,
                    "kind": "feature",
                    "value": _json_value(values[unit]),
                    "contribution": contribution,
                }
            )

    explanation = {
        "available": True,
        "method": "treeshap_lightgbm",
        "model_version": model.model_version,
        "scale": "logit",
        "raw_score": raw_score,
        "base_value": base_value,
        "contributions": entries,
        "other": {
            "n_units": len(rest),
            "contribution": math.fsum(contribution for _, contribution in rest),
        },
        "unexplained": raw_score - base_value - math.fsum(totals.values()),
    }
    _ensure_finite(explanation)
    return explanation


class Tier1Explainer:
    """Penghitung penjelasan SHAP yang memuat model tiap versi sekali, saat pertama dibutuhkan.

    Args:
        groups: Kelompok kolom yang dilaporkan sebagai satu baris.
    """

    def __init__(self, groups: FeatureGroups) -> None:
        self._groups = groups
        self._models: dict[str, Tier1ExplainerModel] = {}

    def explain(self, message: ScoredMessage) -> dict[str, Any]:
        """Penjelasan untuk satu pesan; tidak pernah melempar exception.

        Kegagalan di sini harus dilaporkan sebagai penjelasan tidak tersedia, bukan dilempar:
        consumer yang berhenti tanpa commit akan membaca ulang pesan yang sama dan berhenti
        lagi, sehingga satu item bermasalah menahan seluruh antrean.
        """
        if message.engineered_features is None:
            return unavailable_explanation()
        try:
            model = self._models.get(message.model_version)
            if model is None:
                model = load_explainer_model(message.model_version)
                self._models[message.model_version] = model
            return build_explanation(
                model, self._groups, message.transaction, message.engineered_features
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Penjelasan SHAP gagal untuk TransactionID %s (%s)",
                message.transaction.TransactionID,
                message.model_version,
            )
            return unavailable_explanation()
