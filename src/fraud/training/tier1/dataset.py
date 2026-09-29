"""Penyiapan data pelatihan tier 1: fitur agregat per baris, encoding, dan split temporal.

Modul ini dipakai bersama oleh seluruh langkah pelatihan tier 1 (kandidat, kalibrasi,
ambang, ekspor, evaluasi), supaya semuanya memakai definisi fitur dan pembagian data
yang identik. Fitur agregat dihitung lewat `fraud.features.aggregate.process_transaction`,
fungsi yang sama dengan jalur scoring real-time, bukan implementasi kedua.
"""

import types
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, get_args, get_origin

import pandas as pd

from fraud.features.aggregate import (
    AggregateFeatures,
    AggregateState,
    process_transaction,
)
from fraud.features.cold_start import get_default_state
from fraud.features.encoding import (
    CategoryMappings,
    apply_category_mappings,
    fit_category_mappings,
)
from fraud.schemas.transaction import Transaction

PARQUET_DIR = Path("data/processed/transactions")

LABEL_COLUMN = "isFraud"

# Pengenal, waktu mentah, dan kunci entitas (perilakunya sudah diwakili fitur agregat).
EXCLUDED_RAW_COLUMNS = ("TransactionID", "TransactionDT", "card1")

AGGREGATE_FEATURE_NAMES = list(AggregateFeatures.__annotations__)

# Kode nominal yang tersimpan numerik: nilainya pengenal, bukan besaran yang punya urutan.
NUMERIC_CATEGORICAL_COLUMNS = (
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "id_13",
    "id_14",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_24",
    "id_25",
    "id_26",
    "id_32",
)

# Batas split berdasarkan posisi baris setelah data urut kronologis: 60% train, 20% validasi.
TRAIN_END_FRACTION = 0.6
VALIDATION_END_FRACTION = 0.8


class FeatureSpec(TypedDict):
    """Kontrak input model, disimpan sebagai artefak bersama model.

    Attributes:
        input_columns: Urutan kolom yang diterima model, fitur agregat lebih dulu.
        encoded_columns: Kolom di `input_columns` yang semula string, kini kode ordinal.
        nominal_numeric_columns: Kolom di `input_columns` yang nominal tapi tetap bernilai
            asli. Model pohon boleh memakainya apa adanya, model linear perlu one-hot.
        category_mappings: Pemetaan nilai ke kode untuk tiap kolom di `encoded_columns`.
    """

    input_columns: list[str]
    encoded_columns: list[str]
    nominal_numeric_columns: list[str]
    category_mappings: CategoryMappings


@dataclass(frozen=True)
class Split:
    """Satu bagian data hasil split temporal.

    Attributes:
        features: Matriks input model, seluruhnya float32 dengan urutan kolom `FeatureSpec`.
        label: Label fraud (0 atau 1) sejajar dengan baris `features`.
        entity_key: `TransactionID` dan `card1` tiap baris, untuk evaluasi per segmen.
    """

    features: pd.DataFrame
    label: pd.Series
    entity_key: pd.DataFrame


@dataclass(frozen=True)
class PreparedData:
    """Data pelatihan tier 1 yang siap dipakai, beserta kontrak inputnya."""

    train: Split
    validation: Split
    test: Split
    spec: FeatureSpec


def load_transactions(parquet_dir: Path = PARQUET_DIR) -> pd.DataFrame:
    """Membaca Parquet dan mengurutkannya kronologis.

    Args:
        parquet_dir: Folder Parquet partisi per hari hasil pemuatan data awal.

    Returns:
        Seluruh transaksi urut `TransactionDT` naik, tanpa kolom partisi `txn_day`.
    """
    df = pd.read_parquet(parquet_dir).drop(columns=["txn_day"])
    # Baca Parquet berpartisi tidak menjamin urutan kronologis, sedangkan split dan fitur
    # agregat bergantung pada urutan. TransactionID memutus seri TransactionDT yang sama.
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur agregat per baris, dihitung hanya dari transaksi sebelumnya.

    Args:
        df: Transaksi urut kronologis (hasil `load_transactions`).

    Returns:
        DataFrame baru dengan satu kolom tambahan per nama di `AGGREGATE_FEATURE_NAMES`.
    """
    states: dict[int, AggregateState] = {}
    columns: dict[str, list] = {name: [] for name in AGGREGATE_FEATURE_NAMES}

    rows = zip(
        df["card1"].to_numpy(),
        df["TransactionAmt"].to_numpy(),
        df["TransactionDT"].to_numpy(),
    )
    for card1, amt, dt in rows:
        entity = int(card1)
        state = states.get(entity)
        if state is None:
            state = get_default_state()
        features, states[entity] = process_transaction(state, float(amt), int(dt))
        for name, value in features.items():
            columns[name].append(value)

    return df.assign(**columns)


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Memotong data urut kronologis menjadi train, validasi, dan uji (60-20-20).

    Args:
        df: Transaksi urut kronologis.

    Returns:
        Tiga bagian berurutan waktu: 60% pertama, 20% berikutnya, 20% terakhir.
    """
    train_end = int(len(df) * TRAIN_END_FRACTION)
    validation_end = int(len(df) * VALIDATION_END_FRACTION)
    return df.iloc[:train_end], df.iloc[train_end:validation_end], df.iloc[validation_end:]


def _is_str_annotation(annotation: object) -> bool:
    """Cek apakah anotasi tipe field pydantic memuat `str`."""
    if annotation is str:
        return True
    return get_origin(annotation) is types.UnionType and str in get_args(annotation)


def _string_columns() -> list[str]:
    """Kolom bertipe string menurut skema Transaction, wajib di-encode sebelum masuk model.

    Ini tidak mencakup seluruh kolom kategorikal: yang tersimpan numerik ada di
    `NUMERIC_CATEGORICAL_COLUMNS`. Skema dipakai sebagai sumber tipe, bukan dtype pandas.
    """
    return [
        name
        for name, field in Transaction.model_fields.items()
        if name not in EXCLUDED_RAW_COLUMNS and _is_str_annotation(field.annotation)
    ]


def _raw_input_columns() -> list[str]:
    """Kolom mentah yang menjadi input model, urutan mengikuti skema Transaction."""
    return [name for name in Transaction.model_fields if name not in EXCLUDED_RAW_COLUMNS]


def _to_split(part: pd.DataFrame, spec: FeatureSpec) -> Split:
    """Mengubah satu bagian data menjadi matriks input float32, label, dan kunci entitas."""
    encoded = apply_category_mappings(part[spec["input_columns"]], spec["category_mappings"])
    return Split(
        features=encoded.astype("float32"),
        label=part[LABEL_COLUMN].astype("int8"),
        entity_key=part[["TransactionID", "card1"]],
    )


def prepare_datasets(parquet_dir: Path = PARQUET_DIR) -> PreparedData:
    """Menyiapkan data pelatihan tier 1 dari Parquet sampai siap dipakai model.

    Pemetaan kategorikal dilatih hanya dari bagian train, lalu diterapkan ke ketiga
    bagian, sehingga validasi dan uji tidak ikut memengaruhi encoding.

    Args:
        parquet_dir: Folder Parquet partisi per hari hasil pemuatan data awal.

    Returns:
        Tiga split dan `FeatureSpec` yang menjelaskan kolom serta pemetaan inputnya.
    """
    df = add_aggregate_features(load_transactions(parquet_dir))
    train_part, validation_part, test_part = split_temporal(df)

    encoded_columns = _string_columns()
    spec = FeatureSpec(
        input_columns=AGGREGATE_FEATURE_NAMES + _raw_input_columns(),
        encoded_columns=encoded_columns,
        nominal_numeric_columns=list(NUMERIC_CATEGORICAL_COLUMNS),
        category_mappings=fit_category_mappings(train_part, encoded_columns),
    )
    return PreparedData(
        train=_to_split(train_part, spec),
        validation=_to_split(validation_part, spec),
        test=_to_split(test_part, spec),
        spec=spec,
    )
