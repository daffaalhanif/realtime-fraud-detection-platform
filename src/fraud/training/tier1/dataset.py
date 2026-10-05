"""Penyiapan data pelatihan tier 1: fitur agregat per baris, encoding, dan split temporal.

Modul ini dipakai bersama oleh seluruh langkah pelatihan tier 1 (kandidat, kalibrasi,
ambang, ekspor, evaluasi), supaya semuanya memakai definisi fitur dan pembagian data
yang identik. Fitur agregat dihitung lewat `fraud.features.aggregate.process_transaction`,
fungsi yang sama dengan jalur scoring real-time, bukan implementasi kedua.
"""

import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, get_origin

import pandas as pd

from fraud.features.aggregate import (
    AggregateFeatures,
    AggregateState,
    process_transaction,
)
from fraud.features.cold_start import get_default_state
from fraud.features.encoding import (
    FeatureSpec,
    build_model_input,
    fit_category_mappings,
)
from fraud.features.offline_store import INITIAL_PARQUET_DIR, LABEL_COLUMN, PARTITION_COLUMN
from fraud.features.online_store import SEQUENCE_RETENTION_LIMIT
from fraud.features.window import WindowFeatures, window_features
from fraud.schemas.transaction import Transaction

# Pengenal, waktu mentah, dan kunci entitas (perilakunya sudah diwakili fitur agregat).
EXCLUDED_RAW_COLUMNS = ("TransactionID", "TransactionDT", "card1")

AGGREGATE_FEATURE_NAMES = list(AggregateFeatures.__annotations__)

WINDOW_FEATURE_NAMES = list(WindowFeatures.__annotations__)

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


def load_transactions(parquet_dir: Path = INITIAL_PARQUET_DIR) -> pd.DataFrame:
    """Membaca Parquet dan mengurutkannya kronologis.

    Args:
        parquet_dir: Folder Parquet partisi per hari hasil pemuatan data awal.

    Returns:
        Seluruh transaksi urut `TransactionDT` naik, tanpa kolom partisi `txn_day`.
    """
    df = pd.read_parquet(parquet_dir).drop(columns=[PARTITION_COLUMN])
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


def add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur jendela waktu per baris, dihitung hanya dari transaksi sebelumnya.

    Args:
        df: Transaksi urut kronologis (hasil `load_transactions`).

    Returns:
        DataFrame baru dengan satu kolom tambahan per nama di `WINDOW_FEATURE_NAMES`.
    """
    # Histori lebih panjang dari batas retensi tidak pernah dibaca `window_features`.
    histories: dict[int, deque[tuple[int, float]]] = {}
    columns: dict[str, list] = {name: [] for name in WINDOW_FEATURE_NAMES}

    rows = zip(
        df["card1"].to_numpy(),
        df["TransactionAmt"].to_numpy(),
        df["TransactionDT"].to_numpy(),
    )
    for card1, amt, dt in rows:
        history = histories.setdefault(int(card1), deque(maxlen=SEQUENCE_RETENTION_LIMIT))
        features = window_features(history, int(dt))
        history.append((int(dt), float(amt)))
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
    matrix = build_model_input(
        {column: part[column].to_numpy() for column in spec["input_columns"]}, spec
    )
    return Split(
        features=pd.DataFrame(matrix, columns=spec["input_columns"], index=part.index),
        label=part[LABEL_COLUMN].astype("int8"),
        entity_key=part[["TransactionID", "card1"]],
    )


def prepare_datasets(
    parquet_dir: Path = INITIAL_PARQUET_DIR, with_window_features: bool = False
) -> PreparedData:
    """Menyiapkan data pelatihan tier 1 dari Parquet sampai siap dipakai model.

    Pemetaan kategorikal dilatih hanya dari bagian train, lalu diterapkan ke ketiga
    bagian, sehingga validasi dan uji tidak ikut memengaruhi encoding.

    Args:
        parquet_dir: Folder Parquet partisi per hari hasil pemuatan data awal.
        with_window_features: True untuk menambahkan fitur jendela waktu tepat setelah fitur
            agregat. Bawaannya False, kontrak input model produksi versi 1.

    Returns:
        Tiga split dan `FeatureSpec` yang menjelaskan kolom serta pemetaan inputnya.
    """
    df = add_aggregate_features(load_transactions(parquet_dir))
    engineered_columns = list(AGGREGATE_FEATURE_NAMES)
    if with_window_features:
        df = add_window_features(df)
        engineered_columns += WINDOW_FEATURE_NAMES
    train_part, validation_part, test_part = split_temporal(df)

    encoded_columns = _string_columns()
    spec = FeatureSpec(
        input_columns=engineered_columns + _raw_input_columns(),
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
