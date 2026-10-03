"""Tata letak offline store (Parquet), dipakai pemuatan awal, training, pemulihan, dan pengarsip.

Offline store terdiri dari dua folder dengan skema file yang sama: data awal dari histori
berlabel, dan arsip transaksi yang diputuskan API scoring. Keduanya dipartisi per hari
relatif lewat folder `txn_day=N/`, sehingga bisa dibaca bersama tanpa penyesuaian.
Arsip dipisah karena labelnya belum diketahui, sedangkan jalur training membaca seluruh isi
folder data awal sebagai data berlabel.

Modul ini hanya mengatur format, tanpa membaca atau menulis berkas.
"""

import types
from pathlib import Path
from typing import get_args

import pyarrow as pa

from fraud.schemas.transaction import Transaction

INITIAL_PARQUET_DIR = Path("data/processed/transactions")
ARCHIVE_PARQUET_DIR = Path("data/processed/scored")

PARTITION_COLUMN = "txn_day"
LABEL_COLUMN = "isFraud"

# TransactionDT adalah detik sejak titik acuan dataset, bukan timestamp kalender, jadi
# partisinya indeks hari relatif.
SECONDS_PER_DAY = 86_400


def _numbered(prefix: str, last: int, width: int = 1) -> list[str]:
    return [f"{prefix}{number:0{width}d}" for number in range(1, last + 1)]


# Urutan kolom file data awal mengikuti CSV asal (transaksi lalu identitas), berbeda dari
# urutan field skema Transaction; arsip memakai urutan yang sama supaya skemanya identik.
COLUMN_ORDER = (
    "TransactionID",
    LABEL_COLUMN,
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    *_numbered("card", 6),
    *_numbered("addr", 2),
    *_numbered("dist", 2),
    "P_emaildomain",
    "R_emaildomain",
    *_numbered("C", 14),
    *_numbered("D", 15),
    *_numbered("M", 9),
    *_numbered("V", 339),
    *_numbered("id_", 38, width=2),
    "DeviceType",
    "DeviceInfo",
)

_ARROW_TYPES: dict[object, pa.DataType] = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.large_string(),
}


def _arrow_type(annotation: object) -> pa.DataType:
    """Tipe Arrow dari anotasi field Transaction, misal `float | None` menjadi float64."""
    members = get_args(annotation) if isinstance(annotation, types.UnionType) else (annotation,)
    (base,) = [member for member in members if member is not type(None)]
    return _ARROW_TYPES[base]


def _build_schema() -> pa.Schema:
    expected = set(Transaction.model_fields) | {LABEL_COLUMN}
    if len(COLUMN_ORDER) != len(expected) or set(COLUMN_ORDER) != expected:
        raise RuntimeError(
            "Urutan kolom Parquet tidak lagi sesuai field Transaction: "
            f"kurang {sorted(expected - set(COLUMN_ORDER))}, "
            f"lebih {sorted(set(COLUMN_ORDER) - expected)}"
        )
    field_types = {
        name: _arrow_type(field.annotation) for name, field in Transaction.model_fields.items()
    }
    # Label biner 0/1 di data awal; di arsip selalu kosong karena belum diketahui saat serving.
    field_types[LABEL_COLUMN] = pa.int64()
    return pa.schema([pa.field(name, field_types[name]) for name in COLUMN_ORDER])


PARQUET_SCHEMA = _build_schema()


def partition_day(transaction_dt: int) -> int:
    """Indeks hari relatif tempat transaksi disimpan."""
    return transaction_dt // SECONDS_PER_DAY


def partition_dir(root: Path, day: int) -> Path:
    """Folder partisi satu hari di bawah folder offline store `root`."""
    return root / f"{PARTITION_COLUMN}={day}"


def transactions_to_table(transactions: list[Transaction]) -> pa.Table:
    """Menyusun transaksi tanpa label menjadi tabel Arrow berskema offline store.

    Args:
        transactions: Transaksi mentah; kolom label diisi kosong.

    Returns:
        Tabel dengan kolom dan urutan `PARQUET_SCHEMA`, tanpa kolom partisi.
    """
    return pa.Table.from_pylist(
        [transaction.model_dump() for transaction in transactions], schema=PARQUET_SCHEMA
    )
