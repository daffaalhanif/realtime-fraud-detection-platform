"""Skrip pemuatan data awal: bangun Parquet dan Redis dari histori penuh.

Baca train_transaction.csv dan train_identity.csv, gabungkan, validasi
tiap baris terhadap skema Transaction, lalu proses transaksi per entitas
secara kronologis lewat fraud.features.aggregate.process_transaction,
fungsi yang sama dipakai jalur incremental saat serving nanti. Hasil
akhirnya dua penyimpanan:

- Parquet, partisi per hari: salinan data mentah hasil join, sumber untuk
  pelatihan model dan pemulihan Redis.
- Redis: HASH berisi state agregat terakhir tiap entitas, LIST berisi
  histori transaksi mentah terbaru tiap entitas (input tier 2).

Dijalankan lewat: uv run python scripts/load_initial_data.py
"""

import os
import types
from pathlib import Path
from typing import get_args, get_origin

import pandas as pd
import redis
from dotenv import load_dotenv
from pydantic import TypeAdapter

from fraud.schemas.transaction import Transaction
from redis_batch import compute_features_and_sequences, write_redis

DATA_DIR = Path("data")
PARQUET_OUTPUT_DIR = Path("data/processed/transactions")

SECONDS_PER_DAY = 86400


def _is_str_annotation(annotation: object) -> bool:
    """Cek apakah anotasi tipe field pydantic memuat `str`."""
    if annotation is str:
        return True
    return get_origin(annotation) is types.UnionType and str in get_args(
        annotation
    )


def _string_field_columns(df: pd.DataFrame) -> list[str]:
    """Kolom string menurut skema Transaction, bukan dtype pandas.

    Dtype yang pandas infer dari CSV tidak bisa diandalkan untuk kolom
    yang kebetulan kosong semua (jadi float64, bukan string) - skema
    Transaction adalah sumber kebenaran tipe tiap field.
    """
    return [
        name
        for name, field in Transaction.model_fields.items()
        if _is_str_annotation(field.annotation) and name in df.columns
    ]


def load_and_join() -> pd.DataFrame:
    """Baca dan gabungkan dua CSV histori transaksi, urut kronologis."""
    df_transaction = pd.read_csv(DATA_DIR / "train_transaction.csv")
    df_identity = pd.read_csv(DATA_DIR / "train_identity.csv")
    df = df_transaction.merge(df_identity, on="TransactionID", how="left")
    # TransactionID sebagai tie-breaker: banyak baris berbagi TransactionDT
    # yang sama persis, dan TransactionID terbukti selalu naik mengikuti
    # urutan kronologis asli, jadi urutan hasil sort jadi deterministik.
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )


def validate_rows(df: pd.DataFrame) -> None:
    """Validasi seluruh baris terhadap skema Transaction.

    Args:
        df: Data hasil join, belum diubah bentuknya.

    Raises:
        pydantic.ValidationError: Kalau ada baris yang tidak sesuai skema.
    """
    string_columns = _string_field_columns(df)
    validation_df = df.copy()
    validation_df[string_columns] = validation_df[string_columns].astype(
        object
    )
    validation_df[string_columns] = validation_df[string_columns].where(
        validation_df[string_columns].notna(), None
    )
    records = validation_df.to_dict(orient="records")
    TypeAdapter(list[Transaction]).validate_python(records)


def write_parquet(df: pd.DataFrame, output_dir: Path) -> None:
    """Tulis data mentah hasil join ke Parquet, partisi per hari.

    Tidak ada tanggal kalender asli pada dataset ini (TransactionDT
    adalah detik berlalu sejak titik acuan, bukan timestamp absolut),
    jadi partisinya berupa indeks hari relatif (hari ke berapa sejak
    data mulai tercatat), bukan tanggal kalender sungguhan.
    """
    df = df.copy()
    df["txn_day"] = df["TransactionDT"] // SECONDS_PER_DAY
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir, partition_cols=["txn_day"], index=False)


def main() -> None:
    load_dotenv()
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=True,
    )

    print("Membaca dan menggabungkan data...")
    df = load_and_join()
    print(f"Total baris: {len(df)}")

    print("Memvalidasi seluruh baris terhadap skema Transaction...")
    validate_rows(df)
    print("Validasi berhasil.")

    print("Menghitung fitur agregat dan sequence per entitas...")
    states, sequences = compute_features_and_sequences(df)
    print(f"Total entitas: {len(states)}")

    print("Menulis Parquet...")
    write_parquet(df, PARQUET_OUTPUT_DIR)

    print("Menulis Redis...")
    write_redis(states, sequences, redis_client)

    print("Selesai.")


if __name__ == "__main__":
    main()
