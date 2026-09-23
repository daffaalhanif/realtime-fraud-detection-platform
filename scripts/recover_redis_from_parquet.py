"""Skrip pemulihan Redis dari Parquet, dijalankan tanpa menyentuh CSV mentah.

Redis diperlakukan sebagai cache yang bisa dihapus dan dibangun ulang
kapan saja tanpa kehilangan data secara permanen, karena Parquet sudah
menyimpan histori mentah lengkap dan fungsi fitur tunggal memastikan hasil
perhitungan ulang identik dengan hasil incremental aslinya. Skrip ini juga
dipakai untuk menyiapkan tumpukan uji beban dari kondisi Redis kosong.

Dijalankan lewat: uv run python scripts/recover_redis_from_parquet.py
"""

import os
from pathlib import Path

import pandas as pd
import redis
from dotenv import load_dotenv

from redis_batch import compute_features_and_sequences, write_redis

PARQUET_DIR = Path("data/processed/transactions")

# Kolom partisi Parquet, bukan bagian data mentah hasil join - harus
# dibuang sebelum diproses supaya sequence yang ditulis ulang identik
# dengan hasil pemuatan awal, yang tidak pernah punya kolom ini.
PARTITION_COLUMN = "txn_day"


def load_from_parquet(parquet_dir: Path) -> pd.DataFrame:
    """Baca kembali data mentah dari Parquet, urut kronologis.

    Args:
        parquet_dir: Direktori Parquet hasil scripts/load_initial_data.py,
            partisi per hari.

    Returns:
        Data mentah hasil join, identik strukturnya dengan yang diproses
        scripts/load_initial_data.py sebelum ditulis ke Parquet.
    """
    df = pd.read_parquet(parquet_dir)
    df = df.drop(columns=[PARTITION_COLUMN])
    # TransactionID sebagai tie-breaker: banyak baris berbagi TransactionDT
    # yang sama persis, dan TransactionID terbukti selalu naik mengikuti
    # urutan kronologis asli, jadi urutan hasil sort jadi deterministik.
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(
        drop=True
    )


def main() -> None:
    load_dotenv()
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=True,
    )

    print("Membaca data mentah dari Parquet...")
    df = load_from_parquet(PARQUET_DIR)
    print(f"Total baris: {len(df)}")

    print("Menghitung ulang fitur agregat dan sequence per entitas...")
    states, sequences = compute_features_and_sequences(df)
    print(f"Total entitas: {len(states)}")

    print("Menulis ulang Redis...")
    write_redis(states, sequences, redis_client)

    print("Pemulihan selesai.")


if __name__ == "__main__":
    main()
