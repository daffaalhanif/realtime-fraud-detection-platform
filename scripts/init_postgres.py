"""Skrip pembuatan skema dan tabel Postgres, aman dijalankan berulang.

Satu database menampung tiga keperluan dengan skema terpisah: `audit` untuk catatan
keputusan, `review` untuk antrean review analis, dan `mlflow` untuk model registry. Tabel di
skema `mlflow` dikelola MLflow sendiri; skrip ini hanya memastikan skemanya ada.

Seluruh perintah berjalan dalam satu transaksi, jadi kegagalan di tengah tidak meninggalkan
skema setengah jadi. Tabel yang sudah ada tidak diubah.

Dijalankan lewat: uv run python scripts/init_postgres.py
"""

import os

from dotenv import load_dotenv

from fraud.consumers.base import connect_postgres

_DDL = """
CREATE SCHEMA IF NOT EXISTS mlflow;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS review;

CREATE TABLE IF NOT EXISTS audit.decisions (
    -- Primary key sekaligus penyaring pesan ganda dari Kafka.
    transaction_id BIGINT PRIMARY KEY,
    decision TEXT NOT NULL CHECK (decision IN ('decline', 'approve_review', 'approve')),
    tier1_score DOUBLE PRECISION NOT NULL,
    -- Logit tier 2 mode shadow, belum dikalibrasi; kosong di luar zona abu-abu.
    tier2_score DOUBLE PRECISION,
    model_version TEXT NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL,
    -- Jam database saat baris ditulis; selisihnya dengan scored_at adalah lag consumer.
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    transaction JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_scored_at_idx ON audit.decisions (scored_at);

-- Tanpa foreign key ke audit.decisions: consumer narasi dan consumer audit berjalan
-- independen, jadi baris audit belum tentu ada saat item antrean ditulis.
CREATE TABLE IF NOT EXISTS review.queue (
    transaction_id BIGINT PRIMARY KEY,
    tier1_score DOUBLE PRECISION NOT NULL,
    tier2_score DOUBLE PRECISION,
    model_version TEXT NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    transaction JSONB NOT NULL,
    reviewed_at TIMESTAMPTZ,
    reviewed_by TEXT,
    review_outcome TEXT CHECK (review_outcome IN ('fraud', 'legitimate')),
    -- Item hanya bisa terbuka sepenuhnya atau selesai lengkap dengan hasil dan peninjaunya.
    CHECK (
        (reviewed_at IS NULL) = (reviewed_by IS NULL)
        AND (reviewed_at IS NULL) = (review_outcome IS NULL)
    )
);
-- Hanya item terbuka, supaya query backlog tetap murah meski item selesai terus menumpuk.
CREATE INDEX IF NOT EXISTS queue_open_idx ON review.queue (enqueued_at)
    WHERE reviewed_at IS NULL;

-- Ditambahkan setelah tabel pertama kali dibuat, jadi lewat ALTER supaya database yang sudah
-- berjalan ikut mendapatkannya tanpa kehilangan data.
ALTER TABLE audit.decisions ADD COLUMN IF NOT EXISTS tier2_model_version TEXT;
ALTER TABLE review.queue ADD COLUMN IF NOT EXISTS tier2_model_version TEXT;
"""


def main() -> None:
    load_dotenv()
    with connect_postgres() as connection:
        connection.execute(_DDL)
    print(f"Skema audit, review, dan mlflow siap di database {os.environ['POSTGRES_DB']}.")


if __name__ == "__main__":
    main()
