"""Consumer pencatat audit: menyimpan setiap keputusan dari topik `scored` ke Postgres.

Tiap batch disimpan dalam satu transaksi Postgres, dan pesan ganda dibuang lewat primary key
`transaction_id`. Duplikat yang isinya berbeda dari baris tersimpan dicatat sebagai
peringatan: dua request identik yang saling mendahului di API bisa menitip dua pesan berbeda
untuk satu transaksi, dan yang tersimpan adalah pesan yang lebih dulu tiba, belum tentu
jawaban yang diterima pemanggil.

Dijalankan lewat: uv run python -m fraud.consumers.audit_consumer
"""

import logging
import os

import psycopg
from dotenv import load_dotenv

from fraud.consumers.base import (
    BatchPolicy,
    ScoredConsumer,
    connect_postgres,
    run_until_signalled,
    transaction_document,
)
from fraud.schemas.scored_message import ScoredMessage

GROUP_ID = "scored-audit"

# Tenggat membatasi tambahan lag audit akibat batching; batas jumlah hanya tercapai saat
# consumer mengejar ketertinggalan, dan membatasi ukuran satu transaksi Postgres.
BATCH_POLICY = BatchPolicy(max_messages=500, max_wait_seconds=1.0)

_INSERT = """
INSERT INTO audit.decisions (
    transaction_id, decision, tier1_score, tier2_score, model_version, tier2_model_version,
    scored_at, transaction
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transaction_id) DO NOTHING
RETURNING transaction_id
"""

_SELECT_STORED = """
SELECT transaction_id, decision, tier1_score, tier2_score, model_version, tier2_model_version
FROM audit.decisions
WHERE transaction_id = ANY(%s)
"""

logger = logging.getLogger(__name__)

_DecisionFields = tuple[str, float, float | None, str, str | None]


def _decision_fields(message: ScoredMessage) -> _DecisionFields:
    return (
        message.decision.value,
        message.tier1_score,
        message.tier2_score,
        message.model_version,
        message.tier2_model_version,
    )


class AuditWriter:
    """Handler batch yang menulis keputusan ke `audit.decisions`, idempoten per transaksi.

    Args:
        connection: Koneksi Postgres dalam mode autocommit; tiap batch membuka
            transaksinya sendiri.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def __call__(self, batch: list[ScoredMessage]) -> None:
        """Menyimpan satu batch secara utuh atau tidak sama sekali.

        Raises:
            psycopg.Error: Kalau batch gagal disimpan; tidak ada baris batch ini yang tersimpan.
        """
        with self._connection.transaction(), self._connection.cursor() as cursor:
            cursor.executemany(_INSERT, [self._row(message) for message in batch], returning=True)
            # Satu hasil per pesan, urut sesuai batch; hasil kosong berarti ditolak ON CONFLICT.
            # Dicocokkan per posisi supaya duplikat di dalam batch yang sama ikut terhitung.
            duplicates: list[ScoredMessage] = []
            for message in batch:
                if cursor.fetchone() is None:
                    duplicates.append(message)
                cursor.nextset()
            if duplicates:
                self._report_conflicts(cursor, duplicates)

    @staticmethod
    def _row(message: ScoredMessage) -> tuple[object, ...]:
        return (
            message.transaction.TransactionID,
            message.decision.value,
            message.tier1_score,
            message.tier2_score,
            message.model_version,
            message.tier2_model_version,
            message.scored_at,
            transaction_document(message.transaction),
        )

    @staticmethod
    def _report_conflicts(cursor: psycopg.Cursor, duplicates: list[ScoredMessage]) -> None:
        """Mencatat duplikat yang isinya berbeda dari baris yang sudah tersimpan."""
        ids = [message.transaction.TransactionID for message in duplicates]
        cursor.execute(_SELECT_STORED, (ids,))
        stored = {row[0]: tuple(row[1:]) for row in cursor.fetchall()}
        conflicts = 0
        for message in duplicates:
            transaction_id = message.transaction.TransactionID
            incoming = _decision_fields(message)
            # Float dibandingkan persis: pesan dari perhitungan yang sama identik bit demi bit
            # setelah melewati JSON dan DOUBLE PRECISION, jadi selisih berarti perhitungan beda.
            if stored[transaction_id] != incoming:
                conflicts += 1
                logger.warning(
                    "Duplikat berbeda isi untuk TransactionID %s: tersimpan %s, pesan %s",
                    transaction_id,
                    stored[transaction_id],
                    incoming,
                )
        logger.info(
            "%d pesan ganda dibuang, %d di antaranya berbeda isi", len(duplicates), conflicts
        )


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with connect_postgres(autocommit=True) as connection:
        consumer = ScoredConsumer(
            GROUP_ID, AuditWriter(connection), BATCH_POLICY, os.environ["KAFKA_BOOTSTRAP_SERVERS"]
        )
        run_until_signalled(consumer)


if __name__ == "__main__":
    main()
