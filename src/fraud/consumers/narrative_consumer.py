"""Consumer penyusun narasi: memasukkan transaksi zona abu-abu dari `scored` ke antrean review.

Hanya kelas `approve_review` yang masuk antrean. Transaksinya sudah disetujui, dan antrean
ini hanya menandainya untuk ditinjau analis setelahnya. Kelas keputusan dibaca dari pesan,
bukan dihitung ulang dari skor, supaya antrean selalu sejalan dengan jawaban API walau ambang
berubah. Narasi penjelasan untuk analis belum diisi di consumer ini.

Pesan ganda tidak pernah menimpa item yang sudah ada, supaya item yang sudah direview analis
tidak kembali terbuka saat pesannya dibaca ulang. Duplikat yang isinya berbeda sudah dicatat
consumer audit, yang membaca pesan yang sama.

Dijalankan lewat: uv run python -m fraud.consumers.narrative_consumer
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
from fraud.schemas.score_response import DecisionClass
from fraud.schemas.scored_message import ScoredMessage

GROUP_ID = "scored-narrative"

# Item sebaiknya muncul di antrean analis dalam hitungan detik, dan beban per pesan setara
# consumer audit: satu INSERT tanpa pemrosesan tambahan.
BATCH_POLICY = BatchPolicy(max_messages=500, max_wait_seconds=1.0)

_INSERT = """
INSERT INTO review.queue (
    transaction_id, tier1_score, tier2_score, model_version, tier2_model_version, scored_at,
    transaction
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transaction_id) DO NOTHING
RETURNING transaction_id
"""

logger = logging.getLogger(__name__)


class ReviewQueueWriter:
    """Handler batch yang memasukkan transaksi zona abu-abu ke `review.queue`.

    Args:
        connection: Koneksi Postgres dalam mode autocommit; tiap batch membuka
            transaksinya sendiri.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def __call__(self, batch: list[ScoredMessage]) -> None:
        """Menyimpan item zona abu-abu satu batch secara utuh atau tidak sama sekali.

        Raises:
            psycopg.Error: Kalau batch gagal disimpan; tidak ada item batch ini yang tersimpan.
        """
        items = [message for message in batch if message.decision is DecisionClass.APPROVE_REVIEW]
        if not items:
            return
        inserted = 0
        with self._connection.transaction(), self._connection.cursor() as cursor:
            cursor.executemany(_INSERT, [self._row(message) for message in items], returning=True)
            for _ in items:
                if cursor.fetchone() is not None:
                    inserted += 1
                cursor.nextset()
        logger.info(
            "Antrean review: %d item baru, %d pesan ganda dibuang", inserted, len(items) - inserted
        )

    @staticmethod
    def _row(message: ScoredMessage) -> tuple[object, ...]:
        return (
            message.transaction.TransactionID,
            message.tier1_score,
            message.tier2_score,
            message.model_version,
            message.tier2_model_version,
            message.scored_at,
            transaction_document(message.transaction),
        )


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with connect_postgres(autocommit=True) as connection:
        consumer = ScoredConsumer(
            GROUP_ID,
            ReviewQueueWriter(connection),
            BATCH_POLICY,
            os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        )
        run_until_signalled(consumer)


if __name__ == "__main__":
    main()
