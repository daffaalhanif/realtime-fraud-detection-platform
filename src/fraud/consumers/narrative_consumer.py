"""Consumer penyusun narasi: memasukkan transaksi zona abu-abu dari `scored` ke antrean review.

Hanya kelas `approve_review` yang masuk antrean. Transaksinya sudah disetujui, dan antrean
ini hanya menandainya untuk ditinjau analis setelahnya. Kelas keputusan dibaca dari pesan,
bukan dihitung ulang dari skor, supaya antrean selalu sejalan dengan jawaban API walau ambang
berubah. Setelah item masuk antrean, consumer ini menghitung penjelasan SHAP-nya.

Pesan ganda tidak pernah menimpa item yang sudah ada, supaya item yang sudah direview analis
tidak kembali terbuka saat pesannya dibaca ulang. Duplikat yang isinya berbeda sudah dicatat
consumer audit, yang membaca pesan yang sama. Penjelasan hanya dihitung untuk item yang
penjelasannya masih kosong, dan hanya ditulis selama masih kosong, jadi membaca ulang pesan
tidak mengulang perhitungan maupun menimpa hasil sebelumnya.

Dijalankan lewat: uv run python -m fraud.consumers.narrative_consumer
"""

import logging
import os
from collections.abc import Callable
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from fraud.consumers.base import (
    BatchPolicy,
    ScoredConsumer,
    connect_postgres,
    run_until_signalled,
    transaction_document,
)
from fraud.explainability.tier1_shap import Tier1Explainer, load_feature_groups
from fraud.schemas.score_response import DecisionClass
from fraud.schemas.scored_message import ScoredMessage

GROUP_ID = "scored-narrative"

# Item sebaiknya muncul di antrean analis dalam hitungan detik. Batas jumlah kecil karena tiap
# item review dihitung SHAP-nya, yang lambat; satu batch harus selesai jauh sebelum
# `max.poll.interval.ms`, atau consumer dikeluarkan dari group di tengah batch.
BATCH_POLICY = BatchPolicy(max_messages=50, max_wait_seconds=1.0)

_INSERT = """
INSERT INTO review.queue (
    transaction_id, tier1_score, tier2_score, model_version, tier2_model_version, scored_at,
    transaction
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transaction_id) DO NOTHING
RETURNING transaction_id
"""

_PENDING_EXPLANATION = """
SELECT transaction_id FROM review.queue
WHERE transaction_id = ANY(%s) AND explanation IS NULL
"""

# Kondisi IS NULL membuat penulisan aman diulang dan tidak pernah menimpa hasil yang ada.
_STORE_EXPLANATION = """
UPDATE review.queue SET explanation = %s
WHERE transaction_id = %s AND explanation IS NULL
"""

Explain = Callable[[ScoredMessage], dict[str, Any]]

logger = logging.getLogger(__name__)


class ReviewQueueWriter:
    """Handler batch yang memasukkan transaksi zona abu-abu ke `review.queue`.

    Args:
        connection: Koneksi Postgres dalam mode autocommit; tiap batch membuka
            transaksinya sendiri.
        explain: Menghitung dokumen penjelasan satu item, dan tidak boleh melempar exception.
    """

    def __init__(self, connection: psycopg.Connection, explain: Explain) -> None:
        self._connection = connection
        self._explain = explain

    def __call__(self, batch: list[ScoredMessage]) -> None:
        """Menyimpan item zona abu-abu satu batch secara utuh atau tidak sama sekali.

        Raises:
            psycopg.Error: Kalau Postgres gagal; item yang belum tersimpan dibaca ulang dari Kafka.
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
        self._store_explanations(items)

    def _store_explanations(self, items: list[ScoredMessage]) -> None:
        """Menghitung dan menyimpan penjelasan item yang penjelasannya masih kosong.

        Tiap hasil ditulis segera setelah dihitung, bukan di akhir batch, supaya consumer yang
        mati di tengah batch tidak membuang perhitungan item yang sudah selesai.
        """
        ids = [message.transaction.TransactionID for message in items]
        pending = {
            row[0] for row in self._connection.execute(_PENDING_EXPLANATION, (ids,)).fetchall()
        }
        explained = 0
        unavailable = 0
        for message in items:
            transaction_id = message.transaction.TransactionID
            # Pesan ganda dalam satu batch dihitung sekali.
            if transaction_id not in pending:
                continue
            pending.discard(transaction_id)
            explanation = self._explain(message)
            self._connection.execute(_STORE_EXPLANATION, (Jsonb(explanation), transaction_id))
            explained += 1
            unavailable += not explanation.get("available", False)
        if explained:
            logger.info(
                "Penjelasan SHAP: %d item dihitung, %d tidak tersedia", explained, unavailable
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
        explainer = Tier1Explainer(load_feature_groups())
        consumer = ScoredConsumer(
            GROUP_ID,
            ReviewQueueWriter(connection, explainer.explain),
            BATCH_POLICY,
            os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        )
        run_until_signalled(consumer)


if __name__ == "__main__":
    main()
