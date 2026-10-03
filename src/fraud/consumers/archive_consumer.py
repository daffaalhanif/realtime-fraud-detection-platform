"""Consumer pengarsip: menulis transaksi dari topik `scored` ke arsip Parquet offline store.

Tiap batch ditulis sebagai satu file baru per partisi hari yang tersentuh. Transaksi yang
`TransactionID`-nya sudah ada di partisi yang sama, baik di arsip maupun di data awal, dibuang
sebelum ditulis. Dengan begitu `TransactionID` unik di seluruh offline store: pesan ganda dari
Kafka dan transaksi dataset yang diputar ulang lewat API tidak tersimpan dua kali, dan
pemulihan Redis dari gabungan kedua folder tidak melipat transaksi yang sama dua kali.

File baru selalu muncul utuh atau tidak muncul sama sekali, karena file setengah jadi akan
menggagalkan pembaca yang membaca seluruh folder, termasuk jalur training.

Dijalankan lewat: uv run python -m fraud.consumers.archive_consumer
"""

import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

from fraud.consumers.base import BatchPolicy, ScoredConsumer, run_until_signalled
from fraud.features.offline_store import (
    ARCHIVE_PARQUET_DIR,
    INITIAL_PARQUET_DIR,
    partition_day,
    partition_dir,
    transactions_to_table,
)
from fraud.schemas.scored_message import ScoredMessage
from fraud.schemas.transaction import Transaction

GROUP_ID = "scored-archive"

# Tiap batch menghasilkan minimal satu file, jadi batas yang longgar menahan jumlah file kecil;
# arsip dipakai secara offline, sehingga tertinggal sampai tenggat tidak berdampak.
BATCH_POLICY = BatchPolicy(max_messages=5000, max_wait_seconds=60.0)

logger = logging.getLogger(__name__)


class ArchiveWriter:
    """Handler batch yang menulis transaksi ke arsip Parquet, idempoten per transaksi.

    Args:
        archive_dir: Folder arsip yang ditulis.
        initial_dir: Folder data awal, hanya dibaca untuk pengecekan duplikat.
    """

    def __init__(
        self, archive_dir: Path = ARCHIVE_PARQUET_DIR, initial_dir: Path = INITIAL_PARQUET_DIR
    ) -> None:
        self._archive_dir = archive_dir
        self._initial_dir = initial_dir

    def __call__(self, batch: list[ScoredMessage]) -> None:
        """Menulis transaksi batch yang belum tersimpan, satu file per partisi hari.

        Raises:
            OSError: Kalau file gagal ditulis; partisi yang sudah tertulis akan dikenali
                sebagai duplikat saat batch dibaca ulang.
        """
        by_day: defaultdict[int, list[Transaction]] = defaultdict(list)
        for message in batch:
            by_day[partition_day(message.transaction.TransactionDT)].append(message.transaction)

        written = 0
        for day, transactions in by_day.items():
            stored_ids = self._stored_ids(day)
            fresh: list[Transaction] = []
            for transaction in transactions:
                # Ditambahkan ke stored_ids supaya salinan kedua di batch yang sama ikut dibuang.
                if transaction.TransactionID not in stored_ids:
                    stored_ids.add(transaction.TransactionID)
                    fresh.append(transaction)
            if fresh:
                self._write(day, fresh)
                written += len(fresh)
        logger.info(
            "Arsip: %d transaksi ditulis ke %d partisi hari, %d dibuang karena sudah tersimpan",
            written,
            len(by_day),
            len(batch) - written,
        )

    def _stored_ids(self, day: int) -> set[int]:
        """`TransactionID` yang sudah ada di partisi hari ini, di arsip maupun data awal."""
        ids: set[int] = set()
        for root in (self._archive_dir, self._initial_dir):
            for path in _parquet_files(partition_dir(root, day)):
                ids.update(pq.read_table(path, columns=["TransactionID"]).column(0).to_pylist())
        return ids

    def _write(self, day: int, transactions: list[Transaction]) -> None:
        """Menulis satu file baru di partisi hari, lewat nama sementara lalu ganti nama atomik."""
        directory = partition_dir(self._archive_dir, day)
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"{uuid.uuid4().hex}.parquet"
        # Pembaca Parquet melewati file berawalan titik, jadi sisa file ini saat proses mati
        # tidak pernah ikut terbaca.
        temporary = directory / f".{final.name}.tmp"
        pq.write_table(transactions_to_table(transactions), temporary)
        _fsync(temporary)
        os.replace(temporary, final)
        # Ganti nama baru tahan mati listrik setelah isi foldernya ikut di-fsync; offset Kafka
        # di-commit setelah ini, jadi file harus sudah pasti ada.
        _fsync(directory)


def _parquet_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [path for path in directory.glob("*.parquet") if not path.name.startswith((".", "_"))]


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    consumer = ScoredConsumer(
        GROUP_ID, ArchiveWriter(), BATCH_POLICY, os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    )
    run_until_signalled(consumer)


if __name__ == "__main__":
    main()
