"""Loop consumer bersama untuk topik `scored`: kumpulkan batch, proses, baru commit offset.

Offset hanya di-commit setelah handler selesai memproses seluruh batch tanpa galat. Proses
yang mati di antara keduanya membuat batch itu dibaca ulang saat consumer dinyalakan lagi,
jadi setiap handler wajib membuang pesan ganda berdasarkan `TransactionID`.

Handler yang gagal menghentikan proses tanpa commit, sehingga satu-satunya jalur pemulihan
adalah membaca ulang dari offset terakhir. Pesan yang tidak bisa dibaca sebagai
`ScoredMessage` dicatat lalu dilewati, karena membacanya ulang tidak akan pernah berhasil
dan hanya akan menahan seluruh consumer group di pesan yang sama.

Modul ini juga memuat utilitas proses yang dipakai bersama consumer: koneksi Postgres, bentuk
JSON transaksi di tabel Postgres, dan penghentian rapi lewat sinyal.
"""

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType

import psycopg
from psycopg.types.json import Jsonb
from confluent_kafka import Consumer, KafkaError, KafkaException, Message, TopicPartition
from pydantic import ValidationError

from fraud.schemas.scored_message import SCORED_TOPIC, ScoredMessage
from fraud.schemas.transaction import Transaction

# Batas satu putaran menunggu pesan, sekaligus batas jeda sebelum `stop` dan tenggat batch
# diperhatikan, karena signal handler Python baru jalan setelah panggilan librdkafka kembali.
_POLL_TIMEOUT_SECONDS = 0.5

logger = logging.getLogger(__name__)

BatchHandler = Callable[[list[ScoredMessage]], None]


@dataclass(frozen=True)
class BatchPolicy:
    """Batas ukuran satu batch; batch diproses begitu salah satu batas tercapai.

    Attributes:
        max_messages: Jumlah pesan maksimum per batch, termasuk pesan tidak valid.
        max_wait_seconds: Batas umur batch, dihitung sejak pesan pertamanya diterima.
    """

    max_messages: int
    max_wait_seconds: float


class ScoredConsumer:
    """Consumer satu consumer group atas topik `scored`.

    Args:
        group_id: Nama consumer group; tiap tanggung jawab punya group sendiri sehingga
            progres bacanya independen dari consumer lain.
        handle_batch: Dipanggil dengan seluruh pesan valid satu batch. Harus idempoten
            terhadap pesan yang pernah diterima sebelumnya, dan melempar exception kalau
            batch tidak tersimpan utuh.
        policy: Batas ukuran batch.
        bootstrap_servers: Alamat broker Kafka.
        topic: Topik yang dibaca; selain `scored` hanya untuk pengujian.
    """

    def __init__(
        self,
        group_id: str,
        handle_batch: BatchHandler,
        policy: BatchPolicy,
        bootstrap_servers: str,
        topic: str = SCORED_TOPIC,
    ) -> None:
        self._group_id = group_id
        self._handle_batch = handle_batch
        self._policy = policy
        self._topic = topic
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                # Satu proses per group: proses yang dinyalakan ulang dengan ID yang sama langsung
                # mengambil kembali partisinya, tanpa menunggu broker mengeluarkan anggota lama
                # yang mati lewat session timeout. Proses kedua dengan ID sama ditolak broker.
                "group.instance.id": group_id,
                "enable.auto.commit": False,
                # Group yang baru pertama jalan ikut membaca keputusan yang dititip sebelum
                # consumer ini ada; default `latest` akan melewatinya selamanya.
                "auto.offset.reset": "earliest",
            }
        )
        self._stopping = threading.Event()
        self._batch: list[ScoredMessage] = []
        self._uncommitted = 0
        self._batch_deadline: float | None = None

    def run(self) -> None:
        """Membaca dan memproses pesan sampai `stop` dipanggil, lalu memproses batch terakhir.

        Raises:
            Exception: Apapun yang dilempar handler, atau galat fatal Kafka. Batch yang sedang
                diproses tidak di-commit dan akan dibaca ulang saat consumer dinyalakan lagi.
        """
        self._consumer.subscribe([self._topic], on_revoke=self._discard_batch)
        logger.info("Consumer group %s mulai membaca topik %s", self._group_id, self._topic)
        try:
            while not self._stopping.is_set():
                messages = self._consumer.consume(
                    num_messages=self._policy.max_messages - self._uncommitted,
                    timeout=self._poll_timeout(),
                )
                for message in messages:
                    self._accept(message)
                if self._batch_is_due():
                    self._flush()
            self._flush()
        finally:
            # Anggota statis tetap terdaftar di group setelah ditutup, sampai session timeout.
            self._consumer.close()
        logger.info("Consumer group %s berhenti", self._group_id)

    def stop(self) -> None:
        """Meminta loop berhenti setelah batch yang sedang terkumpul diproses dan di-commit."""
        self._stopping.set()

    def _poll_timeout(self) -> float:
        """Lama menunggu pesan pada putaran ini, dipotong supaya tenggat batch tidak terlewat."""
        if self._batch_deadline is None:
            return _POLL_TIMEOUT_SECONDS
        return max(0.0, min(_POLL_TIMEOUT_SECONDS, self._batch_deadline - time.monotonic()))

    def _accept(self, message: Message) -> None:
        """Memasukkan satu pesan ke batch, atau mencatat dan melewatinya kalau tidak valid."""
        error = message.error()
        if error is not None:
            self._handle_kafka_error(error)
            return

        if self._uncommitted == 0:
            self._batch_deadline = time.monotonic() + self._policy.max_wait_seconds
        self._uncommitted += 1
        value = message.value()
        try:
            if value is None:
                raise ValueError("pesan tanpa isi")
            self._batch.append(ScoredMessage.model_validate_json(value))
        except (ValidationError, ValueError) as invalid:
            # Satu baris log per pesan; isi mentah sudah dicatat utuh, jadi tidak diulang per galat.
            reason = (
                invalid.errors(include_url=False, include_input=False)
                if isinstance(invalid, ValidationError)
                else str(invalid)
            )
            logger.error(
                "Pesan tidak valid dilewati: partisi %s offset %s key %r, %s, isi mentah %r",
                message.partition(),
                message.offset(),
                message.key(),
                reason,
                value,
            )

    def _handle_kafka_error(self, error: KafkaError) -> None:
        """Galat fatal menghentikan consumer; galat sementara dicoba ulang librdkafka sendiri."""
        if error.fatal():
            raise KafkaException(error)
        logger.warning("Galat Kafka pada consumer group %s: %s", self._group_id, error)

    def _batch_is_due(self) -> bool:
        if self._uncommitted == 0:
            return False
        if self._uncommitted >= self._policy.max_messages:
            return True
        return self._batch_deadline is not None and time.monotonic() >= self._batch_deadline

    def _flush(self) -> None:
        """Memproses batch lewat handler, lalu commit offset seluruh pesan yang sudah dibaca."""
        if self._uncommitted == 0:
            return
        started = time.monotonic()
        if self._batch:
            self._handle_batch(self._batch)
        # Selama handler berjalan tidak ada pesan baru yang dibaca, jadi posisi baca saat ini
        # tepat berada di akhir batch yang baru diproses.
        self._consumer.commit(asynchronous=False)
        logger.info(
            "Consumer group %s memproses %d pesan (%d valid) dalam %.1f ms",
            self._group_id,
            self._uncommitted,
            len(self._batch),
            (time.monotonic() - started) * 1000,
        )
        self._reset_batch()

    def _discard_batch(self, _consumer: Consumer, _partitions: list[TopicPartition]) -> None:
        """Membuang batch yang belum diproses saat partisi ditarik dalam rebalance.

        Pemilik partisi berikutnya membaca ulang dari offset terakhir yang di-commit, jadi
        batch ini tetap akan diproses, hanya oleh pemilik yang baru.
        """
        if self._uncommitted:
            logger.info(
                "Rebalance: %d pesan belum diproses dibuang, akan dibaca ulang",
                self._uncommitted,
            )
        self._reset_batch()

    def _reset_batch(self) -> None:
        self._batch = []
        self._uncommitted = 0
        self._batch_deadline = None


def run_until_signalled(consumer: ScoredConsumer) -> None:
    """Menjalankan consumer sampai SIGTERM atau SIGINT diterima, lalu berhenti dengan rapi.

    Args:
        consumer: Consumer yang akan dijalankan di thread pemanggil, yang harus main thread
            karena hanya main thread yang boleh memasang signal handler.
    """

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("Sinyal %s diterima, menyelesaikan batch terakhir", signal.Signals(signum).name)
        consumer.stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    consumer.run()


def connect_postgres(autocommit: bool = False) -> psycopg.Connection:
    """Membuka koneksi ke database aplikasi dari environment variable `POSTGRES_*`.

    Args:
        autocommit: `True` untuk pemanggil yang membatasi transaksinya sendiri lewat
            `connection.transaction()`.
    """
    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
        autocommit=autocommit,
    )


def transaction_document(transaction: Transaction) -> Jsonb:
    """Transaksi mentah dalam bentuk kolom JSONB, sama untuk tabel audit dan antrean review.

    Field yang dibuang hanya yang bernilai `None`, dan semuanya ber-default `None` di skema
    `Transaction`, jadi `Transaction.model_validate` atas dokumen ini menghasilkan transaksi
    yang identik dengan aslinya.
    """
    return Jsonb(transaction.model_dump(mode="json", exclude_none=True))
