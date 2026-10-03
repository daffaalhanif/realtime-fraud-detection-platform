"""Penitipan hasil keputusan ke topik Kafka `scored` dengan batas waktu tunggu yang tegas.

Menunggu tanda terima broker tidak boleh menahan jawaban ke sistem otorisasi pembayaran.
Kalau tanda terima belum datang saat batas waktu habis, pesan tidak dibatalkan: librdkafka
tetap mencobanya di latar belakang, dan hanya pesan yang akhirnya gagal diserahkan ke
penangan kegagalan (antrean cadangan). Dengan begitu satu pesan tidak pernah ada di dua
tempat sekaligus, sehingga tidak terkirim ganda.
"""

import logging
import threading
from collections.abc import Callable
from enum import Enum

from confluent_kafka import KafkaError, KafkaException, Message, Producer

from fraud.schemas.scored_message import SCORED_TOPIC, ScoredMessage

# Jatah langkah titip pada anggaran latensi awal, direvisi setelah pengukuran Locust.
DELIVERY_WAIT_SECONDS = 0.015

# Default librdkafka 5 menit menahan pesan di memori proses selama Kafka mati; diserahkan
# lebih cepat ke antrean cadangan supaya jendela kehilangan saat proses mati ikut pendek.
# Pesan yang sudah terkirim ke socket saat broker mati tetap baru gagal setelah putusnya
# koneksi terdeteksi, batas ini tidak mempercepatnya.
_MESSAGE_TIMEOUT_MS = 5_000

_POLL_INTERVAL_SECONDS = 0.1

# Hanya menunda start-up saat broker mati; saat broker sehat pemanasan selesai jauh lebih cepat.
_WARM_UP_TIMEOUT_SECONDS = 5.0

# Callback hasil purge tidak langsung siap, flush(0) kembali sebelum callback itu sempat jalan.
_PURGE_DRAIN_SECONDS = 1.0

# Callback final selalu datang setelah batas waktu pesan terlampaui; batas ini hanya penjaga
# supaya pengirim ulang tidak menggantung. Harus lebih pendek dari sewa antrean cadangan.
_DELIVER_WAIT_CAP_SECONDS = 60.0

logger = logging.getLogger(__name__)

DeliveryFailureHandler = Callable[[bytes, bytes], None]


class DeliveryOutcome(str, Enum):
    """Status penitipan satu pesan saat batas waktu tunggu request berakhir."""

    DELIVERED = "delivered"
    FAILED = "failed"
    PENDING = "pending"


class ScoredProducer:
    """Producer topik `scored` yang dipakai bersama seluruh request dalam satu proses.

    Satu thread latar belakang melayani tanda terima dari broker, termasuk untuk pesan
    yang sudah tidak ditunggu request pengirimnya. Wajib ditutup lewat `close` saat
    aplikasi berhenti supaya pesan yang masih tertahan diserahkan ke penangan kegagalan.

    Args:
        bootstrap_servers: Alamat broker Kafka, misal `127.0.0.1:9092`.
        on_delivery_failure: Dipanggil dengan `(key, value)` untuk tiap pesan yang gagal
            dititip. Dipanggil dari thread latar belakang, jadi harus thread-safe.
        wait_seconds: Batas waktu request menunggu tanda terima broker.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        on_delivery_failure: DeliveryFailureHandler,
        wait_seconds: float = DELIVERY_WAIT_SECONDS,
    ) -> None:
        self._on_delivery_failure = on_delivery_failure
        self._wait_seconds = wait_seconds
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                # Satu pesan mewakili satu transaksi yang jawabannya sedang ditunggu,
                # jadi pesan tidak ditahan untuk dikumpulkan dengan pesan lain.
                "linger.ms": 0,
                "acks": "all",
                # Pengiriman ulang internal librdkafka setelah koneksi putus tidak
                # menghasilkan pesan ganda di broker.
                "enable.idempotence": True,
                "message.timeout.ms": _MESSAGE_TIMEOUT_MS,
            }
        )
        self._warm_up()
        self._stopping = threading.Event()
        self._poller = threading.Thread(
            target=self._serve_delivery_reports, name="kafka-delivery-poller", daemon=True
        )
        self._poller.start()

    def send(self, message: ScoredMessage) -> DeliveryOutcome:
        """Menitip satu hasil keputusan dan menunggu tanda terima sampai batas waktu.

        Args:
            message: Hasil keputusan satu transaksi.

        Returns:
            `DELIVERED` kalau broker sudah mengonfirmasi, `FAILED` kalau pesan sudah
            diserahkan ke penangan kegagalan, `PENDING` kalau batas waktu habis dan
            librdkafka masih mencoba di latar belakang.
        """
        key = str(message.transaction.TransactionID).encode()
        value = message.model_dump_json().encode()
        acknowledged = threading.Event()
        errors: list[KafkaError] = []

        def on_delivery(error: KafkaError | None, delivered: Message) -> None:
            if error is not None:
                errors.append(error)
                self._hand_over_failure(delivered.key(), delivered.value(), error)
            acknowledged.set()

        try:
            self._producer.produce(SCORED_TOPIC, key=key, value=value, on_delivery=on_delivery)
        except BufferError:
            # Antrean lokal librdkafka penuh: pesan tidak pernah masuk, callback tidak jalan.
            self._hand_over_failure(key, value, "antrean lokal producer penuh")
            return DeliveryOutcome.FAILED

        if not acknowledged.wait(self._wait_seconds):
            return DeliveryOutcome.PENDING
        return DeliveryOutcome.FAILED if errors else DeliveryOutcome.DELIVERED

    def deliver(self, key: bytes, value: bytes) -> bool:
        """Mengirim ulang satu pesan dan menunggu hasil akhirnya, untuk antrean cadangan.

        Berbeda dari `send`, kegagalan di sini tidak diserahkan ke penangan kegagalan,
        karena pesannya masih tersimpan di antrean cadangan pemanggil.

        Args:
            key: Key pesan Kafka.
            value: Isi pesan Kafka.

        Returns:
            `True` hanya kalau broker sudah mengonfirmasi pesan.
        """
        finished = threading.Event()
        errors: list[KafkaError] = []

        def on_delivery(error: KafkaError | None, _delivered: Message) -> None:
            if error is not None:
                errors.append(error)
            finished.set()

        try:
            self._producer.produce(SCORED_TOPIC, key=key, value=value, on_delivery=on_delivery)
        except BufferError:
            return False
        return finished.wait(_DELIVER_WAIT_CAP_SECONDS) and not errors

    def close(self, timeout_seconds: float = 5.0) -> None:
        """Menghentikan producer, menyerahkan pesan yang belum terkirim ke penangan kegagalan.

        Args:
            timeout_seconds: Batas waktu menunggu pesan yang masih tertahan terkirim.
        """
        self._stopping.set()
        self._poller.join()
        remaining = self._producer.flush(timeout_seconds)
        if remaining:
            # Purge membuat callback tiap pesan tersisa dipanggil dengan error, sehingga
            # pesan itu ikut masuk antrean cadangan alih-alih hilang bersama proses.
            self._producer.purge()
            self._producer.flush(_PURGE_DRAIN_SECONDS)

    def _warm_up(self) -> None:
        """Membuka koneksi ke broker sebelum request pertama, tanpa menggagalkan start-up.

        Producer idempoten baru boleh mengirim setelah memperoleh producer ID dari broker,
        dan tanpa pemanasan biaya itu jatuh ke pesan pertama hingga melewati batas tunggu.
        Broker yang mati saat start-up hanya dicatat, karena Kafka mati tidak boleh
        menahan jalur keputusan.
        """
        try:
            metadata = self._producer.list_topics(SCORED_TOPIC, timeout=_WARM_UP_TIMEOUT_SECONDS)
        except KafkaException as error:
            logger.warning("Broker Kafka belum bisa dihubungi saat start-up: %s", error)
            return
        topic_error = metadata.topics[SCORED_TOPIC].error
        if topic_error is not None:
            logger.error("Topik %s tidak tersedia: %s", SCORED_TOPIC, topic_error)

    def _serve_delivery_reports(self) -> None:
        """Loop thread latar belakang: callback librdkafka hanya jalan di dalam `poll`."""
        while not self._stopping.is_set():
            self._producer.poll(_POLL_INTERVAL_SECONDS)

    def _hand_over_failure(
        self, key: bytes | None, value: bytes | None, reason: KafkaError | str
    ) -> None:
        """Menyerahkan pesan gagal ke penangan kegagalan tanpa mematikan thread pemanggil."""
        logger.warning("Titip ke Kafka gagal untuk key %s: %s", key, reason)
        if key is None or value is None:
            logger.error("Pesan gagal tanpa key atau value, tidak bisa diserahkan: %s", reason)
            return
        try:
            self._on_delivery_failure(key, value)
        except Exception:
            logger.exception("Penangan kegagalan gagal, pesan untuk key %s hilang", key)
