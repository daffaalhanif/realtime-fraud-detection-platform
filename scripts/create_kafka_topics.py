"""Skrip pembuatan topik Kafka `scored` secara eksplisit, aman dijalankan berulang.

Pembuatan topik otomatis di broker sengaja dimatikan, karena topik yang terbentuk dari
pesan pertama hanya punya 1 partisi, sedangkan jumlah partisi tidak bisa dikurangi setelah
topik ada. Skrip ini membuat topik dengan jumlah partisi yang dirancang, dan kalau topik
sudah ada, hanya memverifikasi jumlah partisinya tanpa mengubah apapun.

Dijalankan lewat: uv run python scripts/create_kafka_topics.py
"""

import os
import sys
import time

from confluent_kafka import KafkaError, KafkaException
# Jalur resmi menurut dokumentasi; library mengekspornya tanpa pola yang dikenali pyright.
from confluent_kafka.admin import AdminClient, NewTopic  # pyright: ignore[reportPrivateImportUsage]
from dotenv import load_dotenv

from fraud.schemas.scored_message import SCORED_TOPIC

SCORED_TOPIC_PARTITIONS = 3

# Broker hanya satu instans, jadi tidak ada broker lain untuk menampung salinan partisi.
_REPLICATION_FACTOR = 1

_ADMIN_TIMEOUT_SECONDS = 10.0

# Topik yang baru dibuat belum tentu langsung terlihat di metadata broker.
_METADATA_WAIT_SECONDS = 10.0
_METADATA_POLL_SECONDS = 0.2


class TopicMismatchError(RuntimeError):
    """Topik sudah ada dengan jumlah partisi yang berbeda dari rancangan."""


def _partition_count(admin: AdminClient, topic: str) -> int | None:
    """Jumlah partisi topik menurut metadata broker, `None` kalau topik belum ada."""
    metadata = admin.list_topics(topic, timeout=_ADMIN_TIMEOUT_SECONDS)
    topic_metadata = metadata.topics.get(topic)
    if topic_metadata is None or topic_metadata.error is not None:
        return None
    return len(topic_metadata.partitions)


def _wait_until_visible(admin: AdminClient, topic: str) -> int:
    """Menunggu topik muncul di metadata broker, lalu mengembalikan jumlah partisinya."""
    deadline = time.monotonic() + _METADATA_WAIT_SECONDS
    while True:
        count = _partition_count(admin, topic)
        if count is not None:
            return count
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Topik {topic} belum terlihat di metadata broker")
        time.sleep(_METADATA_POLL_SECONDS)


def ensure_topic(admin: AdminClient, topic: str, partitions: int) -> bool:
    """Membuat topik kalau belum ada, lalu memastikan jumlah partisinya sesuai.

    Topik yang sudah ada dengan jumlah partisi berbeda tidak diubah: menambah partisi
    memindahkan sebagian key ke partisi lain, dan itu keputusan yang harus diambil sadar.

    Args:
        admin: Klien admin Kafka yang sudah terhubung ke broker.
        topic: Nama topik.
        partitions: Jumlah partisi yang dirancang.

    Returns:
        `True` kalau topik baru dibuat oleh panggilan ini, `False` kalau sudah ada.

    Raises:
        TopicMismatchError: Topik sudah ada dengan jumlah partisi berbeda.
        TimeoutError: Topik tidak muncul di metadata broker setelah dibuat.
    """
    created = False
    if _partition_count(admin, topic) is None:
        new_topic = NewTopic(
            topic, num_partitions=partitions, replication_factor=_REPLICATION_FACTOR
        )
        future = admin.create_topics([new_topic], operation_timeout=_ADMIN_TIMEOUT_SECONDS)[topic]
        try:
            future.result(timeout=_ADMIN_TIMEOUT_SECONDS)
            created = True
        except KafkaException as error:
            # Proses lain bisa membuat topik yang sama di antara pengecekan dan pembuatan.
            if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise

    actual = _wait_until_visible(admin, topic)
    if actual != partitions:
        raise TopicMismatchError(
            f"Topik {topic} punya {actual} partisi, rancangannya {partitions}"
        )
    return created


def main() -> None:
    load_dotenv()
    admin = AdminClient(
        {"bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")}
    )
    try:
        created = ensure_topic(admin, SCORED_TOPIC, SCORED_TOPIC_PARTITIONS)
    except TopicMismatchError as error:
        sys.exit(str(error))
    status = "dibuat" if created else "sudah ada"
    print(f"Topik {SCORED_TOPIC} {status}, {SCORED_TOPIC_PARTITIONS} partisi.")


if __name__ == "__main__":
    main()
