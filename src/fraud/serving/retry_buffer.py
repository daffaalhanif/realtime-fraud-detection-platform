"""Antrean cadangan lokal untuk pesan yang gagal dititip ke Kafka, dikirim ulang berkala.

Fungsinya terbatas sebagai penyangga sementara selama Kafka tidak bisa menerima pesan,
bukan jalur utama pengiriman hasil keputusan. Antrean disimpan di SQLite supaya pesan yang
sudah masuk tetap selamat walau proses API mati atau di-restart.

Beberapa proses worker API boleh memakai berkas yang sama. Tiap pesan diambil lewat sewa
(lease) berbatas waktu, sehingga satu pesan tidak dikirim ulang oleh dua proses sekaligus,
dan pesan yang sewanya habis karena prosesnya mati akan diambil proses lain.
"""

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

# Selang antar siklus kirim ulang; siklus langsung berhenti di kegagalan pertama, jadi
# selang ini juga menjadi jeda sebelum broker yang masih mati dicoba lagi.
RESEND_INTERVAL_SECONDS = 5.0

# Harus lebih lama dari waktu terlama satu kirim ulang, kalau tidak pesan yang masih
# dikirim bisa diambil proses lain dan terkirim ganda.
_LEASE_SECONDS = 120.0

# Menunggu kunci tulis proses lain, bukan langsung gagal "database is locked".
_BUSY_TIMEOUT_MS = 5_000

logger = logging.getLogger(__name__)

Deliver = Callable[[bytes, bytes], bool]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retry_queue (
    id INTEGER PRIMARY KEY,
    msg_key BLOB NOT NULL,
    msg_value BLOB NOT NULL,
    enqueued_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_until REAL NOT NULL DEFAULT 0
)
"""

# Urut attempts lebih dulu: pesan yang terus gagal (misal ditolak permanen oleh broker)
# turun ke belakang dan tidak menghalangi pesan lain terkirim.
_CLAIM_ONE = """
UPDATE retry_queue SET lease_until = :lease_until
WHERE id = (
    SELECT id FROM retry_queue WHERE lease_until <= :now ORDER BY attempts, id LIMIT 1
)
RETURNING id, msg_key, msg_value
"""


class RetryBuffer:
    """Antrean cadangan berbasis SQLite dengan thread pengirim ulang di latar belakang.

    Args:
        path: Lokasi berkas SQLite. Folder induknya dibuat kalau belum ada.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Satu koneksi dipakai thread poller Kafka dan thread pengirim ulang, dijaga lock.
        self._connection = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None
        )
        self._lock = threading.Lock()
        with self._lock:
            self._connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            # WAL membuat pembaca dan penulis dari proses berbeda tidak saling mengunci.
            self._connection.execute("PRAGMA journal_mode = WAL")
            # Tiap pesan di sini adalah satu-satunya salinan, jadi fsync per commit sepadan.
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute(_SCHEMA)
        self._stopping = threading.Event()
        self._resender: threading.Thread | None = None

    def enqueue(self, key: bytes, value: bytes) -> None:
        """Menyimpan satu pesan yang gagal dititip. Aman dipanggil dari thread mana pun.

        Args:
            key: Key pesan Kafka (`TransactionID`).
            value: Isi pesan Kafka.
        """
        with self._lock:
            self._connection.execute(
                "INSERT INTO retry_queue (msg_key, msg_value, enqueued_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )

    def count(self) -> int:
        """Jumlah pesan yang masih menunggu dikirim ulang."""
        with self._lock:
            (total,) = self._connection.execute("SELECT COUNT(*) FROM retry_queue").fetchone()
        return total

    def start(self, deliver: Deliver) -> None:
        """Menjalankan thread pengirim ulang berkala.

        Args:
            deliver: Mengirim `(key, value)` ke Kafka dan menunggu hasil akhirnya. Harus
                mengembalikan `True` hanya kalau broker sudah mengonfirmasi pesan.
        """
        self._resender = threading.Thread(
            target=self._resend_loop, args=(deliver,), name="retry-buffer-resender", daemon=True
        )
        self._resender.start()

    def stop(self) -> None:
        """Menghentikan thread pengirim ulang, antrean tetap menerima `enqueue`.

        Dipanggil sebelum producer ditutup, karena penutupan producer masih menyerahkan
        pesan yang tertahan ke antrean ini.
        """
        self._stopping.set()
        if self._resender is not None:
            self._resender.join()

    def close(self) -> None:
        """Menutup berkas antrean. Pesan tersisa tetap tersimpan untuk proses berikutnya."""
        self.stop()
        with self._lock:
            self._connection.close()

    def resend_pending(self, deliver: Deliver) -> int:
        """Satu siklus kirim ulang, berhenti di kegagalan pertama.

        Args:
            deliver: Sama seperti pada `start`.

        Returns:
            Jumlah pesan yang berhasil dikirim ulang dan dihapus dari antrean.
        """
        resent = 0
        while not self._stopping.is_set():
            claimed = self._claim_one()
            if claimed is None:
                break
            row_id, key, value = claimed
            if not self._try_deliver(deliver, key, value):
                self._release(row_id)
                break
            with self._lock:
                self._connection.execute("DELETE FROM retry_queue WHERE id = ?", (row_id,))
            resent += 1
        return resent

    def _resend_loop(self, deliver: Deliver) -> None:
        """Loop thread latar belakang: satu siklus kirim ulang tiap selang waktu."""
        while not self._stopping.wait(RESEND_INTERVAL_SECONDS):
            try:
                resent = self.resend_pending(deliver)
            except sqlite3.Error:
                logger.exception("Siklus kirim ulang antrean cadangan gagal")
                continue
            if resent:
                logger.info("Antrean cadangan: %d pesan terkirim ulang", resent)

    def _claim_one(self) -> tuple[int, bytes, bytes] | None:
        """Mengambil satu pesan secara atomik antar proses dengan menyewanya sementara."""
        now = time.time()
        with self._lock:
            # fetchall, bukan fetchone: statement RETURNING baru selesai dan melepas kunci
            # tulis setelah seluruh hasilnya dibaca.
            rows = self._connection.execute(
                _CLAIM_ONE, {"now": now, "lease_until": now + _LEASE_SECONDS}
            ).fetchall()
        return rows[0] if rows else None

    def _release(self, row_id: int) -> None:
        """Mengembalikan pesan yang gagal dikirim supaya siklus berikutnya mencobanya lagi."""
        with self._lock:
            self._connection.execute(
                "UPDATE retry_queue SET lease_until = 0, attempts = attempts + 1 WHERE id = ?",
                (row_id,),
            )

    @staticmethod
    def _try_deliver(deliver: Deliver, key: bytes, value: bytes) -> bool:
        """Memanggil `deliver` tanpa membiarkan error-nya menghentikan thread pengirim ulang."""
        try:
            return deliver(key, value)
        except Exception:
            logger.exception("Kirim ulang gagal untuk key %s", key)
            return False
