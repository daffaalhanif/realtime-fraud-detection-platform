"""Pencatatan keputusan per `TransactionID` supaya request ulang dijawab tanpa dihitung ulang.

Sistem otorisasi pembayaran bisa mengirim ulang request yang sama karena timeout, padahal
request pertama sudah selesai diputuskan. Request ulang itu harus mendapat jawaban yang sama
persis, tanpa menitip pesan baru ke message broker dan tanpa melipat transaksi yang sama dua
kali ke fitur agregat entitas.
"""

from typing import cast

import redis

from fraud.schemas.score_response import ScoreResponse

REDIS_DECIDED_KEY_PREFIX = "decided"


def _decided_key(transaction_id: int) -> str:
    return f"{REDIS_DECIDED_KEY_PREFIX}:{transaction_id}"


def get_decided(redis_client: redis.Redis, transaction_id: int) -> ScoreResponse | None:
    """Mengambil keputusan yang sudah pernah diberikan untuk satu transaksi.

    Args:
        redis_client: Koneksi Redis online store.
        transaction_id: `TransactionID` dari request.

    Returns:
        Jawaban yang dulu diberikan, atau `None` kalau transaksi ini belum pernah diputuskan.
    """
    stored = redis_client.get(_decided_key(transaction_id))
    if stored is None:
        return None
    return ScoreResponse.model_validate_json(stored)


def record_decision(redis_client: redis.Redis, response: ScoreResponse) -> ScoreResponse | None:
    """Mencatat keputusan satu transaksi, hanya kalau belum ada catatan sebelumnya.

    Dua request identik yang datang hampir bersamaan bisa sama-sama lolos `get_decided`
    lalu sama-sama menghitung keputusan. Fungsi ini memilih satu pemenang secara atomik:
    hanya pemanggil yang menerima `None` yang boleh memperbarui fitur agregat entitas.

    Catatan tidak diberi TTL. Duplikat yang datang setelah catatan kedaluwarsa akan dihitung
    ulang dan dilipat kedua kalinya ke fitur agregat, dan kesalahan itu tidak bisa dikenali
    lagi setelah terjadi.

    Args:
        redis_client: Koneksi Redis online store.
        response: Jawaban yang baru dihitung pemanggil.

    Returns:
        `None` kalau catatan ini yang tersimpan. Kalau request lain sudah lebih dulu mencatat,
        jawaban request itu dikembalikan dan catatan tidak ditimpa.
    """
    # SET NX GET (Redis 7) mengecek dan menulis dalam satu perintah atomik, sehingga tidak ada
    # celah di antara pengecekan dan penulisan yang bisa diselipi request duplikat lain.
    # Stub redis-py menggabungkan tipe kembalian semua opsi SET; dengan get=True hasilnya
    # selalu nilai lama atau None.
    existing = cast(
        bytes | None,
        redis_client.set(
            _decided_key(response.transaction_id),
            response.model_dump_json(),
            nx=True,
            get=True,
        ),
    )
    if existing is None:
        return None
    return ScoreResponse.model_validate_json(existing)
