"""Fitur jendela waktu per kunci entitas, dihitung dari histori transaksi terbaru.

Berbeda dari `fraud.features.aggregate` yang cukup membawa state berjalan, fitur di sini
membutuhkan waktu dan nominal tiap transaksi sebelumnya. Sumbernya saat serving adalah Redis
LIST per entitas, yang hanya menyimpan `SEQUENCE_RETENTION_LIMIT` transaksi terakhir. Fungsi
ini sendiri yang memotong histori ke batas itu, sehingga jalur batch yang memegang histori
lebih panjang tetap menghasilkan angka yang sama persis dengan jalur serving.
"""

from collections.abc import Collection, Sequence
from itertools import islice
from typing import TypedDict

from fraud.features.online_store import SEQUENCE_RETENTION_LIMIT

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 604_800


class WindowFeatures(TypedDict):
    """Aktivitas kunci entitas pada beberapa jendela waktu sebelum transaksi sekarang.

    Jendela berbentuk setengah terbuka: transaksi sebelumnya masuk jendela kalau selisih
    waktunya terhadap transaksi sekarang kurang dari panjang jendela, termasuk selisih nol.
    Transaksi yang sedang dinilai tidak ikut dihitung.

    Attributes:
        count_last_1h: Jumlah transaksi sebelumnya dalam satu jam terakhir.
        count_last_24h: Jumlah transaksi sebelumnya dalam 24 jam terakhir.
        count_last_7d: Jumlah transaksi sebelumnya dalam tujuh hari terakhir, paling banyak
            `SEQUENCE_RETENTION_LIMIT`.
        amt_sum_last_1h: Total nominal transaksi sebelumnya dalam satu jam terakhir.
        amt_sum_last_24h: Total nominal transaksi sebelumnya dalam 24 jam terakhir.
        amt_sum_last_7d: Total nominal transaksi sebelumnya dalam tujuh hari terakhir.
    """

    count_last_1h: int
    count_last_24h: int
    count_last_7d: int
    amt_sum_last_1h: float
    amt_sum_last_24h: float
    amt_sum_last_7d: float


WINDOW_FEATURE_NAMES = list(WindowFeatures.__annotations__)


def requires_window_features(input_columns: Collection[str]) -> bool:
    """Apakah kontrak input sebuah model memuat fitur jendela waktu."""
    return any(name in input_columns for name in WINDOW_FEATURE_NAMES)


def window_features(history: Sequence[tuple[int, float]], transaction_dt: int) -> WindowFeatures:
    """Menghitung fitur jendela waktu satu transaksi dari histori kunci entitasnya.

    Dipanggil identik dari jalur batch (histori dibangun dari loop kronologis) maupun jalur
    serving (histori diurai dari Redis LIST). Hanya `SEQUENCE_RETENTION_LIMIT` elemen terakhir
    yang dibaca, apa pun panjang `history`.

    Args:
        history: Pasangan `(TransactionDT, TransactionAmt)` transaksi sebelumnya, urut
            kronologis naik dan tidak lebih baru dari transaksi sekarang. Boleh kosong.
        transaction_dt: Waktu (`TransactionDT`) transaksi yang sedang dinilai.

    Returns:
        Fitur jendela waktu; seluruhnya nol kalau tidak ada transaksi sebelumnya di jendela.
    """
    count_1h = count_24h = count_7d = 0
    sum_1h = sum_24h = sum_7d = 0.0
    for previous_dt, previous_amt in islice(reversed(history), SEQUENCE_RETENTION_LIMIT):
        elapsed = transaction_dt - previous_dt
        # Histori urut naik, jadi elemen berikutnya (lebih lama) pasti juga di luar jendela.
        if elapsed >= SECONDS_PER_WEEK:
            break
        count_7d += 1
        sum_7d += previous_amt
        if elapsed < SECONDS_PER_DAY:
            count_24h += 1
            sum_24h += previous_amt
        if elapsed < SECONDS_PER_HOUR:
            count_1h += 1
            sum_1h += previous_amt

    return WindowFeatures(
        count_last_1h=count_1h,
        count_last_24h=count_24h,
        count_last_7d=count_7d,
        amt_sum_last_1h=sum_1h,
        amt_sum_last_24h=sum_24h,
        amt_sum_last_7d=sum_7d,
    )
