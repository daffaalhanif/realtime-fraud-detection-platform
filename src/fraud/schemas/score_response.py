"""Skema response `POST /score`."""

from enum import Enum

from pydantic import BaseModel


class DecisionClass(str, Enum):
    """Tiga kelas keputusan yang mungkin dihasilkan untuk satu transaksi.

    Zona `APPROVE_REVIEW` tetap menyetujui transaksi seperti `APPROVE`,
    hanya ditandai untuk ditinjau analis setelahnya. Customer tidak merasakan
    perbedaan perlakuan antara kedua kelas ini.
    """

    DECLINE = "decline"
    APPROVE_REVIEW = "approve_review"
    APPROVE = "approve"


class ScoreResponse(BaseModel):
    """Jawaban langsung ke sistem otorisasi pembayaran untuk satu transaksi.

    Attributes:
        transaction_id: Penanda transaksi, dipakai pemanggil untuk
            mencocokkan response ini dengan request yang dikirim.
        decision: Salah satu dari tiga kelas keputusan.
        score: Peluang fraud yang sudah dikalibrasi, dari tier yang
            menentukan keputusan akhir.
    """

    transaction_id: int
    decision: DecisionClass
    score: float
