"""Skema isi pesan pada topik Kafka `scored`."""

from datetime import datetime

from pydantic import BaseModel

from fraud.schemas.score_response import DecisionClass
from fraud.schemas.transaction import Transaction

SCORED_TOPIC = "scored"


class ScoredMessage(BaseModel):
    """Isi pesan yang dititip ke message broker setelah satu transaksi diputuskan.

    Pesan ini berdiri sendiri: consumer manapun (pencatat audit, pengarsip,
    penyusun narasi) bisa memprosesnya tanpa perlu memanggil balik API
    scoring untuk data tambahan. `tier2_score` tetap disimpan terpisah dari
    `tier1_score` (bukan cuma skor akhir gabungan) supaya performa tier 2
    bisa dievaluasi terhadap baseline tier 1 pada transaksi yang sama.

    Attributes:
        transaction: Data transaksi mentah persis seperti yang diterima `POST /score`.
        decision: Salah satu dari tiga kelas keputusan.
        tier1_score: Peluang fraud terkalibrasi dari tier 1, selalu ada
            karena tier 1 menilai semua transaksi tanpa kecuali.
        tier2_score: Logit tier 2 yang dipanggil dalam mode shadow, belum
            dikalibrasi sehingga bukan peluang dan tidak ikut menentukan
            keputusan. Hanya terisi untuk transaksi zona abu-abu.
        model_version: Versi model tier 1 yang menghasilkan skor ini.
        tier2_model_version: Versi model tier 2 yang menghasilkan
            `tier2_score`, kosong kalau tier 2 tidak dipanggil.
        scored_at: Waktu keputusan diambil.
    """

    transaction: Transaction
    decision: DecisionClass
    tier1_score: float
    tier2_score: float | None = None
    model_version: str
    tier2_model_version: str | None = None
    scored_at: datetime
