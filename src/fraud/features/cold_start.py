"""Nilai default fitur agregat untuk entitas yang belum pernah tercatat.

Dipakai saat scoring transaksi PERTAMA suatu entitas: baik entitas yang
benar-benar baru, maupun saat penyimpanan online baru dibangun ulang dari
data historis dan belum sempat memuat entitas ini. Nilai di sini murni
representasi sementara untuk satu kali scoring, begitu keputusan atas
transaksi itu diambil, `fraud.features.aggregate.process_transaction`
menggantikannya dengan state yang dihitung dari data transaksi entitas
itu sendiri.
"""

from fraud.features.aggregate import AggregateState

# Median nominal historis, bukan mean: distribusi condong ekstrem ke nilai kecil.
_DEFAULT_AMT_MEAN = 68.77

# Sentinel jarak waktu jauh, bukan nol: histori minim terbukti tidak lebih berisiko di data ini.
_SENTINEL_LAST_TXN_DT = -999_999_999


def get_default_state() -> AggregateState:
    """Membuat state placeholder untuk entitas yang belum punya histori.

    Selalu mengembalikan dict baru, bukan referensi ke objek yang sama,
    supaya pemanggil yang memodifikasi hasilnya tidak saling mencemari
    antar request yang berbeda.

    Returns:
        State agregat sementara, valid dipakai langsung sebagai argumen
        `process_transaction` untuk scoring transaksi pertama entitas ini.
    """
    return AggregateState(
        txn_count=0,
        amt_mean=_DEFAULT_AMT_MEAN,
        amt_m2=0.0,
        amt_max=0.0,
        last_txn_dt=_SENTINEL_LAST_TXN_DT,
    )
