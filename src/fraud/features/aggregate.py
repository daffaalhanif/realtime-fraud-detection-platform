"""Fungsi fitur agregat per entitas (card1), dipakai jalur batch maupun incremental.

Fitur diturunkan dari statistik berjalan (running state) per entitas,
bukan dihitung ulang dari seluruh histori mentah tiap kali dipanggil,
supaya biaya pembaruan tetap O(1) walau satu entitas bisa memiliki
belasan ribu transaksi. Modul ini tidak menyediakan nilai state untuk
entitas yang belum pernah tercatat sama sekali; itu tanggung jawab
`fraud.features.cold_start`.
"""

from typing import TypedDict


class AggregateState(TypedDict):
    """State agregat satu entitas, disimpan sebagai Redis HASH.

    Invariant yang harus dipenuhi pemanggil (termasuk placeholder cold
    start): `amt_mean` selalu bernilai positif karena nominal transaksi
    tidak pernah nol atau negatif, dan `txn_count` bernilai 0 hanya untuk
    entitas yang benar-benar belum punya histori.

    Attributes:
        txn_count: Jumlah transaksi entitas ini sebelum transaksi sekarang.
        amt_mean: Rata-rata nominal transaksi historis entitas ini.
        amt_m2: Jumlah kuadrat selisih terhadap rata-rata berjalan, dipakai
            untuk menurunkan varians tanpa menyimpan seluruh nilai mentah.
        amt_max: Nominal transaksi tertinggi dalam histori entitas ini.
        last_txn_dt: Waktu (`TransactionDT`) transaksi terakhir entitas ini.
    """

    txn_count: int
    amt_mean: float
    amt_m2: float
    amt_max: float
    last_txn_dt: int


class AggregateFeatures(TypedDict):
    """Fitur agregat siap pakai sebagai input model.

    Seluruh nilai dihitung hanya dari transaksi sebelum transaksi yang
    sedang dinilai, sehingga fungsi yang sama aman dipakai baik untuk
    membangun data pelatihan dari histori penuh maupun untuk scoring
    transaksi real-time, tanpa risiko keduanya menghasilkan definisi
    fitur yang berbeda.

    Attributes:
        txn_count_so_far: Jumlah transaksi entitas ini sebelum transaksi
            sekarang.
        seconds_since_last_txn: Jarak waktu ke transaksi terakhir entitas
            ini, dalam detik.
        amt_mean_so_far: Rata-rata nominal transaksi historis entitas ini.
        amt_std_so_far: Deviasi standar nominal transaksi historis entitas
            ini.
        amt_max_so_far: Nominal transaksi tertinggi dalam histori entitas
            ini.
        amt_ratio_to_mean: Rasio nominal transaksi sekarang terhadap
            rata-rata historisnya.
    """

    txn_count_so_far: int
    seconds_since_last_txn: int
    amt_mean_so_far: float
    amt_std_so_far: float
    amt_max_so_far: float
    amt_ratio_to_mean: float


def process_transaction(
    state: AggregateState,
    transaction_amt: float,
    transaction_dt: int,
) -> tuple[AggregateFeatures, AggregateState]:
    """Menurunkan fitur scoring dari state entitas, sekaligus melipat transaksi ini ke state baru.

    Dipanggil identik dari jalur batch (loop kronologis per entitas saat
    membangun ulang data pelatihan) maupun jalur incremental (satu kali
    per transaksi masuk saat serving). Kedua nilai balik dipakai pada
    momen berbeda: `features` dipakai untuk scoring saat ini juga,
    `new_state` ditulis ke penyimpanan setelah keputusan atas transaksi
    ini diambil.

    Args:
        state: State agregat entitas SEBELUM transaksi ini.
        transaction_amt: Nominal (`TransactionAmt`) transaksi yang sedang
            diproses.
        transaction_dt: Waktu (`TransactionDT`) transaksi yang sedang
            diproses.

    Returns:
        Pasangan `(features, new_state)`. `features` adalah fitur agregat
        untuk scoring transaksi ini. `new_state` adalah state entitas
        setelah transaksi ini dilipat masuk, menggantikan (bukan
        menambah pada) state lama.
    """
    # Deviasi standar hanya terdefinisi kalau entitas punya histori sebelumnya.
    # Pembulatan bisa membuat amt_m2 negatif tipis, dan negatif ** 0.5 di Python jadi kompleks.
    amt_std_so_far = (
        max(state["amt_m2"] / state["txn_count"], 0.0) ** 0.5
        if state["txn_count"] > 0
        else 0.0
    )

    features = AggregateFeatures(
        txn_count_so_far=state["txn_count"],
        seconds_since_last_txn=transaction_dt - state["last_txn_dt"],
        amt_mean_so_far=state["amt_mean"],
        amt_std_so_far=amt_std_so_far,
        amt_max_so_far=state["amt_max"],
        amt_ratio_to_mean=transaction_amt / state["amt_mean"],
    )

    new_count = state["txn_count"] + 1
    delta = transaction_amt - state["amt_mean"]
    new_mean = state["amt_mean"] + delta / new_count
    # Algoritma Welford, bukan sum-of-squares mentah: presisi terjaga di entitas berhistori panjang.
    new_m2 = state["amt_m2"] + delta * (transaction_amt - new_mean)

    new_state = AggregateState(
        txn_count=new_count,
        amt_mean=new_mean,
        amt_m2=new_m2,
        amt_max=max(state["amt_max"], transaction_amt),
        last_txn_dt=transaction_dt,
    )

    return features, new_state
