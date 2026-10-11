"""Narasi berbasis templat, disusun langsung dari dokumen penjelasan SHAP tanpa memanggil LLM.

Narasi ini dibuat untuk setiap item antrean review, sehingga analis selalu punya penjelasan
berbahasa manusia walau modul narasi LLM gagal atau lambat. Istilah teknis yang lazim dalam
bahasa Inggris (entity key, email domain, device, review) sengaja tidak diterjemahkan.

Narasi merujuk pada pola aktivitas pada entity key, bukan pada satu kartu atau nasabah, karena
satu entity key bisa mewakili sekelompok kartu. Makna kolom anonim tidak ditebak: kolom itu
hanya disebut anonim, dengan arah dan besar pengaruhnya.
"""

import math
import re
from typing import Any

# Tiap arah dibatasi supaya narasi tetap ringkas; tabel SHAP lengkap ada di samping narasi.
MAX_FACTORS_PER_DIRECTION = 3
MAX_SHOWN_MEMBERS = 3

UNAVAILABLE_NARRATIVE = (
    "Kontribusi fitur tidak tersedia untuk item ini. Tinjau transaksi secara manual."
)

# Hanya kolom yang artinya pasti dari namanya atau dari fungsi yang menghitungnya.
_LABELS = {
    "txn_count_so_far": "jumlah transaksi sebelumnya pada entity key",
    "seconds_since_last_txn": "jeda (detik) sejak transaksi sebelumnya pada entity key",
    "amt_mean_so_far": "rata-rata nominal historis entity key",
    "amt_std_so_far": "simpangan baku nominal historis entity key",
    "amt_max_so_far": "nominal tertinggi historis entity key",
    "amt_ratio_to_mean": "rasio nominal transaksi terhadap rata-rata historis entity key",
    "count_last_1h": "jumlah transaksi pada entity key dalam 1 jam terakhir",
    "count_last_24h": "jumlah transaksi pada entity key dalam 24 jam terakhir",
    "count_last_7d": "jumlah transaksi pada entity key dalam 7 hari terakhir",
    "amt_sum_last_1h": "total nominal pada entity key dalam 1 jam terakhir",
    "amt_sum_last_24h": "total nominal pada entity key dalam 24 jam terakhir",
    "amt_sum_last_7d": "total nominal pada entity key dalam 7 hari terakhir",
    "TransactionAmt": "nominal transaksi",
    "ProductCD": "kode produk ProductCD",
    "P_emaildomain": "email domain P_emaildomain",
    "R_emaildomain": "email domain R_emaildomain",
    "DeviceType": "tipe device DeviceType",
    "DeviceInfo": "info device DeviceInfo",
}

# Grup C, D, M, V, dan id_*: tidak punya makna bisnis yang dijelaskan sumber data.
_ANONYMOUS_PATTERN = re.compile(r"(?:[CDMV]\d+|id_\d+)")


def _is_anonymous(entry: dict[str, Any]) -> bool:
    return entry["kind"] == "group" or _ANONYMOUS_PATTERN.fullmatch(entry["name"]) is not None


def _format_number(value: float) -> str:
    """Angka berformat Indonesia: titik pemisah ribuan, koma desimal."""
    text = f"{int(value):,}" if value == int(value) else f"{value:,.2f}"
    return text.replace(",", "#").replace(".", ",").replace("#", ".")


def _format_value(value: Any) -> str:
    if value is None:
        return "kosong"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return _format_number(float(value))
    return str(value)


def _describe(entry: dict[str, Any], share: str) -> str:
    """Satu faktor sebagai frasa: label, nilai (kalau fitur tunggal), dan porsi pengaruhnya."""
    if entry["kind"] == "group":
        members = entry["members"]
        shown = ", ".join(members[:MAX_SHOWN_MEMBERS])
        if len(members) > MAX_SHOWN_MEMBERS:
            shown += f", dan {len(members) - MAX_SHOWN_MEMBERS} lainnya"
        return f"kelompok kolom anonim {entry['name']} ({shown}; {share})"

    name = entry["name"]
    if name in _LABELS:
        label = _LABELS[name]
    elif _is_anonymous(entry):
        label = f"kolom anonim {name}"
    else:
        label = f"kolom {name}"
    return f"{label} (nilai {_format_value(entry['value'])}; {share})"


def _share_text(contribution: float, total: float) -> str:
    share = abs(contribution) / total * 100
    return "<1%" if share < 1 else f"{round(share)}%"


def build_template_narrative(explanation: dict[str, Any]) -> str:
    """Menyusun narasi dari dokumen penjelasan `fraud.explainability.tier1_shap`.

    Args:
        explanation: Dokumen penjelasan satu item, tersedia maupun tidak.

    Returns:
        Narasi berbahasa Indonesia: faktor yang paling menaikkan dan menurunkan risiko beserta
        porsi pengaruhnya di antara faktor terbesar, atau pesan bahwa kontribusi fitur tidak
        tersedia.
    """
    if not explanation.get("available"):
        return UNAVAILABLE_NARRATIVE
    # Kontribusi tak berhingga atau NaN tidak bisa dijadikan porsi, jadi entrinya dibuang.
    entries = [
        e
        for e in explanation["contributions"]
        if math.isfinite(e["contribution"]) and e["contribution"] != 0.0
    ]
    total = math.fsum(abs(e["contribution"]) for e in entries)
    if not entries or total == 0.0:
        return UNAVAILABLE_NARRATIVE

    # Daftar kontribusi sudah urut dari pengaruh terbesar.
    raising = [e for e in entries if e["contribution"] > 0][:MAX_FACTORS_PER_DIRECTION]
    lowering = [e for e in entries if e["contribution"] < 0][:MAX_FACTORS_PER_DIRECTION]

    sentences = ["Transaksi ini disetujui dan ditandai untuk review."]
    for heading, group in (
        ("menaikkan", raising),
        ("menurunkan", lowering),
    ):
        if group:
            factors = "; ".join(
                _describe(e, _share_text(e["contribution"], total)) for e in group
            )
            sentences.append(f"Faktor yang paling {heading} risiko: {factors}.")
    sentences.append(
        f"Persentase adalah porsi pengaruh di antara {len(entries)} faktor terbesar, "
        "tidak termasuk faktor lain yang lebih kecil."
    )
    if sum(_is_anonymous(e) for e in entries) > len(entries) / 2:
        sentences.append(
            "Sebagian besar faktor berupa kolom anonim tanpa makna bisnis yang diketahui, "
            "sehingga penjelasan terbatas pada arah dan besar pengaruhnya."
        )
    return " ".join(sentences)
