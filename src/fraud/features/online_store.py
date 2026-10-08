"""Tata letak data per entitas di online store (Redis), dipakai jalur batch maupun serving.

Jalur batch (pemuatan awal dan pemulihan dari Parquet) dan jalur serving menulis key yang
sama. Kalau keduanya punya definisi sendiri dan suatu saat berbeda, serving akan membaca key
kosong dan memperlakukan semua entitas sebagai entitas baru tanpa pesan error apa pun.
Modul ini hanya mengatur format, tanpa membuka koneksi Redis.
"""

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from fraud.features.aggregate import AggregateState
from fraud.schemas.transaction import Transaction

AGGREGATE_KEY_PREFIX = "agg"
SEQUENCE_KEY_PREFIX = "seq"

# Batas retensi Redis LIST per entitas, parameter penyimpanan, bukan seq_len model tier 2
# (keputusan terpisah lewat eksperimen). Dilebihkan jauh di atas kebutuhan wajar (median
# histori entitas cuma 4 transaksi) supaya tidak membatasi eksperimen seq_len nanti.
SEQUENCE_RETENTION_LIMIT = 200

_SEQUENCE_FIELDS = tuple(Transaction.model_fields)


def aggregate_key(card1: int) -> str:
    """Key Redis HASH berisi state agregat satu kunci entitas."""
    return f"{AGGREGATE_KEY_PREFIX}:{card1}"


def sequence_key(card1: int) -> str:
    """Key Redis LIST berisi histori transaksi terbaru satu kunci entitas."""
    return f"{SEQUENCE_KEY_PREFIX}:{card1}"


def state_to_hash(state: AggregateState) -> dict[str, int | float]:
    """Mengubah state agregat menjadi isi Redis HASH."""
    return {
        "txn_count": state["txn_count"],
        "amt_mean": state["amt_mean"],
        "amt_m2": state["amt_m2"],
        "amt_max": state["amt_max"],
        "last_txn_dt": state["last_txn_dt"],
    }


def state_from_hash(raw: Mapping[Any, Any]) -> AggregateState | None:
    """Mengurai isi Redis HASH kembali menjadi state agregat.

    Args:
        raw: Hasil `HGETALL`, key dan value boleh bertipe `str` maupun `bytes`.

    Returns:
        State agregat, atau `None` kalau HASH kosong (entitas belum pernah tercatat).
    """
    if not raw:
        return None
    fields = {(k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()}
    return AggregateState(
        txn_count=int(fields["txn_count"]),
        amt_mean=float(fields["amt_mean"]),
        amt_m2=float(fields["amt_m2"]),
        amt_max=float(fields["amt_max"]),
        last_txn_dt=int(fields["last_txn_dt"]),
    )


def _to_json_value(value: Any) -> Any:
    """Menyeragamkan nilai dari pandas maupun pydantic: tipe numpy ke tipe Python, NaN ke None."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def sequence_entry(fields: Mapping[str, Any]) -> str:
    """Membentuk satu elemen Redis LIST dari data mentah satu transaksi.

    Hanya field skema `Transaction` yang disimpan. Label fraud sengaja tidak ikut: saat
    transaksi dinilai labelnya belum diketahui, dan sequence ini menjadi input model.

    Args:
        fields: Data mentah satu transaksi, baik baris Parquet maupun `Transaction`
            hasil `model_dump()`. Kolom di luar skema diabaikan.

    Returns:
        JSON satu transaksi dengan urutan field mengikuti skema `Transaction`.
    """
    return json.dumps({name: _to_json_value(fields[name]) for name in _SEQUENCE_FIELDS})


def parse_sequence_entries(raw_entries: Iterable[bytes | str]) -> list[dict[str, Any]]:
    """Mengurai elemen Redis LIST hasil `sequence_entry` kembali menjadi data per transaksi.

    Args:
        raw_entries: Elemen `LRANGE`, bytes maupun str, urutan dipertahankan.

    Returns:
        Satu dict per transaksi dengan field skema `Transaction`; nilai kosong berupa `None`.
    """
    return [json.loads(entry) for entry in raw_entries]
