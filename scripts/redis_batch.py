"""Logika bersama pemrosesan batch entitas dan penulisan ke Redis.

Dipakai identik oleh scripts/load_initial_data.py (sumber data: CSV mentah
yang baru di-join dan divalidasi) maupun scripts/recover_redis_from_parquet.py
(sumber data: Parquet yang sudah tervalidasi). Modul terpisah ini mencegah
dua skrip punya implementasi loop batch yang bisa diam-diam menyimpang satu
sama lain seiring waktu, dengan alasan yang sama kenapa fitur agregat
sendiri cuma punya satu implementasi.
"""

import json
from collections import deque
from typing import Any, cast

import numpy as np
import pandas as pd
import redis

from fraud.features.aggregate import AggregateState, process_transaction
from fraud.features.cold_start import get_default_state

# Batas retensi Redis LIST per entitas, parameter penyimpanan, bukan
# seq_len model tier 2 (keputusan terpisah lewat eksperimen tahap model).
# Dilebihkan jauh di atas kebutuhan wajar (median histori entitas cuma 4
# transaksi) supaya tidak membatasi eksperimen seq_len nanti.
SEQUENCE_RETENTION_LIMIT = 200

REDIS_AGGREGATE_KEY_PREFIX = "agg"
REDIS_SEQUENCE_KEY_PREFIX = "seq"


def _json_default(value: object) -> object:
    """Konversi tipe numpy yang tidak dikenali json.dumps secara langsung."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        as_float = float(value)
        return None if as_float != as_float else as_float
    raise TypeError(f"Tidak bisa serialize tipe {type(value)}")


def _row_to_json(row: dict) -> str:
    """Serialize satu baris transaksi mentah, NaN menjadi null JSON."""
    clean = {
        key: (None if isinstance(value, float) and value != value else value)
        for key, value in row.items()
    }
    return json.dumps(clean, default=_json_default)


def compute_features_and_sequences(
    df: pd.DataFrame,
) -> tuple[dict[int, AggregateState], dict[int, list[str]]]:
    """Proses seluruh transaksi kronologis per entitas secara batch.

    Memanggil fraud.features.aggregate.process_transaction identik dengan
    jalur incremental, satu transaksi satu kali, urut TransactionDT naik.
    Histori mentah tiap entitas ditampung di deque berbatas dulu (bukan
    langsung di-serialize), supaya baris yang nanti tergusur dari jendela
    retensi tidak ikut memboroskan biaya serialize JSON yang sia-sia -
    ini penting untuk entitas volume tinggi yang bisa punya belasan ribu
    transaksi padahal cuma beberapa ratus terakhir yang disimpan.

    Args:
        df: Data mentah hasil join, sudah tervalidasi dan terurut
            kronologis berdasarkan TransactionDT, tanpa kolom tambahan
            di luar hasil join (misal kolom partisi Parquet).

    Returns:
        Pasangan `(states, sequences)`. `states` adalah state agregat
        TERAKHIR tiap entitas setelah seluruh histori diproses, siap
        ditulis ke Redis HASH. `sequences` adalah histori transaksi
        mentah terbaru tiap entitas (JSON per baris), siap ditulis ke
        Redis LIST.
    """
    states: dict[int, AggregateState] = {}
    raw_sequences: dict[int, deque] = {}

    columns = df.columns.tolist()
    for row_tuple in df.itertuples(index=False, name=None):
        raw_row = dict(zip(columns, row_tuple))
        card1 = int(raw_row["card1"])
        amt = float(raw_row["TransactionAmt"])
        dt = int(raw_row["TransactionDT"])

        state = states.get(card1)
        if state is None:
            state = get_default_state()
        _, new_state = process_transaction(state, amt, dt)
        states[card1] = new_state

        sequence = raw_sequences.setdefault(
            card1, deque(maxlen=SEQUENCE_RETENTION_LIMIT)
        )
        sequence.append(raw_row)

    sequences = {
        card1: [_row_to_json(row) for row in rows]
        for card1, rows in raw_sequences.items()
    }
    return states, sequences


def write_redis(
    states: dict[int, AggregateState],
    sequences: dict[int, list[str]],
    redis_client: redis.Redis,
) -> None:
    """Tulis state agregat dan sequence ke Redis lewat pipeline."""
    pipeline = redis_client.pipeline(transaction=False)
    for card1, state in states.items():
        key = f"{REDIS_AGGREGATE_KEY_PREFIX}:{card1}"
        # Stub redis-py menolak tipe untuk mapping; cast Any murni type checker, teruji di runtime.
        pipeline.hset(key, mapping=cast(Any, dict(state)))
    pipeline.execute()

    pipeline = redis_client.pipeline(transaction=False)
    for card1, sequence in sequences.items():
        key = f"{REDIS_SEQUENCE_KEY_PREFIX}:{card1}"
        pipeline.delete(key)
        pipeline.rpush(key, *sequence)
    pipeline.execute()
