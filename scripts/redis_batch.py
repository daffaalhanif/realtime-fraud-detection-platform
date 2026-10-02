"""Logika bersama pemrosesan batch entitas dan penulisan ke Redis.

Dipakai identik oleh scripts/load_initial_data.py (sumber data: CSV mentah
yang baru di-join dan divalidasi) maupun scripts/recover_redis_from_parquet.py
(sumber data: Parquet yang sudah tervalidasi). Modul terpisah ini mencegah
dua skrip punya implementasi loop batch yang bisa diam-diam menyimpang satu
sama lain seiring waktu, dengan alasan yang sama kenapa fitur agregat
sendiri cuma punya satu implementasi.
"""

from collections import deque
from typing import Any, cast

import pandas as pd
import redis

from fraud.features.aggregate import AggregateState, process_transaction
from fraud.features.cold_start import get_default_state
from fraud.features.online_store import (
    SEQUENCE_RETENTION_LIMIT,
    aggregate_key,
    sequence_entry,
    sequence_key,
    state_to_hash,
)


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
        mentah terbaru tiap entitas (JSON per baris, field skema
        Transaction saja tanpa label), siap ditulis ke Redis LIST.
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
        card1: [sequence_entry(row) for row in rows]
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
        # Stub redis-py menolak tipe untuk mapping; cast Any murni type checker, teruji di runtime.
        pipeline.hset(aggregate_key(card1), mapping=cast(Any, state_to_hash(state)))
    pipeline.execute()

    pipeline = redis_client.pipeline(transaction=False)
    for card1, sequence in sequences.items():
        key = sequence_key(card1)
        pipeline.delete(key)
        pipeline.rpush(key, *sequence)
    pipeline.execute()
