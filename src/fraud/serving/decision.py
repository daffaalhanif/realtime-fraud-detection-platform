"""Alur keputusan satu transaksi, dari data mentah sampai salah satu dari tiga kelas keputusan.

Urutan langkah setelah keputusan diambil tidak boleh dibalik: hasil dititip ke Kafka lebih
dulu, baru online store diperbarui. Kalau online store diperbarui lebih dulu lalu penitipan
gagal tanpa jejak, fitur agregat entitas sudah berubah oleh transaksi yang tidak pernah
tercatat di jalur audit.

Pemanggilan tier 2 untuk zona abu-abu belum ada di alur ini, sehingga ambang bawah tier 1
belum dipakai.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import redis

from fraud.features.aggregate import AggregateState, process_transaction
from fraud.features.cold_start import get_default_state
from fraud.features.encoding import FeatureSpec, build_model_input
from fraud.features.online_store import (
    SEQUENCE_RETENTION_LIMIT,
    aggregate_key,
    parse_sequence_entries,
    sequence_entry,
    sequence_key,
    state_from_hash,
    state_to_hash,
)
from fraud.features.window import requires_window_features, window_features
from fraud.schemas.score_response import DecisionClass, ScoreResponse
from fraud.schemas.scored_message import ScoredMessage
from fraud.schemas.transaction import Transaction
from fraud.serving.idempotency import get_decided, record_decision
from fraud.serving.kafka_producer import ScoredProducer
from fraud.serving.model_loader import Tier1Model, Tier1Thresholds

# Tiap putaran WATCH meloloskan minimal satu penulis, jadi penulis yang terus kalah tertunda
# paling banyak sejumlah penulis serentak pada kunci entitas yang sama. Batas ini hanya
# penjaga loop tanpa ujung dan sengaja jauh di atas jumlah thread request yang bisa berebut.
_MAX_FOLD_ATTEMPTS = 1_000

logger = logging.getLogger(__name__)


class RequestTrace:
    """Catatan durasi tiap langkah dan penanda kejadian dalam satu request scoring.

    Dicatat di log satu baris per request, supaya langkah yang menghabiskan jatah latensi
    bisa dilacak saat uji beban berjalan.

    Attributes:
        durations_ms: Durasi per nama langkah dalam milidetik, urut sesuai eksekusi.
        marks: Penanda kejadian yang memengaruhi lama request, misal hasil titip Kafka.
    """

    def __init__(self) -> None:
        self.durations_ms: dict[str, float] = {}
        self.marks: dict[str, str | int | bool] = {}

    @contextmanager
    def step(self, name: str) -> Generator[None]:
        """Mengukur durasi blok kode sebagai satu langkah bernama."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations_ms[name] = (time.perf_counter() - start) * 1000

    def mark(self, name: str, value: str | int | bool) -> None:
        """Mencatat satu penanda kejadian."""
        self.marks[name] = value


def classify(score: float, thresholds: Tier1Thresholds) -> DecisionClass:
    """Menentukan kelas keputusan dari skor terkalibrasi tier 1.

    Aturannya sama dengan yang dipakai saat ambang dipilih: tolak kalau skor mencapai ambang
    tolak, setujui dan tandai review kalau skor mencapai ambang review, selain itu setujui.

    Args:
        score: Peluang fraud terkalibrasi.
        thresholds: Ambang keputusan versi model yang menghasilkan skor.

    Returns:
        Salah satu dari tiga kelas keputusan.
    """
    if score >= thresholds.reject:
        return DecisionClass.DECLINE
    if score >= thresholds.review:
        return DecisionClass.APPROVE_REVIEW
    return DecisionClass.APPROVE


def decide(
    transaction: Transaction,
    model: Tier1Model,
    redis_client: redis.Redis,
    producer: ScoredProducer,
    trace: RequestTrace | None = None,
) -> ScoreResponse:
    """Memutuskan satu transaksi dan mencatat hasilnya.

    Transaksi yang sudah pernah diputuskan dijawab dengan keputusan yang sama tanpa dihitung
    ulang dan tanpa pesan baru ke Kafka.

    Args:
        transaction: Data transaksi mentah dari request.
        model: Model tier 1 yang aktif saat request ini masuk.
        redis_client: Koneksi online store, dengan `decode_responses=False`.
        producer: Producer topik `scored`.
        trace: Penampung catatan waktu per langkah. Dibuat sendiri kalau tidak diberikan.

    Returns:
        Keputusan untuk transaksi ini.
    """
    trace = RequestTrace() if trace is None else trace

    with trace.step("idempotency_check"):
        previous = get_decided(redis_client, transaction.TransactionID)
    if previous is not None:
        trace.mark("already_decided", True)
        return previous

    model_input = _features_for(transaction, model.feature_spec, redis_client, trace)
    with trace.step("tier1_score"):
        score = model.score(model_input)
        decision = classify(score, model.thresholds)
    response = ScoreResponse(
        transaction_id=transaction.TransactionID, decision=decision, score=score
    )

    with trace.step("kafka_send"):
        outcome = producer.send(
            ScoredMessage(
                transaction=transaction,
                decision=decision,
                tier1_score=score,
                model_version=model.model_version,
                scored_at=datetime.now(UTC),
            )
        )
    trace.mark("kafka_outcome", outcome.value)

    with trace.step("redis_update"):
        # Request duplikat yang datang bersamaan bisa lolos pengecekan di awal; hanya satu
        # yang tercatat, dan hanya dia yang boleh melipat transaksi ini ke online store.
        earlier = record_decision(redis_client, response)
        if earlier is None:
            trace.mark("fold_attempts", _fold_into_online_store(redis_client, transaction))
    if earlier is not None:
        trace.mark("already_decided", True)
        return earlier
    return response


def _read_entity(
    redis_client: redis.Redis, card1: int, with_history: bool
) -> tuple[AggregateState, list[Any]]:
    """State agregat kunci entitas dan, bila diminta, elemen sequence terakhirnya.

    Keduanya diambil dalam satu kiriman supaya hanya ada satu perjalanan jaringan ke Redis.

    Returns:
        State agregat (default kalau entitas belum pernah tercatat) dan elemen Redis LIST,
        kosong kalau `with_history` False.
    """
    if with_history:
        with redis_client.pipeline(transaction=False) as pipe:
            pipe.hgetall(aggregate_key(card1))
            pipe.lrange(sequence_key(card1), -SEQUENCE_RETENTION_LIMIT, -1)
            raw_state, raw_history = pipe.execute()
    else:
        raw_state, raw_history = redis_client.hgetall(aggregate_key(card1)), []
    state = state_from_hash(cast(dict[Any, Any], raw_state))
    return (get_default_state() if state is None else state), list(raw_history)


def _features_for(
    transaction: Transaction, spec: FeatureSpec, redis_client: redis.Redis, trace: RequestTrace
) -> np.ndarray:
    """Vektor input model satu transaksi, persis sesuai kontrak input versi model yang aktif.

    Fitur jendela waktu hanya dihitung, dan sequence hanya dibaca dari Redis, kalau kontrak
    input memintanya; versi model tanpa fitur itu tidak menanggung biayanya.
    """
    with_history = requires_window_features(spec["input_columns"])
    with trace.step("redis_read"):
        state, raw_history = _read_entity(redis_client, transaction.card1, with_history)
    values: dict[str, Any] = transaction.model_dump()
    if with_history:
        with trace.step("window_features"):
            history = [
                (int(entry["TransactionDT"]), float(entry["TransactionAmt"]))
                for entry in parse_sequence_entries(raw_history)
            ]
            values.update(window_features(history, transaction.TransactionDT))
        trace.mark("history_length", len(history))
    with trace.step("feature_compute"):
        features, _ = process_transaction(
            state, transaction.TransactionAmt, transaction.TransactionDT
        )
        values.update(features)
        return build_model_input({name: [values[name]] for name in spec["input_columns"]}, spec)


def _fold_into_online_store(redis_client: redis.Redis, transaction: Transaction) -> int:
    """Melipat transaksi ke state agregat dan sequence entitasnya tanpa menimpa tulisan lain.

    State dibaca ulang di sini, tidak memakai state saat scoring: transaksi lain dari kunci
    entitas yang sama bisa sudah tercatat sejak itu. WATCH membatalkan penulisan kalau state
    berubah di antara baca dan tulis, lalu pelipatan diulang dari state terbaru.

    Returns:
        Jumlah percobaan yang dipakai; lebih dari satu berarti ada tulisan lain yang bentrok.
    """
    agg_key = aggregate_key(transaction.card1)
    seq_key = sequence_key(transaction.card1)
    entry = sequence_entry(transaction.model_dump())
    with redis_client.pipeline() as pipe:
        for attempt in range(1, _MAX_FOLD_ATTEMPTS + 1):
            try:
                pipe.watch(agg_key)
                state = state_from_hash(cast(dict[Any, Any], pipe.hgetall(agg_key)))
                _, new_state = process_transaction(
                    get_default_state() if state is None else state,
                    transaction.TransactionAmt,
                    transaction.TransactionDT,
                )
                pipe.multi()
                # Stub redis-py menolak tipe untuk mapping; cast Any murni type checker.
                pipe.hset(agg_key, mapping=cast(Any, state_to_hash(new_state)))
                pipe.rpush(seq_key, entry)
                pipe.ltrim(seq_key, -SEQUENCE_RETENTION_LIMIT, -1)
                pipe.execute()
                return attempt
            except redis.WatchError:
                continue
    logger.error(
        "State agregat card1=%s tidak diperbarui untuk TransactionID %s setelah %d percobaan",
        transaction.card1,
        transaction.TransactionID,
        _MAX_FOLD_ATTEMPTS,
    )
    return _MAX_FOLD_ATTEMPTS
