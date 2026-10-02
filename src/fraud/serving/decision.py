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

from fraud.features.aggregate import AggregateFeatures, AggregateState, process_transaction
from fraud.features.cold_start import get_default_state
from fraud.features.encoding import FeatureSpec, build_model_input
from fraud.features.online_store import (
    SEQUENCE_RETENTION_LIMIT,
    aggregate_key,
    sequence_entry,
    sequence_key,
    state_from_hash,
    state_to_hash,
)
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

    with trace.step("redis_read"):
        state = _load_state(redis_client, transaction.card1)
    with trace.step("feature_compute"):
        features, _ = process_transaction(
            state, transaction.TransactionAmt, transaction.TransactionDT
        )
        model_input = _model_input(transaction, features, model.feature_spec)
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


def _load_state(redis_client: redis.Redis, card1: int) -> AggregateState:
    """State agregat terakhir kunci entitas, atau nilai default kalau belum pernah tercatat."""
    raw = cast(dict[Any, Any], redis_client.hgetall(aggregate_key(card1)))
    state = state_from_hash(raw)
    return get_default_state() if state is None else state


def _model_input(
    transaction: Transaction, features: AggregateFeatures, spec: FeatureSpec
) -> np.ndarray:
    """Menyusun vektor input model satu baris dari transaksi dan fitur agregatnya."""
    values = {**transaction.model_dump(), **features}
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
