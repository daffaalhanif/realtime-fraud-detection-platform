"""API scoring: endpoint `POST /score` yang dipanggil sistem otorisasi pembayaran.

Dijalankan lewat: uv run uvicorn fraud.serving.main:app --host 127.0.0.1 --port 8000
"""

import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Security
from starlette.types import ASGIApp, Receive, Scope, Send

from fraud.schemas.score_response import ScoreResponse
from fraud.schemas.transaction import Transaction
from fraud.serving.api_key_auth import load_scoring_api_key, verify_api_key
from fraud.serving.decision import RequestTrace, decide
from fraud.serving.kafka_producer import ScoredProducer
from fraud.serving.model_loader import Tier1Model, load_production_tier1
from fraud.serving.retry_buffer import RetryBuffer

# Endpoint sinkron dijalankan di threadpool. Ukurannya ditetapkan eksplisit karena ukuran pool
# koneksi Redis diturunkan dari angka yang sama.
REQUEST_THREADS = 40

# Tiap thread request memegang paling banyak satu koneksi Redis, jadi menunggu koneksi tidak
# terjadi dalam kondisi normal; batas ini hanya penjaga supaya request tidak menggantung.
_REDIS_POOL_WAIT_SECONDS = 1

request_logger = logging.getLogger("fraud.serving.request")


class _ReceivedAtMiddleware:
    """Mencatat waktu request tiba, sebelum body dibaca dan divalidasi.

    Ditulis sebagai middleware ASGI murni karena middleware berbasis `BaseHTTPMiddleware`
    menambah overhead yang justru ikut terukur sebagai bagian langkah ini.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["received_at"] = time.perf_counter()
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Menyiapkan seluruh dependensi saat start-up dan menutupnya dengan urutan yang aman.

    Konfigurasi yang salah (API key, alamat registry, alias produksi) menggagalkan start-up,
    supaya tidak ada satu pun request yang dilayani dengan setelan keliru. Kafka yang mati
    saat start-up hanya dicatat, karena tidak boleh menahan jalur keputusan.
    """
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_scoring_api_key()
    anyio.to_thread.current_default_thread_limiter().total_tokens = REQUEST_THREADS

    model = load_production_tier1()
    redis_pool = redis.BlockingConnectionPool(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        max_connections=REQUEST_THREADS,
        timeout=_REDIS_POOL_WAIT_SECONDS,
    )
    retry_buffer = RetryBuffer(Path(os.environ["RETRY_BUFFER_PATH"]))
    producer = ScoredProducer(os.environ["KAFKA_BOOTSTRAP_SERVERS"], retry_buffer.enqueue)
    retry_buffer.start(producer.deliver)

    app.state.model = model
    app.state.redis_client = redis.Redis(connection_pool=redis_pool)
    app.state.producer = producer
    try:
        yield
    finally:
        # Penutupan producer masih menyerahkan pesan tertahan ke antrean cadangan, jadi
        # pengirim ulang dihentikan lebih dulu dan berkas antrean ditutup paling akhir.
        retry_buffer.stop()
        producer.close()
        retry_buffer.close()
        redis_pool.disconnect()


app = FastAPI(title="Fraud Scoring API", lifespan=lifespan)
app.add_middleware(_ReceivedAtMiddleware)


@app.post("/score", response_model=ScoreResponse, dependencies=[Security(verify_api_key)])
def score(transaction: Transaction, request: Request) -> ScoreResponse:
    """Menilai satu transaksi dan mengembalikan salah satu dari tiga kelas keputusan."""
    trace = RequestTrace()
    trace.durations_ms["validation_and_framework"] = (
        time.perf_counter() - request.state.received_at
    ) * 1000
    # Referensi model dibaca sekali, supaya satu request memakai satu versi model sampai selesai.
    model: Tier1Model = request.app.state.model
    response = decide(
        transaction, model, request.app.state.redis_client, request.app.state.producer, trace
    )
    request_logger.info(
        json.dumps(
            {
                "transaction_id": response.transaction_id,
                "decision": response.decision.value,
                "model_version": model.model_version,
                "total_ms": round((time.perf_counter() - request.state.received_at) * 1000, 3),
                "steps_ms": {name: round(ms, 3) for name, ms in trace.durations_ms.items()},
                "marks": trace.marks,
            }
        )
    )
    return response
