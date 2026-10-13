"""Pencatatan panggilan LLM narasi ke Langfuse.

Yang dicatat persis yang dikirim ke LLM beserta keluarannya, waktu, dan jumlah token, tanpa
pengenal item. Kegagalan pencatatan tidak boleh menggagalkan narasi: semua panggilan ke
Langfuse dibungkus supaya error-nya hanya masuk log. Tanpa kunci Langfuse, pencatatan mati
dan pemanggil LLM dipakai apa adanya.
"""

import logging
import os
from collections.abc import Callable

from langfuse import Langfuse

from fraud.explainability.narrative import Complete, Completion

GENERATION_NAME = "review-narrative"

logger = logging.getLogger(__name__)


def build_langfuse_client() -> Langfuse | None:
    """Membuat klien dari `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

    Returns:
        Klien Langfuse, atau None kalau salah satu kunci tidak diset.
    """
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        logger.warning("Kunci Langfuse tidak diset: pencatatan panggilan LLM dimatikan.")
        return None
    return Langfuse()


def shutdown_langfuse(client: Langfuse | None) -> None:
    """Mengirim sisa data yang tertahan lalu menghentikan klien; tidak melempar exception."""
    if client is not None:
        _safely(client.shutdown)


def traced(complete: Complete, model: str, client: Langfuse | None) -> Complete:
    """Membungkus pemanggil LLM supaya tiap panggilan tercatat sebagai satu generation.

    Args:
        complete: Pemanggil LLM yang dibungkus.
        model: Nama model yang dicatat.
        client: Klien Langfuse; None berarti pencatatan mati dan `complete` dikembalikan apa
            adanya.

    Returns:
        Pemanggil dengan kontrak sama seperti `complete`, termasuk exception-nya.
    """
    if client is None:
        return complete

    def call(instructions: str, input_text: str) -> Completion:
        generation = _safely(
            lambda: client.start_observation(
                name=GENERATION_NAME,
                as_type="generation",
                model=model,
                input={"instructions": instructions, "input": input_text},
            )
        )
        try:
            completion = complete(instructions, input_text)
        except Exception as error:
            if generation is not None:
                _safely(
                    lambda: generation.update(level="ERROR", status_message=str(error)).end()
                )
            raise
        if generation is not None:
            usage = {
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
            }
            _safely(
                lambda: generation.update(
                    output=completion.text,
                    usage_details={k: v for k, v in usage.items() if v is not None},
                    metadata={"complete": completion.complete},
                ).end()
            )
        return completion

    return call


def _safely[T](action: Callable[[], T]) -> T | None:
    try:
        return action()
    except Exception:  # noqa: BLE001
        logger.warning("Pencatatan ke Langfuse gagal", exc_info=True)
        return None
