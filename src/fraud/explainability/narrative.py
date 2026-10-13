"""Narasi LLM untuk item antrean review, disusun dari dokumen penjelasan SHAP.

Narasi ini pelengkap narasi templat, bukan penggantinya: kegagalan di sini hanya mengosongkan
narasi LLM, sedangkan narasi templat dan tabel SHAP tetap ada. LLM hanya menerima ringkasan
faktor teratas dengan angka yang sudah dihitung di kode, dan tidak menerima identitas
transaksi, skor tier 2, ataupun data transaksi mentah.

Keluaran LLM tidak dipercaya begitu saja. Setiap narasi diperiksa terhadap aturan yang bisa
diperiksa mesin; narasi yang melanggar ditolak dan dicoba ulang satu kali dengan alasan
penolakannya, lalu dibuang kalau tetap melanggar.
"""

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from fraud.explainability.template import (
    factor_label,
    format_value,
    group_members_text,
    share_text,
    total_contribution,
    usable_entries,
)

PROMPT_PATH = Path(__file__).parent / "prompts" / "review_narrative.txt"

# Token reasoning dihitung dalam batas ini, jadi batasnya jauh di atas panjang narasi supaya
# narasi tidak terpotong oleh reasoning.
MAX_OUTPUT_TOKENS = 1500
REASONING_EFFORT = "low"

# Satu percobaan ulang SDK untuk gangguan jaringan sesaat; lebih dari itu menahan satu batch
# terlalu lama karena panggilan dilakukan berurutan.
SDK_MAX_RETRIES = 1

MIN_NARRATIVE_CHARS = 40
MAX_NARRATIVE_CHARS = 1200

# Frasa yang melanggar prinsip bahasa: entity key bukan kartu atau nasabah, dan keputusan
# transaksi ini adalah approve, bukan decline.
FORBIDDEN_PHRASES = (
    "kartu ini",
    "nasabah ini",
    "pelanggan ini",
    "kunci entitas",
    "pasti fraud",
    "terbukti fraud",
    "ditolak",
    "decline",
)

_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Completion:
    """Hasil satu panggilan LLM.

    Attributes:
        text: Teks keluaran model.
        complete: False kalau keluaran terpotong oleh batas token.
    """

    text: str
    complete: bool


@dataclass(frozen=True)
class NarrativeResult:
    """Narasi LLM yang lolos pemeriksaan.

    Attributes:
        text: Isi narasi.
        model: Nama model yang menulisnya.
        attempts: Jumlah panggilan yang dibutuhkan, 1 atau 2.
    """

    text: str
    model: str
    attempts: int


# Menerima instruksi dan masukan, mengembalikan keluaran model; melempar exception kalau gagal.
Complete = Callable[[str, str], Completion]


class OpenAiCompleter:
    """Pemanggil Responses API untuk satu model.

    Args:
        client: Klien OpenAI dengan batas waktu dan percobaan ulang yang sudah diatur.
        model: Nama model, misal `gpt-5.6-terra`.
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self.model = model

    @classmethod
    def from_environment(cls) -> "OpenAiCompleter":
        """Membuat pemanggil dari `OPENAI_API_KEY`, `LLM_MODEL`, dan `LLM_TIMEOUT_SECONDS`."""
        client = OpenAI(
            timeout=float(os.environ["LLM_TIMEOUT_SECONDS"]), max_retries=SDK_MAX_RETRIES
        )
        return cls(client, os.environ["LLM_MODEL"])

    def __call__(self, instructions: str, input_text: str) -> Completion:
        response = self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
            reasoning={"effort": REASONING_EFFORT},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            # Data transaksi tidak perlu disimpan di sisi penyedia.
            store=False,
        )
        return Completion(text=response.output_text, complete=response.status == "completed")


def load_instructions(path: Path = PROMPT_PATH) -> str:
    """Membaca instruksi sistem dari berkas prompt."""
    return path.read_text().strip()


def build_input(explanation: dict[str, Any]) -> str | None:
    """Menyusun masukan LLM dari dokumen penjelasan.

    Porsi dan arah dihitung di sini, bukan oleh LLM. Identitas transaksi, skor, dan skala logit
    sengaja tidak disertakan.

    Returns:
        Teks JSON berisi faktor teratas, atau None kalau tidak ada faktor yang bisa dijelaskan.
    """
    if not explanation.get("available"):
        return None
    entries = usable_entries(explanation)
    total = total_contribution(entries)
    if not entries or total == 0.0:
        return None

    factors = []
    for entry in entries:
        factor: dict[str, Any] = {
            "nama": entry["name"],
            "label": factor_label(entry),
            "arah": "menaikkan risiko" if entry["contribution"] > 0 else "menurunkan risiko",
            "porsi": share_text(entry["contribution"], total),
        }
        if entry["kind"] == "group":
            factor["anggota"] = group_members_text(entry)
        else:
            factor["nilai"] = format_value(entry["value"])
        factors.append(factor)

    document: dict[str, Any] = {
        "keputusan": "disetujui dan ditandai untuk review",
        "jumlah_faktor": len(factors),
        "faktor": factors,
    }
    other_units = explanation["other"]["n_units"]
    if other_units:
        document["faktor_lain"] = f"{other_units} faktor lain yang lebih kecil tidak dirinci"
    return json.dumps(document, ensure_ascii=False, indent=2)


def violation_reason(narrative: str, input_text: str) -> str | None:
    """Memeriksa narasi terhadap aturan yang bisa diperiksa mesin.

    Returns:
        Alasan penolakan, atau None kalau narasi lolos.
    """
    text = narrative.strip()
    if len(text) < MIN_NARRATIVE_CHARS:
        return "Narasi terlalu pendek atau kosong."
    if len(text) > MAX_NARRATIVE_CHARS:
        return f"Narasi lebih dari {MAX_NARRATIVE_CHARS} karakter."

    lowered = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            return f'Narasi memuat frasa terlarang "{phrase}".'

    # Angka dari data, termasuk yang ada di nama dan label kolom, boleh muncul; yang lain
    # berarti model menambah atau mengubah angka.
    allowed = set(_NUMBER_PATTERN.findall(input_text))
    invented = sorted(set(_NUMBER_PATTERN.findall(text)) - allowed)
    if invented:
        return f"Narasi memuat angka yang tidak ada di data: {', '.join(invented)}."
    return None


def generate_review_narrative(
    complete: Complete, model: str, explanation: dict[str, Any]
) -> NarrativeResult | None:
    """Menulis narasi LLM untuk satu item review; tidak pernah melempar exception.

    Args:
        complete: Pemanggil LLM, biasanya `OpenAiCompleter`.
        model: Nama model yang dicatat bersama narasi.
        explanation: Dokumen penjelasan dari `fraud.explainability.tier1_shap`.

    Returns:
        Narasi yang lolos pemeriksaan, atau None kalau penjelasan tidak tersedia, panggilan
        gagal, atau narasi tetap melanggar setelah satu kali percobaan ulang.
    """
    input_text = build_input(explanation)
    if input_text is None:
        return None
    try:
        instructions = load_instructions()
        attempt_input = input_text
        for attempt in (1, 2):
            completion = complete(instructions, attempt_input)
            reason = (
                violation_reason(completion.text, input_text)
                if completion.complete
                else "Narasi terpotong sebelum selesai."
            )
            if reason is None:
                return NarrativeResult(completion.text.strip(), model, attempt)
            logger.warning("Narasi LLM ditolak (percobaan %d): %s", attempt, reason)
            attempt_input = (
                f"{input_text}\n\nNarasi sebelumnya ditolak karena: {reason} "
                "Tulis ulang dengan memperbaikinya."
            )
    except Exception:  # noqa: BLE001
        logger.exception("Panggilan LLM gagal")
    return None
