import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import OpenAI

from fraud.explainability.narrative import (
    MAX_OUTPUT_TOKENS,
    Completion,
    NarrativeResult,
    OpenAiCompleter,
    build_input,
    generate_review_narrative,
    violation_reason,
)

GOOD = (
    "Transaksi ini disetujui dan ditandai untuk review. Faktor yang paling menaikkan risiko "
    "adalah email domain P_emaildomain dengan nilai gmail.com (porsi 67%), sedangkan kode "
    "produk ProductCD bernilai W menurunkan risiko (porsi 33%)."
)


def _explanation() -> dict[str, Any]:
    return {
        "available": True,
        "raw_score": -1.5,
        "base_value": -3.0,
        "unexplained": 0.01,
        "other": {"n_units": 4, "contribution": 0.1},
        "contributions": [
            {"name": "P_emaildomain", "kind": "feature", "value": "gmail.com", "contribution": 2.0},
            {"name": "ProductCD", "kind": "feature", "value": "W", "contribution": -1.0},
        ],
    }


class _Script:
    """Pemanggil LLM tiruan yang mengembalikan jawaban berurutan dan mencatat masukannya."""

    def __init__(self, *answers: Completion | Exception) -> None:
        self._answers = list(answers)
        self.inputs: list[str] = []

    def __call__(self, instructions: str, input_text: str) -> Completion:
        self.inputs.append(input_text)
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _ok(text: str = GOOD) -> Completion:
    return Completion(text=text, complete=True)


def test_good_narrative_is_returned_on_first_attempt() -> None:
    script = _Script(_ok())

    result = generate_review_narrative(script, "m", _explanation())

    assert result == NarrativeResult(GOOD, "m", 1)
    assert len(script.inputs) == 1


def test_violation_is_retried_once_with_the_reason() -> None:
    script = _Script(_ok(GOOD.replace("Transaksi ini", "Kartu ini")), _ok())

    result = generate_review_narrative(script, "m", _explanation())

    assert result is not None and result.attempts == 2
    assert 'frasa terlarang "kartu ini"' in script.inputs[1]
    assert script.inputs[1].startswith(script.inputs[0])


def test_two_violations_give_no_narrative() -> None:
    bad = _ok("terlalu pendek")

    assert generate_review_narrative(_Script(bad, bad), "m", _explanation()) is None


def test_truncated_output_is_rejected_then_retried() -> None:
    script = _Script(Completion(text=GOOD, complete=False), _ok())

    result = generate_review_narrative(script, "m", _explanation())

    assert result is not None and result.attempts == 2
    assert "terpotong" in script.inputs[1]


def test_api_failure_gives_no_narrative_and_does_not_raise() -> None:
    assert generate_review_narrative(_Script(TimeoutError()), "m", _explanation()) is None


def test_unavailable_explanation_never_calls_the_llm() -> None:
    script = _Script()

    assert generate_review_narrative(script, "m", {"available": False}) is None
    assert script.inputs == []


def test_invented_number_is_rejected() -> None:
    input_text = build_input(_explanation())
    assert input_text is not None

    reason = violation_reason(GOOD.replace("67%", "70%"), input_text)

    assert reason is not None and "70" in reason


def test_copied_numbers_and_column_names_are_allowed() -> None:
    explanation = _explanation()
    explanation["contributions"].append(
        {"name": "count_last_1h", "kind": "feature", "value": 1500.0, "contribution": 0.5}
    )
    input_text = build_input(explanation)
    assert input_text is not None
    narrative = (
        "Transaksi ini disetujui dan ditandai untuk review. Email domain P_emaildomain "
        "menaikkan risiko (57%), jumlah transaksi 1 jam terakhir bernilai 1.500 juga "
        "menaikkan (14%), sedangkan kode produk ProductCD menurunkan risiko (29%)."
    )

    assert violation_reason(narrative, input_text) is None


@pytest.mark.parametrize("phrase", ["kartu ini", "nasabah ini", "kunci entitas", "ditolak"])
def test_forbidden_phrases_are_rejected(phrase: str) -> None:
    input_text = build_input(_explanation())
    assert input_text is not None

    assert violation_reason(f"{GOOD} Ini {phrase}.", input_text) is not None


def test_input_carries_only_summarised_factors() -> None:
    explanation = _explanation()
    explanation["transaction_id"] = 3_000_123

    document = json.loads(cast(str, build_input(explanation)))

    assert set(document) == {"keputusan", "jumlah_faktor", "faktor", "faktor_lain"}
    assert document["faktor"][0] == {
        "nama": "P_emaildomain",
        "label": "email domain P_emaildomain",
        "arah": "menaikkan risiko",
        "porsi": "67%",
    }
    assert "3000123" not in json.dumps(document)
    assert "raw_score" not in json.dumps(document)


def test_group_factor_lists_members_instead_of_value() -> None:
    explanation = _explanation()
    explanation["contributions"].append(
        {"name": "V-12", "kind": "group", "members": ["V1", "V2"], "contribution": 0.5}
    )

    group = json.loads(cast(str, build_input(explanation)))["faktor"][2]

    assert group["anggota"] == "V1, V2"
    assert "nilai" not in group


def test_input_ignores_non_finite_contributions() -> None:
    explanation = _explanation()
    explanation["contributions"].insert(
        0, {"name": "f0", "kind": "feature", "value": 1.0, "contribution": math.inf}
    )

    document = json.loads(cast(str, build_input(explanation)))

    assert document["jumlah_faktor"] == 2
    assert all(factor["nama"] != "f0" for factor in document["faktor"])


def test_completer_sends_the_expected_request() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(output_text="isi", status="completed")

    client = cast(OpenAI, SimpleNamespace(responses=SimpleNamespace(create=create)))

    completion = OpenAiCompleter(client, "m")("aturan", "data")

    assert completion == Completion(text="isi", complete=True)
    assert calls == [
        {
            "model": "m",
            "instructions": "aturan",
            "input": "data",
            "reasoning": {"effort": "low"},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        }
    ]


def test_completer_marks_incomplete_response() -> None:
    def create(**kwargs: Any) -> Any:
        return SimpleNamespace(output_text="isi", status="incomplete")

    client = cast(OpenAI, SimpleNamespace(responses=SimpleNamespace(create=create)))

    assert OpenAiCompleter(client, "m")("a", "b").complete is False


def test_values_are_shared_only_for_whitelisted_engineered_features() -> None:
    explanation = {
        "available": True,
        "other": {"n_units": 0, "contribution": 0.0},
        "contributions": [
            {"name": "count_last_1h", "kind": "feature", "value": 4.0, "contribution": 1.0},
            {"name": "amt_max_so_far", "kind": "feature", "value": 2300.5, "contribution": 0.9},
            {"name": "TransactionAmt", "kind": "feature", "value": 89.5, "contribution": 0.8},
            {"name": "card1", "kind": "feature", "value": 9500.0, "contribution": 0.7},
            {"name": "P_emaildomain", "kind": "feature", "value": "gmail.com", "contribution": 0.6},
            {"name": "C13", "kind": "feature", "value": 55.0, "contribution": 0.5},
        ],
    }

    document = json.loads(cast(str, build_input(explanation)))
    values = {factor["nama"]: factor.get("nilai") for factor in document["faktor"]}

    assert values == {
        "count_last_1h": "4",
        "amt_max_so_far": None,
        "TransactionAmt": None,
        "card1": None,
        "P_emaildomain": None,
        "C13": None,
    }
    sent = json.dumps(document)
    for secret in ("2.300", "2300", "89,50", "9.500", "9500", "gmail.com"):
        assert secret not in sent
