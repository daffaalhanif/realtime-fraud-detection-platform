from typing import Any, cast

import pytest
from langfuse import Langfuse

from fraud.explainability.narrative import Completion
from fraud.explainability.tracing import build_langfuse_client, shutdown_langfuse, traced


class _Generation:
    def __init__(self, fail_update: bool = False) -> None:
        self.updates: list[dict[str, Any]] = []
        self.ended = False
        self._fail_update = fail_update

    def update(self, **kwargs: Any) -> "_Generation":
        if self._fail_update:
            raise RuntimeError("langfuse update down")
        self.updates.append(kwargs)
        return self

    def end(self) -> None:
        self.ended = True


class _Client:
    def __init__(self, generation: _Generation | None = None, fail_start: bool = False) -> None:
        self.generation = generation or _Generation()
        self.started: list[dict[str, Any]] = []
        self.shutdown_called = False
        self._fail_start = fail_start

    def start_observation(self, **kwargs: Any) -> _Generation:
        if self._fail_start:
            raise RuntimeError("langfuse start down")
        self.started.append(kwargs)
        return self.generation

    def shutdown(self) -> None:
        self.shutdown_called = True


def _langfuse(client: _Client) -> Langfuse:
    return cast(Langfuse, client)


def _complete(instructions: str, input_text: str) -> Completion:
    return Completion(text="narasi", complete=True, input_tokens=11, output_tokens=5)


def test_without_client_the_completer_is_returned_unchanged() -> None:
    assert traced(_complete, "m", None) is _complete


def test_generation_records_exactly_what_was_sent_and_returned() -> None:
    client = _Client()

    completion = traced(_complete, "m", _langfuse(client))("aturan", "data")

    assert completion.text == "narasi"
    assert client.started == [
        {
            "name": "review-narrative",
            "as_type": "generation",
            "model": "m",
            "input": {"instructions": "aturan", "input": "data"},
        }
    ]
    assert client.generation.updates == [
        {
            "output": "narasi",
            "usage_details": {"input_tokens": 11, "output_tokens": 5},
            "metadata": {"complete": True},
        }
    ]
    assert client.generation.ended


def test_missing_token_counts_are_left_out_of_usage() -> None:
    client = _Client()

    def complete(instructions: str, input_text: str) -> Completion:
        return Completion(text="narasi", complete=True)

    traced(complete, "m", _langfuse(client))("a", "b")

    assert client.generation.updates[0]["usage_details"] == {}


def test_llm_error_is_recorded_and_still_raised() -> None:
    client = _Client()

    def failing(instructions: str, input_text: str) -> Completion:
        raise TimeoutError("lambat")

    with pytest.raises(TimeoutError):
        traced(failing, "m", _langfuse(client))("a", "b")

    assert client.generation.updates == [{"level": "ERROR", "status_message": "lambat"}]
    assert client.generation.ended


def test_langfuse_failure_at_start_does_not_break_the_call() -> None:
    client = _Client(fail_start=True)

    assert traced(_complete, "m", _langfuse(client))("a", "b").text == "narasi"


def test_langfuse_failure_at_update_does_not_break_the_call() -> None:
    client = _Client(_Generation(fail_update=True))

    assert traced(_complete, "m", _langfuse(client))("a", "b").text == "narasi"


def test_client_is_not_built_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert build_langfuse_client() is None


def test_client_is_not_built_with_only_one_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert build_langfuse_client() is None


def test_shutdown_tolerates_missing_client_and_failures() -> None:
    shutdown_langfuse(None)

    class Broken:
        def shutdown(self) -> None:
            raise RuntimeError("down")

    shutdown_langfuse(cast(Langfuse, Broken()))


def test_shutdown_flushes_the_client() -> None:
    client = _Client()

    shutdown_langfuse(_langfuse(client))

    assert client.shutdown_called
