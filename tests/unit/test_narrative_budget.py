from typing import Any

from fraud.consumers.narrative_consumer import narrate_within_budget
from fraud.explainability.narrative import NarrativeResult


class _Clock:
    """Jam tiruan yang maju hanya saat dipanggil `advance`."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _items(count: int) -> list[tuple[int, dict[str, Any]]]:
    return [(index, {"available": True}) for index in range(count)]


def test_all_items_are_written_within_budget() -> None:
    clock = _Clock()
    stored: list[int] = []

    def narrate(_: dict[str, Any]) -> NarrativeResult | None:
        clock.now += 10
        return NarrativeResult("teks", "m", 1)

    counts = narrate_within_budget(
        _items(3), narrate, lambda tid, _: stored.append(tid), 100, clock
    )

    assert counts == (3, 0, 0)
    assert stored == [0, 1, 2]


def test_items_after_the_budget_are_skipped_and_not_stored() -> None:
    clock = _Clock()
    stored: list[int] = []

    def narrate(_: dict[str, Any]) -> NarrativeResult | None:
        clock.now += 60
        return NarrativeResult("teks", "m", 1)

    counts = narrate_within_budget(
        _items(5), narrate, lambda tid, _: stored.append(tid), 100, clock
    )

    assert counts == (2, 0, 3)
    assert stored == [0, 1]


def test_failed_items_are_counted_and_not_stored() -> None:
    stored: list[int] = []

    def narrate(explanation: dict[str, Any]) -> NarrativeResult | None:
        return None

    counts = narrate_within_budget(
        _items(2), narrate, lambda tid, _: stored.append(tid), 100, _Clock()
    )

    assert counts == (0, 2, 0)
    assert stored == []


def test_nothing_to_narrate() -> None:
    assert narrate_within_budget([], lambda _: None, lambda *_: None, 100, _Clock()) == (0, 0, 0)
