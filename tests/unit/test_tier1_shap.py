import math
from typing import Any

import pytest

from fraud.explainability.tier1_shap import _ensure_finite


def _document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "raw_score": 0.5,
        "base_value": -3.0,
        "unexplained": 0.01,
        "other": {"n_units": 3, "contribution": 0.2},
        "contributions": [{"name": "f0", "kind": "feature", "contribution": 1.0}],
    }
    document.update(overrides)
    return document


def test_finite_document_passes() -> None:
    _ensure_finite(_document())


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("field", ["raw_score", "base_value", "unexplained"])
def test_non_finite_top_level_number_is_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValueError):
        _ensure_finite(_document(**{field: bad}))


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_non_finite_contribution_is_rejected(bad: float) -> None:
    contributions = [{"name": "f0", "kind": "feature", "contribution": bad}]

    with pytest.raises(ValueError):
        _ensure_finite(_document(contributions=contributions))


def test_non_finite_other_contribution_is_rejected() -> None:
    with pytest.raises(ValueError):
        _ensure_finite(_document(other={"n_units": 3, "contribution": math.inf}))
