import math
from typing import Any

import pytest

from fraud.explainability.template import UNAVAILABLE_NARRATIVE, build_template_narrative


def _explanation(*contributions: float) -> dict[str, Any]:
    return {
        "available": True,
        "contributions": [
            {"name": f"f{index}", "kind": "feature", "value": 1.0, "contribution": value}
            for index, value in enumerate(contributions)
        ],
    }


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_contribution_is_ignored(bad: float) -> None:
    narrative = build_template_narrative(_explanation(bad, 1.0, -0.5))

    assert "kolom f0" not in narrative
    assert "kolom f1 (nilai 1; 67%)" in narrative
    assert "kolom f2 (nilai 1; 33%)" in narrative
    assert "di antara 2 faktor terbesar" in narrative


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_only_non_finite_contributions_is_unavailable(bad: float) -> None:
    assert build_template_narrative(_explanation(bad)) == UNAVAILABLE_NARRATIVE


def test_unavailable_explanation_gives_unavailable_narrative() -> None:
    assert build_template_narrative({"available": False}) == UNAVAILABLE_NARRATIVE


def test_all_zero_contributions_is_unavailable() -> None:
    assert build_template_narrative(_explanation(0.0, 0.0)) == UNAVAILABLE_NARRATIVE


def test_closing_sentence_excludes_smaller_factors() -> None:
    narrative = build_template_narrative(_explanation(1.0, 0.5, -0.5))

    assert narrative.endswith(
        "Persentase adalah porsi pengaruh di antara 3 faktor terbesar, "
        "tidak termasuk faktor lain yang lebih kecil."
    )
