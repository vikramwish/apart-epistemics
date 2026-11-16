"""Tests for domain models, dual-track datasets, and adversarial numeric/JSON parsers."""

from importlib import resources

import pytest
from pydantic import ValidationError

import apart_epistemics
from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.metrics import clamp, quantile_sorted
from apart_epistemics.models import (
    ConditionConfig,
    HeterogeneityType,
    NationalityType,
    NumericEstimateRecord,
)
from apart_epistemics.parsers import parse_numeric_estimate


def test_condition_config_validates_bounds() -> None:
    cfg = ConditionConfig(
        condition_id="us_k1",
        label="Monolithic US",
        heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
        nationality_type=NationalityType.US_ONLY,
        num_models=1,
        temperature=0.8,
        top_p=0.95,
    )
    assert cfg.num_models == 1
    assert cfg.temperature == pytest.approx(0.8)

    with pytest.raises(ValidationError):
        ConditionConfig(
            condition_id="bad_temp",
            label="Bad",
            heterogeneity_type=HeterogeneityType.MONOLITHIC_SHARED,
            nationality_type=NationalityType.US_ONLY,
            num_models=1,
            temperature=3.5,
        )


def test_datasets_provide_reproducible_agent_population() -> None:
    agents = generate_agent_population(num_agents=30, seed=42)
    assert len(agents) == 30
    regions = {a.region for a in agents}
    assert len(regions) >= 5  # Stratified global regions, not monocultural
    assert all(0.05 <= a.prior_belief <= 0.95 for a in agents)
    # Reproducibility across calls with same seed
    agents_repeat = generate_agent_population(num_agents=30, seed=42)
    assert [a.model_dump() for a in agents] == [a.model_dump() for a in agents_repeat]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final estimate: 1,200,000", 1_200_000.0),
        (
            "Based on scaling laws, I suggest roughly 500k.\nFinal estimate: 1.2 million",
            1_200_000.0,
        ),
        ("Final estimate: 4.5 billion", 4_500_000_000.0),
        ("Final estimate: ~350 thousand", 350_000.0),
        ("Final estimate: 3.2e5", 320_000.0),
        ("Final estimate: 10,935 m", 10_935.0),
        ("Final estimate: 1522 m/s", 1_522.0),
        ("Final estimate: 5.4 \u00d7 10^4", 54_000.0),
        ("Final estimate: 5.4 x 10^4", 54_000.0),
        ("Final estimate: 10,900 m, as measured in 2019.", 10_900.0),
        ("My reasoning covers 1990 to 2020. Final estimate: 42,500", 42_500.0),
    ],
)
def test_numeric_parser_handles_units_commas_and_distractors(text: str, expected: float) -> None:
    result = parse_numeric_estimate(text)
    assert result.miss is False
    assert result.value == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "bad_text",
    [
        "",
        "I cannot provide an exact number without more data.",
        "Final estimate: 0",  # non-positive cannot enter log10
        "Final estimate: -150",  # negative cannot enter log10
        "Final estimate: N/A",
        "Final estimate: 1e9999",  # float overflow must become a clean miss
        "Final estimate: 2,840-2,900 km",  # range, not a single estimate
        "Final estimate: 10 to 20",  # range written in words
        "Final estimate: $12.5k",  # ambiguous single-letter multiplier
        "The depth is roughly 10,900 m, as measured in 2019.",  # no prefix: never use last number
        "Estimate without prefix is 87500.",
    ],
)
def test_numeric_parser_marks_misses_safely_without_crashing(bad_text: str) -> None:
    result = parse_numeric_estimate(bad_text, require_positive=True)
    assert result.miss is True
    assert result.value is None


def test_math_helpers_and_public_package_surface() -> None:
    assert clamp(float("nan"), 0.0, 1.0) == 0.0
    assert clamp(float("inf"), 0.0, 1.0) == 0.0
    assert clamp(1.5, 0.0, 1.0) == 1.0
    assert quantile_sorted([], 0.5) == 0.0
    assert quantile_sorted([4.2], 0.75) == pytest.approx(4.2)
    assert quantile_sorted([0.0, 10.0], 0.25) == pytest.approx(2.5)
    for symbol in apart_epistemics.__all__:
        assert hasattr(apart_epistemics, symbol)
    assert (resources.files("apart_epistemics") / "py.typed").is_file()


def test_numeric_estimate_record_log10_consistency() -> None:
    rec = NumericEstimateRecord(
        condition_id="C1",
        question_id="q1",
        replicate=0,
        round_idx=1,
        agent_id="a1",
        model_slug="openai/gpt-6-luna",
        provider="openai",
        raw_reply="Final estimate: 1000",
        parsed_value=1000.0,
        assistant_log10=3.0,
        agent_log10_before=2.0,
        agent_log10_after=2.3,
        miss=False,
        reasoning_tokens=12,
        temperature=0.8,
        top_p=0.95,
    )
    assert rec.agent_log10_after == pytest.approx(0.7 * 2.0 + 0.3 * 3.0)
