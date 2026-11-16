"""Regressions for numeric experiment validity, persistence, and provider safety."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import pytest
from conftest import StubNumericBackend, sample_numeric_tasks

from apart_epistemics.backends import (
    BudgetExceededError,
    GeminiBackend,
    OpenRouterBackend,
)
from apart_epistemics.cli import _execute_numeric_track
from apart_epistemics.engine import analyze_numeric_core, run_numeric_core
from apart_epistemics.models import ExperimentSuiteConfig
from apart_epistemics.run_state import configure_budget


def chat_response(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "provider": "test-provider",
        "choices": [{"message": {"content": "Final estimate: 250"}}],
    }


def numeric(path: Path, **overrides: Any) -> Any:
    args: dict[str, Any] = dict(
        tasks=sample_numeric_tasks(1),
        conditions=["C1", "C2"],
        n_agents=4,
        n_rounds=2,
        replicates=1,
        runs_path=path,
        backend=StubNumericBackend(seed=42),
        seed=42,
        alpha=0.3,
    )
    args.update(overrides)
    return run_numeric_core(**args)


@pytest.mark.parametrize(
    "change",
    [
        dict(seed=99),
        dict(alpha=1),
        dict(n_agents=5),
        dict(backend=StubNumericBackend(seed=99)),
    ],
)
def test_resume_rejects_changed_experiment_without_touching_data(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    path = tmp_path / "numeric.jsonl"
    numeric(path)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="manifest"):
        numeric(path, **change)
    assert path.read_bytes() == original


def test_partial_results_cannot_masquerade_as_final_or_zero_effect(tmp_path: Path) -> None:
    records = numeric(tmp_path / "numeric.jsonl")
    tasks = sample_numeric_tasks(1)
    assert analyze_numeric_core(records, tasks)["completion_status"] == "complete"
    for partial in [
        records[:-1],
        [r for r in records if r.condition_id == "C1" or r.round_idx == 1],
    ]:
        result = analyze_numeric_core(partial, tasks)
        assert result["completion_status"] == "incomplete_or_unverified"
        assert result["by_condition_final"] == {}
        assert result["contrasts"] == []
    with pytest.raises(RuntimeError, match="truth"):
        analyze_numeric_core(records, [tasks[0].model_copy(update={"numeric_truth": 2.0})])


def test_zero_budget_prevents_all_http_calls_and_reservations_persist(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(_u: str, _h: dict[str, str], p: dict[str, Any]) -> dict[str, Any]:
        calls.append(p)
        return chat_response(p["model"])

    cfg = ExperimentSuiteConfig(
        seed=42,
        num_agents=4,
        num_rounds=1,
        tasks_per_track=1,
        numeric_tasks=1,
        numeric_tasks_file="configs/polyphony_benchmark_tasks.json",
        numeric_replicates=1,
        numeric_conditions=["P0"],
        numeric_models=["lab/model-a"],
        conditions=[],
    )
    backend = OpenRouterBackend(model="agent", api_key="FAKE", transport=transport)
    with pytest.raises(BudgetExceededError):
        _execute_numeric_track(cfg, backend, tmp_path / "zero", n_agents=4, budget_usd=0.0)
    assert calls == []

    configure_budget(backend, tmp_path / "ledger", 2, 1)
    backend.for_model("other").generate_chat("s", "u")
    # Reconstructing the backend/ledger must not grant a fresh allowance.
    fresh = OpenRouterBackend(model="agent", api_key="FAKE", transport=transport)
    configure_budget(fresh, tmp_path / "ledger", 2, 1)
    fresh.generate_chat("s", "u")
    with pytest.raises(BudgetExceededError):
        fresh.generate_chat("s", "u")
    assert len(calls) == 2
    assert json.loads((tmp_path / "ledger/budget.json").read_text())["reserved_usd"] == 2
    assert len((tmp_path / "ledger/provider_requests.jsonl").read_text().splitlines()) == 2


def test_failed_requests_consume_allowance_and_tracebacks_are_redacted(tmp_path: Path) -> None:
    def fail(*_args: Any) -> Any:
        raise RuntimeError("url?key=AUDIT_FAKE_SECRET")

    backend = OpenRouterBackend(model="agent", api_key="AUDIT_FAKE_SECRET", transport=fail)
    configure_budget(backend, tmp_path, 1)
    try:
        backend.generate_chat("s", "u")
    except RuntimeError:
        assert "AUDIT_FAKE_SECRET" not in traceback.format_exc()
    else:
        pytest.fail("Expected transport failure")
    with pytest.raises(BudgetExceededError):
        backend.generate_chat("s", "u")


def test_gemini_uses_header_instead_of_secret_url() -> None:
    def transport(url: str, headers: dict[str, str], _payload: dict[str, Any]) -> dict[str, Any]:
        assert "SECRET" not in url
        assert headers["x-goog-api-key"] == "SECRET"
        return {}

    GeminiBackend(api_key="SECRET", transport=transport).generate_chat("s", "u")
