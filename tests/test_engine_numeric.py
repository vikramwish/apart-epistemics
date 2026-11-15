"""Integration tests for Kevin's crash-resumable C1/C2/C3 numeric engine & Page analysis."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.backends import BudgetExceededError, SyntheticEpistemicBackend
from apart_epistemics.datasets import build_numeric_core_tasks
from apart_epistemics.engine import (
    TokenPricing,
    analyze_numeric_core,
    build_numeric_prompts,
    format_prompt_number,
    run_numeric_core,
)
from apart_epistemics.engine.numeric_core import _estimate_call_cost_usd
from apart_epistemics.models import AgentProfile, TaskItem


def test_c1_c2_c3_share_one_system_prompt_and_c3_only_appends_profile(
    sample_agent: AgentProfile,
) -> None:
    def system_for(cond: str) -> str:
        return build_numeric_prompts(
            condition_id=cond,
            agent=sample_agent,
            question="Length of river in km?",
            current_x=1500.0,
            pop_stats=(2100.0, 1400.0, 3100.0),
            history=[],
        )[0]

    sys_c1, sys_c2, sys_c3 = system_for("C1"), system_for("C2"), system_for("C3")
    assert sys_c1 == sys_c2
    # C3 keeps the shared prompt verbatim as a prefix and only adds the profile sentence.
    assert sys_c3.startswith(sys_c1)
    added = sys_c3[len(sys_c1) :]
    assert sample_agent.profession in added
    assert sample_agent.region in added
    assert sample_agent.age_band in added
    # No role/persona change: C3 must not swap in a different "advisor" framing.
    assert "advisor" not in sys_c3.lower()


def test_numeric_prompt_asks_for_best_estimate_in_task_unit(sample_agent: AgentProfile) -> None:
    _, usr = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="Maximum depth of Lake Baikal?",
        current_x=1500.0,
        pop_stats=None,
        history=[],
        unit="meters",
    )
    assert "My current estimate is 1,500 meters." in usr
    assert "What is your best estimate of the true value?" in usr
    assert "Final estimate: <plain number in meters>" in usr
    assert "nothing after the number" in usr
    assert "updated" not in usr

    _, usr_no_unit = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="q",
        current_x=2.0,
        pop_stats=None,
        history=[],
    )
    assert "Final estimate: <plain number>" in usr_no_unit


def test_runner_passes_task_unit_into_prompt(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    seen: list[str] = []

    class Spy(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["user_prompt"])
            return super().generate_numeric_reply(**kwargs)

    run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "runs.jsonl",
        backend=Spy(seed=1),
    )
    assert seen
    assert all(f"Final estimate: <plain number in {task.unit}>" in p for p in seen)


def test_cost_estimate_does_not_double_count_reasoning_tokens() -> None:
    # completion_tokens (output_tokens) already includes the 400 reasoning tokens.
    resp = {"input_tokens": 1000, "output_tokens": 500, "reasoning_tokens": 400}
    cost = _estimate_call_cost_usd(resp, TokenPricing(input_per_m=1.0, output_per_m=2.0))
    assert cost == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1_000_000)
    # Reasoning tokens alone change nothing.
    no_reasoning = {**resp, "reasoning_tokens": 0}
    assert _estimate_call_cost_usd(no_reasoning) == pytest.approx(_estimate_call_cost_usd(resp))


def test_prices_are_configurable_and_drive_the_budget_guard(tmp_path: Path) -> None:
    resp = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert _estimate_call_cost_usd(resp, TokenPricing(0.1, 0.5)) == pytest.approx(0.1)
    assert _estimate_call_cost_usd(resp, TokenPricing(4.0, 0.5)) == pytest.approx(4.0)

    task = build_numeric_core_tasks()[0]

    class Costly(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            out = super().generate_numeric_reply(**kwargs)
            out["input_tokens"], out["output_tokens"] = 1_000_000, 0
            return out

    def run(pricing: TokenPricing, name: str) -> None:
        run_numeric_core(
            tasks=[task],
            conditions=["C1"],
            n_agents=2,
            n_rounds=1,
            replicates=1,
            runs_path=tmp_path / name,
            backend=Costly(seed=1),
            budget_usd=1.0,
            pricing=pricing,
        )

    run(TokenPricing(0.1, 0.5), "cheap.jsonl")  # 2 calls x $0.10 stays under the $1 cap
    with pytest.raises(BudgetExceededError):
        run(TokenPricing(4.0, 0.5), "dear.jsonl")  # first call alone exceeds the cap


def test_runner_records_finish_reason_and_forwards_max_tokens(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    seen_max_tokens: list[int] = []

    class Backend(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen_max_tokens.append(kwargs["max_tokens"])
            out = super().generate_numeric_reply(**kwargs)
            out["finish_reason"] = "length"
            return out

    default_recs = run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "a.jsonl",
        backend=Backend(seed=1),
    )
    assert set(seen_max_tokens) == {1024}
    assert all(r.finish_reason == "length" for r in default_recs)
    # Sampling was never sent, so it must not be logged as if it had been.
    assert all(r.temperature is None and r.top_p is None for r in default_recs)

    seen_max_tokens.clear()
    run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "b.jsonl",
        backend=Backend(seed=1),
        max_tokens=4096,
    )
    assert set(seen_max_tokens) == {4096}


def test_changing_max_tokens_requires_a_new_output_folder(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    kwargs: dict[str, Any] = {
        "tasks": [task],
        "conditions": ["C1"],
        "n_agents": 2,
        "n_rounds": 1,
        "replicates": 1,
        "runs_path": tmp_path / "runs.jsonl",
        "backend": SyntheticEpistemicBackend(seed=1),
    }
    run_numeric_core(**kwargs, max_tokens=1024)
    run_numeric_core(**kwargs, max_tokens=1024)  # same settings resume fine
    with pytest.raises(RuntimeError, match="new output directory"):
        run_numeric_core(**kwargs, max_tokens=2048)


def _priced_backend(cost: float | None, per_attempt: list[float | None] | None = None) -> Any:
    class Priced(SyntheticEpistemicBackend):
        calls = 0

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            out = super().generate_numeric_reply(**kwargs)
            out["input_tokens"], out["output_tokens"] = 10, 10
            seq = per_attempt if per_attempt is not None else [cost]
            out["cost_usd"] = seq[min(Priced.calls, len(seq) - 1)]
            Priced.calls += 1
            return out

    return Priced(seed=1)


def test_records_store_the_providers_real_cost(tmp_path: Path) -> None:
    recs = _run_priced(tmp_path, "a.jsonl", _priced_backend(0.0004))
    assert [r.cost_usd for r in recs] == [0.0004] * len(recs)
    assert recs[0].model_dump()["cost_usd"] == 0.0004
    # Backends that report no cost leave it unknown rather than guessing.
    unknown = _run_priced(tmp_path, "b.jsonl", _priced_backend(None))
    assert all(r.cost_usd is None for r in unknown)


def _run_priced(tmp_path: Path, name: str, backend: Any, **kwargs: Any) -> Any:
    params: dict[str, Any] = {
        "tasks": build_numeric_core_tasks()[:1],
        "conditions": ["C1"],
        "n_agents": 3,
        "n_rounds": 1,
        "replicates": 1,
        "runs_path": tmp_path / name,
        "backend": backend,
    }
    params.update(kwargs)
    return run_numeric_core(**params)


def test_budget_guard_uses_real_cost_not_the_token_estimate(tmp_path: Path) -> None:
    # Tokens are tiny, so the token estimate is ~1e-5 per call and would never trip $1.
    with pytest.raises(BudgetExceededError):
        _run_priced(tmp_path, "dear.jsonl", _priced_backend(0.6), budget_usd=1.0)
    # A genuinely cheap real cost keeps the same run well inside the cap.
    ok = _run_priced(tmp_path, "cheap.jsonl", _priced_backend(0.0001), budget_usd=1.0)
    assert len(ok) == 3


def test_estimate_prefers_real_cost_and_falls_back_to_tokens() -> None:
    resp = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert _estimate_call_cost_usd({**resp, "cost_usd": 0.02}, TokenPricing(4.0, 1.0)) == 0.02
    assert _estimate_call_cost_usd({**resp, "cost_usd": 0.0}, TokenPricing(4.0, 1.0)) == 0.0
    assert _estimate_call_cost_usd({**resp, "cost_usd": None}, TokenPricing(4.0, 1.0)) == 4.0
    assert _estimate_call_cost_usd(resp, TokenPricing(4.0, 1.0)) == 4.0


def test_row_cost_covers_the_format_retry_too(tmp_path: Path) -> None:
    class RetryOnce(SyntheticEpistemicBackend):
        calls = 0

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            RetryOnce.calls += 1
            reply = "no number here" if RetryOnce.calls == 1 else "Final estimate: 3000"
            return {"raw_reply": reply, "cost_usd": 0.001 * RetryOnce.calls, "provider": "p"}

    recs = _run_priced(tmp_path, "r.jsonl", RetryOnce(seed=1), n_agents=1)
    assert recs[0].cost_usd == pytest.approx(0.001 + 0.002)  # both attempts, not just the last

    # If any attempt reports no cost, the row's cost is unknown rather than understated.
    class PartlyUnknown(RetryOnce):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            out = super().generate_numeric_reply(**kwargs)
            if RetryOnce.calls == 1:
                out["cost_usd"] = None
            return out

    RetryOnce.calls = 0
    recs = _run_priced(tmp_path, "u.jsonl", PartlyUnknown(seed=1), n_agents=1)
    assert recs[0].cost_usd is None


def test_resume_counts_saved_real_costs_against_the_budget(tmp_path: Path) -> None:
    _run_priced(tmp_path, "run.jsonl", _priced_backend(0.4), n_agents=2, budget_usd=5.0)
    # The saved rows already cost $0.80; with a $0.50 cap the resumed run must refuse to start
    # new calls (nothing is pending here, so it just returns the saved rows safely).
    again = _run_priced(tmp_path, "run.jsonl", _priced_backend(0.4), n_agents=2, budget_usd=0.5)
    assert [r.cost_usd for r in again] == [0.4, 0.4]


def test_runner_treats_values_over_three_log_units_away_as_misses(tmp_path: Path) -> None:
    task = build_numeric_core_tasks()[0]
    prompts: list[str] = []

    class UnitSlipBackend(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            prompts.append(kwargs["user_prompt"])
            # A metres-read-as-millions style slip: roughly 1e9 times the true size.
            return {
                "raw_reply": f"Final estimate: {task.numeric_truth * 1e9:g}",
                "model_slug": "m",
                "provider": "p",
                "reasoning_tokens": 0,
            }

    records = run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=2,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "runs.jsonl",
        backend=UnitSlipBackend(seed=1),
    )
    assert len(records) == 2
    assert all(r.miss for r in records)
    assert all(r.parsed_value is None and r.assistant_log10 is None for r in records)
    # Each implausible reply is retried once before being recorded as a miss.
    assert len(prompts) == 4
    # A miss leaves the agent's belief unchanged.
    assert all(r.agent_log10_after == r.agent_log10_before for r in records)


def test_numeric_prompt_builder_and_crash_resumable_runner(
    tmp_path: Path, sample_agent: AgentProfile
) -> None:
    _sys_c1, usr_c1 = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=None,
        history=[],
    )
    assert "Other people" not in usr_c1
    assert "civil engineer" not in _sys_c1

    sys_c2, usr_c2 = build_numeric_prompts(
        condition_id="C2",
        agent=sample_agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=(2100.0, 1400.0, 3100.0),
        history=[],
    )
    assert "Other people asking this recently gave a median of" in usr_c2
    assert "civil engineer" not in sys_c2

    sys_c3, usr_c3 = build_numeric_prompts(
        condition_id="C3",
        agent=sample_agent,
        question="Length of river in km?",
        current_x=1500.0,
        pop_stats=(2100.0, 1400.0, 3100.0),
        history=[("My current estimate is 1200.", "Final estimate: 1800")],
    )
    assert "civil engineer" in sys_c3
    assert "south_asia" in sys_c3
    assert "1500" not in sys_c3
    assert "1800" in usr_c3

    tasks = build_numeric_core_tasks()[:2]
    runs_file = tmp_path / "numeric_runs.jsonl"

    class FlakyBackend(SyntheticEpistemicBackend):
        def __init__(self) -> None:
            super().__init__(seed=7)
            self.calls = 0

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "raw_reply": "Unclear without data.",
                    "model_slug": "m",
                    "provider": "p",
                    "reasoning_tokens": 0,
                }
            if self.calls in (2, 3):
                return {
                    "raw_reply": "Still unclear.",
                    "model_slug": "m",
                    "provider": "p",
                    "reasoning_tokens": 0,
                }
            return super().generate_numeric_reply(**kwargs)

    flaky = FlakyBackend()
    records = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=8,
        n_rounds=3,
        replicates=1,
        runs_path=runs_file,
        backend=flaky,
        seed=7,
    )
    assert len(records) == 2 * 3 * 8 * 3
    assert any(r.miss for r in records)
    calls_first_run = flaky.calls

    # Resuming on the same runs_path must skip every completed cell (0 extra backend calls).
    records_resumed = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=8,
        n_rounds=3,
        replicates=1,
        runs_path=runs_file,
        backend=flaky,
        seed=7,
    )
    assert len(records_resumed) == len(records)
    assert flaky.calls == calls_first_run

    analysis = analyze_numeric_core(records_resumed, tasks)
    assert analysis["max_identity_residual"] < 1e-11
    assert len(analysis["contrasts"]) == 4

    with pytest.raises(BudgetExceededError):
        run_numeric_core(
            tasks=tasks,
            conditions=["C1"],
            n_agents=8,
            n_rounds=2,
            replicates=1,
            runs_path=tmp_path / "budget_runs.jsonl",
            backend=SyntheticEpistemicBackend(seed=7),
            budget_usd=0.0,
            seed=7,
        )


def test_cross_process_determinism_and_truncated_jsonl_resume(tmp_path: Path) -> None:
    outs = []
    for hseed in ("111", "999"):
        sub_out = tmp_path / f"det_{hseed}.jsonl"
        code = (
            "import json; from pathlib import Path; "
            "from apart_epistemics import ("
            "SyntheticEpistemicBackend, build_numeric_core_tasks, run_numeric_core"
            "); "
            f"recs = run_numeric_core(build_numeric_core_tasks()[:1], ['C3'], 4, 2, 1, "
            f"Path({str(sub_out)!r}), SyntheticEpistemicBackend(seed=42), seed=42); "
            "print(json.dumps([round(r.agent_log10_after, 6) for r in recs]))"
        )
        env = dict(os.environ, PYTHONHASHSEED=hseed)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1]

    trunc_path = tmp_path / "trunc.jsonl"
    tasks = build_numeric_core_tasks()[:1]
    backend = SyntheticEpistemicBackend(seed=42)
    full = run_numeric_core(tasks, ["C3"], 4, 2, 1, trunc_path, backend, seed=42)
    lines = trunc_path.read_text(encoding="utf-8").splitlines()
    trunc_path.write_text(
        "\n".join([*lines[:5], '{"condition_id": "C3", "broken']), encoding="utf-8"
    )
    repaired = run_numeric_core(tasks, ["C3"], 4, 2, 1, trunc_path, backend, seed=42)
    assert len(repaired) == len(full)
    assert [round(r.agent_log10_after, 6) for r in repaired] == [
        round(r.agent_log10_after, 6) for r in full
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (104.317, "104"),
        (2.51804e08, "252,000,000"),
        (52.3678, "52"),
        (4.8, "5"),
        (0.2, "1"),
    ],
)
def test_format_prompt_number_rounds_to_three_sig_figs_whole_commas(
    value: float, expected: str
) -> None:
    assert format_prompt_number(value) == expected


def test_prompts_round_estimates_crowd_and_c3_history_while_keeping_exact_state(
    tmp_path: Path, sample_agent: AgentProfile
) -> None:
    _, usr_medals = build_numeric_prompts(
        condition_id="C2",
        agent=sample_agent,
        question="Total medals?",
        current_x=104.317,
        pop_stats=(110.172, 52.3678, 249.614),
        history=[],
        unit="medals",
    )
    assert "My current estimate is 104 medals." in usr_medals
    assert "median of 110 (IQR 52 to 250)." in usr_medals

    _, usr_box = build_numeric_prompts(
        condition_id="C1",
        agent=sample_agent,
        question="Cumulative gross?",
        current_x=2.51804e08,
        pop_stats=None,
        history=[],
        unit="US dollars",
    )
    assert "My current estimate is 252,000,000 US dollars." in usr_box
    assert "e+" not in usr_box

    task = TaskItem(
        id="box_test",
        kind="numeric_estimate",
        domain="box",
        question="Cumulative gross?",
        numeric_truth=395_573_235.0,
        unit="US dollars",
        source_note="The Numbers daily chart, 2 Aug 2026: $395,573,235",
    )
    seen_prompts: list[str] = []

    class SpyBackend(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen_prompts.append(kwargs["user_prompt"])
            return {
                "raw_reply": "Final estimate: 260000000",
                "model_slug": "m",
                "provider": "p",
                "reasoning_tokens": 0,
            }

    runs_path = tmp_path / "rounded_c3.jsonl"
    recs = run_numeric_core(
        tasks=[task],
        conditions=["C3"],
        n_agents=2,
        n_rounds=2,
        replicates=1,
        runs_path=runs_path,
        backend=SpyBackend(seed=1),
        alpha=0.3,
        seed=42,
    )
    assert len(seen_prompts) == 4
    r2_prompt = seen_prompts[2]
    assert "Recent conversation history:\nTurn 1 User: My current estimate is " in r2_prompt
    assert " US dollars. | Assistant: Final estimate: 260000000" in r2_prompt
    assert "e+" not in r2_prompt
    # Internal state remains exact unrounded log10 values.
    r1 = recs[0]
    expected_after = 0.7 * r1.agent_log10_before + 0.3 * math.log10(260_000_000.0)
    assert r1.agent_log10_after == pytest.approx(expected_after, abs=1e-12)
    assert abs(10**r1.agent_log10_before - round(float(f"{10**r1.agent_log10_before:.3g}"))) > 1.0


def test_tightened_sanity_check_catches_thousand_and_million_unit_slips(tmp_path: Path) -> None:
    task = TaskItem(
        id="box_unit",
        kind="numeric_estimate",
        domain="box",
        question="Cumulative gross?",
        numeric_truth=186_004_506.0,
        unit="US dollars",
        source_note="The Numbers daily chart, 2 Aug 2026: $186,004,506",
    )

    class PlannedBackend(SyntheticEpistemicBackend):
        def __init__(self, replies: list[str]) -> None:
            super().__init__(seed=1)
            self.replies = replies
            self.calls = 0

        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            reply = self.replies[min(self.calls, len(self.replies) - 1)]
            self.calls += 1
            return {"raw_reply": reply, "model_slug": "m", "provider": "p", "reasoning_tokens": 0}

    # 235,000 vs ~186M (distance ~2.9) and 350 vs ~186M (distance ~5.7) are rejected as misses.
    slip_backend = PlannedBackend(["Final estimate: 235000", "Final estimate: 350"])
    slip_recs = run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=1,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "slip.jsonl",
        backend=slip_backend,
        start_sd=0.0,
        start_bias_weight=0.0,
    )
    assert slip_backend.calls == 2
    assert slip_recs[0].miss is True

    # A 10x estimate (distance = 1.0 <= 2.0) is accepted on the first try.
    ok_backend = PlannedBackend(["Final estimate: 1860045060"])
    ok_recs = run_numeric_core(
        tasks=[task],
        conditions=["C1"],
        n_agents=1,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "ok.jsonl",
        backend=ok_backend,
        start_sd=0.0,
        start_bias_weight=0.0,
    )
    assert ok_backend.calls == 1
    assert ok_recs[0].miss is False
    assert ok_recs[0].parsed_value == pytest.approx(1_860_045_060.0)


def test_run_numeric_core_parallel_trajectories_match_sequential(tmp_path: Path) -> None:
    tasks = build_numeric_core_tasks()[:3]
    seq = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=6,
        n_rounds=3,
        replicates=2,
        runs_path=tmp_path / "seq.jsonl",
        backend=SyntheticEpistemicBackend(seed=11),
        seed=11,
        workers=1,
    )
    par_path = tmp_path / "par.jsonl"
    par = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=6,
        n_rounds=3,
        replicates=2,
        runs_path=par_path,
        backend=SyntheticEpistemicBackend(seed=11),
        seed=11,
        workers=8,
    )
    assert len(par) == len(seq) == 3 * 3 * 2 * 3 * 6
    assert [r.model_dump() for r in par] == [r.model_dump() for r in seq]

    # Parallel resume with all records present returns identical records without new calls.
    resumed = run_numeric_core(
        tasks=tasks,
        conditions=["C1", "C2", "C3"],
        n_agents=6,
        n_rounds=3,
        replicates=2,
        runs_path=par_path,
        backend=SyntheticEpistemicBackend(seed=11),
        seed=11,
        workers=8,
    )
    assert [r.model_dump() for r in resumed] == [r.model_dump() for r in seq]


def test_hierarchical_page_decomposition_identities_and_edge_cases() -> None:
    from apart_epistemics.metrics import compute_hierarchical_page_decomposition

    # Closed-form balanced case: cluster A=[1,3] (mean 2), cluster B=[5,7] (mean 6), truth=0
    out = compute_hierarchical_page_decomposition([1.0, 3.0, 5.0, 7.0], ["A", "A", "B", "B"], 0.0)
    assert out["crowd_error"] == pytest.approx(16.0)
    assert out["avg_individual_error"] == pytest.approx(21.0)
    assert out["diversity"] == pytest.approx(5.0)
    assert out["between_model_diversity"] == pytest.approx(4.0)
    assert out["within_model_diversity"] == pytest.approx(1.0)
    assert out["excess_between_model_diversity"] == pytest.approx(4.0 - (1 / 2) * 1.0)
    assert out["between_share"] == pytest.approx(0.8)
    assert out["num_models"] == 2.0
    assert out["identity_residual"] < 1e-15
    assert out["hierarchical_identity_residual"] < 1e-15

    # Unbalanced K=3, N=7 clusters still preserve exact ANOVA decomposition.
    unbal = compute_hierarchical_page_decomposition(
        [0.1, -0.3, 0.4, 0.8, 1.0, -0.5, -0.2],
        ["m1", "m1", "m2", "m2", "m2", "m3", "m3"],
        0.2,
    )
    assert unbal["identity_residual"] < 1e-15
    assert unbal["hierarchical_identity_residual"] < 1e-15
    assert unbal["between_model_diversity"] + unbal["within_model_diversity"] == pytest.approx(
        unbal["diversity"], abs=1e-15
    )

    # K=1 monoculture and K=N singletons set excess_between_model_diversity to 0.0.
    mono = compute_hierarchical_page_decomposition([1.0, 2.0, 4.0], ["m1", "m1", "m1"], 0.0)
    assert mono["between_model_diversity"] == pytest.approx(0.0, abs=1e-15)
    assert mono["within_model_diversity"] == pytest.approx(mono["diversity"], abs=1e-15)
    assert mono["excess_between_model_diversity"] == 0.0
    assert mono["between_share"] == 0.0

    singleton = compute_hierarchical_page_decomposition([1.0, 2.0, 4.0], ["a", "b", "c"], 0.0)
    assert singleton["within_model_diversity"] == pytest.approx(0.0, abs=1e-15)
    assert singleton["excess_between_model_diversity"] == 0.0

    zero_div = compute_hierarchical_page_decomposition([2.0, 2.0], ["a", "b"], 2.0)
    assert zero_div["between_share"] == 0.0
    assert zero_div["hierarchical_identity_residual"] < 1e-15

    empty = compute_hierarchical_page_decomposition([], [], 0.0)
    assert all(val == 0.0 for val in empty.values())

    with pytest.raises(ValueError, match="identical length"):
        compute_hierarchical_page_decomposition([1.0], ["a", "b"], 0.0)


def test_c1_through_c6_prompt_symmetry_and_social_routing(
    tmp_path: Path, sample_agent: AgentProfile
) -> None:
    def prompts(cond: str) -> tuple[str, str]:
        return build_numeric_prompts(
            condition_id=cond,
            agent=sample_agent,
            question="How many medals?",
            current_x=104.3,
            pop_stats=(110.2, 52.4, 249.6),
            history=[("Turn 1 u", "Turn 1 a")],
            unit="medals",
            current_date="2026-11-14",
        )

    sys_c1, usr_c1 = prompts("C1")
    sys_c2, usr_c2 = prompts("C2")
    sys_c3, usr_c3 = prompts("C3")
    sys_c4, usr_c4 = prompts("C4")
    sys_c5, usr_c5 = prompts("C5")
    sys_c6, usr_c6 = prompts("C6")

    assert sys_c1 == sys_c2 == sys_c4 == sys_c5 == sys_c6
    assert sys_c3.startswith(sys_c1)
    assert usr_c5 == usr_c2
    assert usr_c6 == usr_c4
    assert "Other people asking this recently gave a median of" not in usr_c1
    assert "dialectical counter-anchoring" not in usr_c2.lower()
    assert "dialectical counter-anchoring" not in usr_c3.lower()
    assert "dialectical counter-anchoring" in usr_c4
    assert usr_c4.index("median of") < usr_c4.index("dialectical counter-anchoring")
    assert usr_c4.index("dialectical counter-anchoring") < usr_c4.index("My current estimate")

    # Runner passes pop_stats into prompts for C2, C3, C4, C5, and C6 (never C1).
    seen_by_cond: dict[str, list[str]] = {}

    class PromptSpy(SyntheticEpistemicBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen_by_cond.setdefault(kwargs["condition_id"], []).append(kwargs["user_prompt"])
            return super().generate_numeric_reply(**kwargs)

    run_numeric_core(
        tasks=build_numeric_core_tasks()[:1],
        conditions=["C1", "C2", "C3", "C4", "C5", "C6"],
        n_agents=4,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "prompts.jsonl",
        backend=PromptSpy(seed=3),
        models=["m1", "m2"],
    )
    assert all("median of" not in p for p in seen_by_cond["C1"])
    for cond in ("C2", "C3", "C4", "C5", "C6"):
        assert all("median of" in p for p in seen_by_cond[cond])
    assert all("dialectical counter-anchoring" in p for p in seen_by_cond["C4"])
    assert all("dialectical counter-anchoring" in p for p in seen_by_cond["C6"])


def test_polyphony_validation_routing_manifest_and_backward_compatibility(tmp_path: Path) -> None:
    from apart_epistemics.backends import BackendConfigError
    from apart_epistemics.models import ExperimentSuiteConfig

    cfg_default = ExperimentSuiteConfig(conditions=[])
    assert cfg_default.numeric_conditions == ["C1", "C2", "C3"]
    assert cfg_default.numeric_models == []

    task = build_numeric_core_tasks()[:1]
    backend = SyntheticEpistemicBackend(seed=5)

    with pytest.raises(BackendConfigError, match="drawn from"):
        run_numeric_core(task, ["C7"], 4, 1, 1, tmp_path / "bad_c.jsonl", backend)

    for bad_models in (None, [], ["m1"], ["m1", "m1"], ["m1", ""]):
        with pytest.raises(BackendConfigError):
            run_numeric_core(
                task, ["C5"], 4, 1, 1, tmp_path / "bad_m.jsonl", backend, models=bad_models
            )

    class NoForModel:
        def generate_numeric_reply(self, **_kwargs: Any) -> dict[str, Any]:
            return {"raw_reply": "Final estimate: 10"}

    with pytest.raises(BackendConfigError, match="multi-model"):
        run_numeric_core(
            task, ["C5"], 4, 1, 1, tmp_path / "no_fm.jsonl", NoForModel(), models=["a", "b"]
        )

    # C1..C3 without models omits numeric_models from run_spec (backward-compatible manifests).
    legacy_recs = run_numeric_core(
        task, ["C1", "C2", "C3"], 4, 1, 1, tmp_path / "legacy.jsonl", backend
    )
    assert "numeric_models" not in legacy_recs[0].run_spec
    assert len(analyze_numeric_core(legacy_recs, task)["contrasts"]) == 4

    # C1..C4 stay on base backend; C5/C6 cycle models[a_idx % len(models)].
    models = ["gemma-4-26b-a4b-it", "deepseek-v4-flash", "qwen3.8-27b", "glm-5.2"]
    recs = run_numeric_core(
        task,
        ["C1", "C4", "C5", "C6"],
        8,
        1,
        1,
        tmp_path / "poly.jsonl",
        backend,
        models=models,
    )
    assert recs[0].run_spec["numeric_models"] == models
    by_cond = {c: [r for r in recs if r.condition_id == c] for c in ("C1", "C4", "C5", "C6")}
    assert {r.model_slug for r in by_cond["C1"]} == {"openai/gpt-frontier"}
    assert {r.model_slug for r in by_cond["C4"]} == {"openai/gpt-frontier"}
    for cond in ("C5", "C6"):
        assert [r.model_slug for r in by_cond[cond]] == models * 2


def test_c1_through_c6_analysis_evrr_nan_guard_contrasts_and_interaction(tmp_path: Path) -> None:
    import json

    from apart_epistemics.cli import _execute_numeric_track, _serialize_numeric_analysis
    from apart_epistemics.models import ExperimentSuiteConfig

    tasks = build_numeric_core_tasks()[:4]
    models = ["gemma-4-26b-a4b-it", "deepseek-v4-flash", "qwen3.8-27b", "glm-5.2"]
    cfg = ExperimentSuiteConfig(
        seed=42,
        num_agents=8,
        num_rounds=2,
        tasks_per_track=4,
        numeric_tasks=4,
        numeric_replicates=1,
        numeric_conditions=["C1", "C2", "C3", "C4", "C5", "C6"],
        numeric_models=models,
        conditions=[],
    )
    analysis, recs = _execute_numeric_track(
        cfg,
        SyntheticEpistemicBackend(seed=42),
        tmp_path / "cli_out",
        n_agents=cfg.num_agents,
        budget_usd=50.0,
    )
    assert len(recs) == 4 * 6 * 8 * 2
    assert analysis["completion_status"] == "complete"
    assert analysis["missing_or_incomplete_cells"] == 0
    assert analysis["max_identity_residual"] < 1e-12
    assert len(analysis["contrasts"]) == 14

    c5_final = analysis["by_condition_final"]["C5"]
    c6_final = analysis["by_condition_final"]["C6"]
    for row in (c5_final, c6_final):
        assert row["between_model_diversity"] > 0.0
        assert row["within_model_diversity"] > 0.0
        assert row["between_model_diversity"] + row["within_model_diversity"] == pytest.approx(
            row["diversity"], abs=1e-5
        )
        assert row["evrr"] == pytest.approx(row["evrr_between"] + row["evrr_within"], abs=1e-5)
        assert row["evrr_cumulative"] is not None

    c2_final = analysis["by_condition_final"]["C2"]
    assert c2_final["between_model_diversity"] == 0.0
    assert c2_final["excess_between_model_diversity"] == 0.0

    inter = analysis["interaction_2x2"]
    assert set(inter) == {"crowd_error", "avg_individual_error", "diversity", "evrr"}
    for item in inter.values():
        assert item["n_cells"] == 4
        assert 0.0 <= item["wilcoxon_p"] <= item["holm_adjusted_p"] <= 1.0

    # Disk-persisted JSON includes completion metadata and interaction_2x2 without NaNs.
    saved = json.loads((tmp_path / "cli_out" / "numeric_analysis.json").read_text(encoding="utf-8"))
    assert saved["completion_status"] == "complete"
    assert saved["missing_or_incomplete_cells"] == 0
    assert set(saved["interaction_2x2"]) == set(inter)
    json.dumps(_serialize_numeric_analysis(analysis), allow_nan=False)

    # Partial run returns empty interaction_2x2 and incomplete_or_unverified.
    partial = analyze_numeric_core(recs[:-1], tasks)
    assert partial["completion_status"] == "incomplete_or_unverified"
    assert partial["interaction_2x2"] == {}

    # Zero-variance initial state (alpha=1.0 -> denom == 0.0) sets EVRR fields to None (never NaN).
    zero_denom_recs = run_numeric_core(
        tasks=tasks[:1],
        conditions=["C2", "C4", "C5", "C6"],
        n_agents=4,
        n_rounds=1,
        replicates=1,
        runs_path=tmp_path / "alpha1.jsonl",
        backend=SyntheticEpistemicBackend(seed=9),
        alpha=1.0,
        models=models[:2],
    )
    z_analysis = analyze_numeric_core(zero_denom_recs, tasks[:1])
    z_row = z_analysis["by_condition_final"]["C5"]
    assert z_row["evrr"] is None
    assert z_row["evrr_between"] is None
    assert z_row["evrr_within"] is None
    assert z_row["evrr_cumulative"] is None
    assert z_analysis["interaction_2x2"] != {}
    assert "evrr" not in z_analysis["interaction_2x2"]
    json.dumps(_serialize_numeric_analysis(z_analysis), allow_nan=False)


def test_h4_slopes_cover_c1_c6_round1_model_fe_and_crowd_pull() -> None:
    import random

    from apart_epistemics.engine.h4 import compute_h4_slopes
    from apart_epistemics.models import NumericEstimateRecord

    tasks = [
        TaskItem(
            id=f"q{i}",
            kind="numeric_estimate",
            domain="d",
            question=f"q{i}?",
            numeric_truth=10.0 ** (i + 1),
        )
        for i in range(4)
    ]
    rng = random.Random(17)
    recs: list[NumericEstimateRecord] = []
    model_offsets = {"gemma": -0.2, "deepseek": 0.05, "qwen": 0.15, "glm": 0.25}

    def add_rec(
        cond: str,
        qid: str,
        truth: float,
        rnd: int,
        slug: str,
        u_err: float,
        c_err: float | None,
        a_err: float,
    ) -> None:
        lt = math.log10(truth)
        recs.append(
            NumericEstimateRecord(
                condition_id=cond,
                question_id=qid,
                replicate=0,
                round_idx=rnd,
                agent_id=f"a_{slug}_{rnd}_{len(recs)}",
                model_slug=slug,
                provider="p",
                raw_reply="Final estimate: 100",
                parsed_value=10.0 ** (lt + a_err),
                assistant_log10=lt + a_err,
                agent_log10_before=lt + u_err,
                agent_log10_after=lt + u_err,
                pop_median_log10=None if c_err is None else lt + c_err,
                miss=False,
            )
        )

    for idx, t in enumerate(tasks):
        truth = float(t.numeric_truth or 100.0)
        pop_r1 = -0.4 + 0.25 * idx  # varies cleanly across questions
        pop_r2 = pop_r1 * 0.5
        for k in range(8):
            u1 = rng.uniform(-0.3, 0.3)
            u2 = rng.uniform(-0.3, 0.3)
            # C1: slope_user = 0.2, constant item baseline 0.1
            add_rec("C1", t.id, truth, 1, "gemma", u1, None, 0.1 + 0.2 * u1)
            add_rec("C1", t.id, truth, 2, "gemma", u2, None, 0.1 + 0.2 * u2)
            # C2: crowd pull = 0.6 * pop_r1 extra relative to C1 in Round 1
            add_rec("C2", t.id, truth, 1, "gemma", u1, pop_r1, 0.1 + 0.2 * u1 + 0.6 * pop_r1)
            add_rec("C2", t.id, truth, 2, "gemma", u2, pop_r2, 0.1 + 0.2 * u2 + 0.6 * pop_r2)
            # C4: low crowd pull = 0.1 * pop_r1 extra relative to C1 in Round 1
            add_rec("C4", t.id, truth, 1, "gemma", u1, pop_r1, 0.1 + 0.2 * u1 + 0.1 * pop_r1)
            add_rec("C4", t.id, truth, 2, "gemma", u2, pop_r2, 0.1 + 0.2 * u2 + 0.1 * pop_r2)
            # C5: K=4 multi-model with strong model offsets, true slope_user = 0.35
            slug = list(model_offsets)[k % 4]
            off = model_offsets[slug]
            add_rec("C5", t.id, truth, 1, slug, u1, pop_r1, off + 0.35 * u1 + 0.3 * pop_r1)
            add_rec("C5", t.id, truth, 2, slug, u2, pop_r2, off + 0.35 * u2 + 0.3 * pop_r2)

    h4 = compute_h4_slopes(recs, tasks)
    by_cond = h4["by_condition"]
    assert "slope_crowd" not in by_cond["C1"]
    assert "slope_crowd" in by_cond["C4"]
    assert "slope_crowd" in by_cond["C5"]
    assert by_cond["C2"]["per_question"]["q0"]["crowd_low_variance"] is True
    assert by_cond["C1"]["slope_user_round1"]["mean"] == pytest.approx(0.2, abs=1e-3)
    assert by_cond["C2"]["slope_user_round1"]["mean"] == pytest.approx(0.2, abs=1e-3)
    assert by_cond["C4"]["slope_user_round1"]["mean"] == pytest.approx(0.2, abs=1e-3)
    assert "slope_user_model_fe" not in by_cond["C1"]
    assert by_cond["C5"]["slope_user_model_fe"]["mean"] == pytest.approx(0.35, abs=1e-3)

    pull = h4["crowd_pull_vs_c1_r1"]
    assert set(pull) == {"C2", "C4"}  # C5 is multi-model and excluded
    assert pull["C2"]["slope"] == pytest.approx(0.6, abs=1e-3)
    assert pull["C4"]["slope"] == pytest.approx(0.1, abs=1e-3)
    assert pull["C2"]["n_questions"] == 4


def test_p_series_prompts_prior_symmetry_and_model_routing(tmp_path: Path) -> None:
    from apart_epistemics.backends import BackendConfigError
    from apart_epistemics.datasets import generate_agent_population
    from apart_epistemics.engine.numeric_core import p_series_system_prompt

    agent = generate_agent_population(num_agents=1, seed=42)[0]
    sys_nodate = p_series_system_prompt("")
    sys_dated = p_series_system_prompt("2026-11-15")
    assert "rigorous quantitative scientific estimator" in sys_nodate
    assert sys_dated == f"{sys_nodate} Today's date is 2026-11-15."

    def prompts(cond: str) -> tuple[str, str]:
        return build_numeric_prompts(
            condition_id=cond,
            agent=agent,
            question="What is Earth's mean radius?",
            current_x=6500.0,
            pop_stats=(6400.0, 6100.0, 6700.0),
            history=[("5000", "6000")],
            unit="kilometers",
            current_date="2026-11-15",
        )

    s0, u0 = prompts("P0")
    s1, u1 = prompts("P1")
    s2, u2 = prompts("P2")
    s3, u3 = prompts("P3")
    s4, u4 = prompts("P4")
    assert s0 == s1 == s2 == s3 == s4 == sys_dated
    assert u1 == u3
    assert u2 == u4
    assert "Prior estimate: 6,500 kilometers." in u0
    assert "Peer consensus telemetry from prior estimators reports a median of" not in u0
    assert "Peer consensus telemetry from prior estimators reports a median of 6,400" in u1
    assert "(interquartile range 6,100 to 6,700)." in u1
    assert "Dialectical counter-anchoring protocol:" not in u1
    assert "Dialectical counter-anchoring protocol:" in u2
    assert "prior estimate and peer median" in u2
    assert u2.index("median of") < u2.index("Dialectical counter-anchoring protocol:")
    assert "End with 'Final estimate: <plain number in kilometers>'" in u0
    # C1..C6 prompt builders remain untouched.
    sys_c1, usr_c1 = build_numeric_prompts("C1", agent, "Q?", 100.0, None, [], "km", "2026-11-15")
    assert "neutral quantitative estimation assistant" in sys_c1
    assert "Question: Q?" in usr_c1

    # Validation & K=4 routing on P3/P4 vs K=1 on P0..P2, plus identical R1 agent priors.
    backend = SyntheticEpistemicBackend(seed=11)
    task = build_numeric_core_tasks()[:1]
    with pytest.raises(BackendConfigError):
        run_numeric_core(task, ["P3"], 4, 1, 1, tmp_path / "bad_p3.jsonl", backend, models=["m1"])
    models = ["gemma-4-26b-a4b-it", "deepseek-v4-flash", "qwen3.8-27b", "glm-5.2"]
    recs = run_numeric_core(
        task,
        ["P0", "P1", "P2", "P3", "P4"],
        8,
        1,
        1,
        tmp_path / "p_route.jsonl",
        backend,
        models=models,
        current_date="2026-11-15",
    )
    by_cond = {c: [r for r in recs if r.condition_id == c] for c in ("P0", "P1", "P2", "P3", "P4")}
    priors_p0 = [r.agent_log10_before for r in by_cond["P0"]]
    for cond in ("P1", "P2", "P3", "P4"):
        assert [r.agent_log10_before for r in by_cond[cond]] == priors_p0
    for cond in ("P0", "P1", "P2"):
        assert {r.model_slug for r in by_cond[cond]} == {"openai/gpt-frontier"}
    for cond in ("P3", "P4"):
        assert [r.model_slug for r in by_cond[cond]] == models * 2


def test_p_series_analysis_matrix_did_h4_and_benchmark_disjointness(tmp_path: Path) -> None:
    import json

    from apart_epistemics.cli import (
        _execute_numeric_track,
        _load_config,
        _serialize_numeric_analysis,
    )
    from apart_epistemics.models import ExperimentSuiteConfig

    bench_path = Path("configs/polyphony_benchmark_tasks.json")
    v3_path = Path("configs/numeric_final_tasks.json")
    cfg_path = Path("configs/polyphony_p0_p4.yaml")
    bench = [TaskItem.model_validate(item) for item in json.loads(bench_path.read_text())]
    v3 = [TaskItem.model_validate(item) for item in json.loads(v3_path.read_text())]
    legacy = build_numeric_core_tasks()

    assert len(bench) == 12
    assert len({t.id for t in bench}) == 12
    assert {t.id for t in bench}.isdisjoint({t.id for t in v3} | {t.id for t in legacy})
    assert {t.question.strip().lower() for t in bench}.isdisjoint(
        {t.question.strip().lower() for t in v3} | {t.question.strip().lower() for t in legacy}
    )
    assert all((t.numeric_truth or 0.0) >= 100.0 for t in bench)
    by_stratum = {
        s: [t for t in bench if t.selection_stratum == s]
        for s in ("control", "low_spread", "high_spread")
    }
    assert all(len(v) == 4 for v in by_stratum.values())
    for tasks_s in by_stratum.values():
        assert sorted(t.popular_misconception_bias for t in tasks_s) == [-0.8, -0.8, 0.8, 0.8]

    loaded_cfg = _load_config(cfg_path)
    assert loaded_cfg.numeric_conditions == ["P0", "P1", "P2", "P3", "P4"]
    assert len(loaded_cfg.numeric_models) == 4
    assert loaded_cfg.alpha == 0.3

    # Full 4-task synthetic run checking 16 contrasts, 5x9 matrix, 2x2 DiD, H4, and serialization.
    models = loaded_cfg.numeric_models
    sub_cfg = ExperimentSuiteConfig(
        seed=42,
        num_agents=8,
        num_rounds=2,
        tasks_per_track=4,
        numeric_tasks=4,
        numeric_replicates=1,
        alpha=0.3,
        numeric_conditions=["P0", "P1", "P2", "P3", "P4"],
        numeric_models=models,
        conditions=[],
    )
    analysis, recs = _execute_numeric_track(
        sub_cfg,
        SyntheticEpistemicBackend(seed=42),
        tmp_path / "p_out",
        n_agents=8,
        budget_usd=50.0,
    )
    assert len(recs) == 4 * 5 * 8 * 2
    assert analysis["completion_status"] == "complete"
    assert analysis["max_identity_residual"] < 1e-12
    assert len(analysis["contrasts"]) == 16
    assert "p4_vs_p1_excess_between_model_diversity" in {
        c.contrast_id for c in analysis["contrasts"]
    }
    assert set(analysis["interaction_2x2"]) == {
        "crowd_error",
        "avg_individual_error",
        "diversity",
        "evrr",
    }

    matrix = analysis["polyphony_contrast_matrix"]
    assert set(matrix) == {
        "crowd_error",
        "diversity",
        "excess_between_model_diversity",
        "between_model_diversity",
        "evrr",
    }
    for metric, items in matrix.items():
        assert len(items) == 9
        assert all(item.metric == metric for item in items)

    # Confirm H4 crowd pull anchors P1/P2 vs P0 (never P3/P4) and includes model FE on P3/P4.
    h4 = analysis["h4_slopes"]
    assert set(h4["crowd_pull_vs_c1_r1"]) == {"P1", "P2"}
    assert "slope_crowd" not in h4["by_condition"]["P0"]
    for p_social in ("P1", "P2", "P3", "P4"):
        assert "slope_crowd" in h4["by_condition"][p_social]
    assert "slope_user_model_fe" in h4["by_condition"]["P3"]
    assert "slope_user_model_fe" in h4["by_condition"]["P4"]

    # Saved JSON preserves polyphony_contrast_matrix cleanly with allow_nan=False.
    saved = json.loads((tmp_path / "p_out" / "numeric_analysis.json").read_text(encoding="utf-8"))
    assert set(saved["polyphony_contrast_matrix"]) == set(matrix)
    json.dumps(_serialize_numeric_analysis(analysis), allow_nan=False)

    # Incomplete run and alpha=1.0 zero-EVRR guard keep polyphony_contrast_matrix safe.
    assert analyze_numeric_core(recs[:-1], legacy[:4])["polyphony_contrast_matrix"] == {}
    z_recs = run_numeric_core(
        legacy[:2],
        ["P0", "P1", "P2", "P3", "P4"],
        4,
        1,
        1,
        tmp_path / "pz.jsonl",
        SyntheticEpistemicBackend(seed=7),
        alpha=1.0,
        models=models[:2],
    )
    z_analysis = analyze_numeric_core(z_recs, legacy[:2])
    assert z_analysis["polyphony_contrast_matrix"]["evrr"] == []
    assert "evrr" not in z_analysis["interaction_2x2"]
    json.dumps(_serialize_numeric_analysis(z_analysis), allow_nan=False)

    # Legacy C-series runs emit empty polyphony_contrast_matrix.
    c_recs = run_numeric_core(
        legacy[:1],
        ["C1", "C2", "C3"],
        4,
        1,
        1,
        tmp_path / "c_only.jsonl",
        SyntheticEpistemicBackend(seed=5),
    )
    assert analyze_numeric_core(c_recs, legacy[:1])["polyphony_contrast_matrix"] == {}
