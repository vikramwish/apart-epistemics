"""Current-date line in the shared system prompt, and the unusable-answer filter."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import StubNumericBackend, sample_numeric_tasks

from apart_epistemics.backends import BackendConfigError
from apart_epistemics.cli import _resolve_current_date, main
from apart_epistemics.datasets.selection import cold_prompts
from apart_epistemics.engine import build_numeric_prompts, run_numeric_core
from apart_epistemics.engine.numeric_core import shared_system_prompt
from apart_epistemics.models import AgentProfile, TaskItem

DATE = "2026-11-13"
LINE = f"Today's date is {DATE}."
SOURCE = "Example Survey Bureau, 2023 annual report, table 4"


def _agent() -> AgentProfile:
    return AgentProfile(
        agent_id="a",
        name="a",
        region="south_asia",
        profession="civil engineer",
        prior_belief=0.5,
        prior_confidence=0.6,
        need_for_cognition=0.5,
    )


def _task(qid: str, truth: float = 100.0) -> TaskItem:
    return TaskItem(
        id=qid,
        kind="numeric_estimate",
        domain="d",
        question=f"Question {qid}?",
        numeric_truth=truth,
        unit="km",
        source_note=SOURCE,
    )


# ------------------------------------------------------------------ date line


def test_shared_prompt_gets_the_date_and_is_unchanged_without_one() -> None:
    plain = shared_system_prompt()
    assert "Today's date" not in plain
    assert plain.startswith("You are a neutral quantitative estimation assistant.")
    assert shared_system_prompt(DATE) == f"{plain} {LINE}"


def test_every_condition_carries_the_same_date_line() -> None:
    def system_for(cond: str) -> str:
        return build_numeric_prompts(
            condition_id=cond,
            agent=_agent(),
            question="q",
            current_x=10.0,
            pop_stats=(10.0, 5.0, 20.0),
            history=[],
            unit="km",
            current_date=DATE,
        )[0]

    c1, c2, c3 = system_for("C1"), system_for("C2"), system_for("C3")
    assert c1 == c2
    assert LINE in c1
    assert c3.startswith(c1)  # C3 still only adds the profile sentence
    assert "civil engineer" in c3[len(c1) :]


def test_cold_prompt_uses_the_same_system_prompt_as_the_experiment() -> None:
    system, user = cold_prompts(_task("a"), DATE)
    exp_system = build_numeric_prompts(
        condition_id="C1",
        agent=_agent(),
        question="q",
        current_x=1.0,
        pop_stats=None,
        history=[],
        current_date=DATE,
    )[0]
    assert system == exp_system
    assert LINE in system
    assert "current estimate" not in user  # the user turn is still cold
    assert LINE not in cold_prompts(_task("a"))[0]  # no date unless asked


def test_runner_sends_the_date_and_records_it_in_the_manifest(tmp_path: Path) -> None:
    seen: list[str] = []

    class Spy(StubNumericBackend):
        def generate_numeric_reply(self, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["system_prompt"])
            return super().generate_numeric_reply(**kwargs)

    kwargs: dict[str, Any] = {
        "tasks": sample_numeric_tasks(1),
        "conditions": ["C1", "C3"],
        "n_agents": 3,
        "n_rounds": 1,
        "replicates": 1,
        "runs_path": tmp_path / "runs.jsonl",
        "backend": Spy(seed=1),
    }
    recs = run_numeric_core(**kwargs, current_date=DATE)
    assert seen
    assert all(LINE in s for s in seen)
    assert recs[0].run_spec["current_date"] == DATE
    run_numeric_core(**kwargs, current_date=DATE)  # same date resumes fine
    with pytest.raises(RuntimeError, match="new output directory"):
        run_numeric_core(**kwargs, current_date="2026-11-14")


def test_current_date_resolution_flag_resume_and_default(tmp_path: Path) -> None:
    assert _resolve_current_date("2026-01-02") == "2026-01-02"
    with pytest.raises(BackendConfigError, match="YYYY-MM-DD"):
        _resolve_current_date("13/11/2026")
    assert _resolve_current_date(None, tmp_path) == date.today().isoformat()
    (tmp_path / "numeric_runs.manifest.json").write_text(
        json.dumps({"current_date": "2026-03-04"}), encoding="utf-8"
    )
    assert _resolve_current_date(None, tmp_path) == "2026-03-04"  # a resumed run keeps its date
    assert _resolve_current_date("2026-05-06", tmp_path) == "2026-05-06"  # flag still wins


def test_cli_run_records_the_date_and_a_resume_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    cfg = tmp_path / "suite.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "num_agents": 4,
                "num_rounds": 1,
                "tasks_per_track": 1,
                "numeric_tasks": 1,
                "numeric_tasks_file": "configs/polyphony_benchmark_tasks.json",
                "numeric_replicates": 1,
                "numeric_conditions": ["P0"],
                "conditions": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "apart_epistemics.cli.create_backend",
        lambda *_a, **_k: StubNumericBackend(seed=1),
    )
    out = tmp_path / "run"
    base = ["run", "--config", str(cfg), "--backend", "openai", "--out", str(out)]
    assert main([*base, "--current-date", DATE]) == 0
    row = json.loads((out / "numeric_runs.jsonl").read_text().splitlines()[0])
    assert row["run_spec"]["current_date"] == DATE
    assert main(base) == 0  # no flag: reuses the saved date instead of today's, so it resumes
    assert main([*base, "--current-date", "2026-11-14"]) == 2  # a different date is refused


def test_cli_rejects_a_malformed_date(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["run", "--dry-run", "--out", str(tmp_path), "--current-date", "tomorrow"])
    assert rc == 2
    assert "YYYY-MM-DD" in capsys.readouterr().err
