"""Shared fixtures and deterministic offline stubs for apart_epistemics tests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from apart_epistemics.models import AgentProfile, TaskItem

BENCHMARK_TASKS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "polyphony_benchmark_tasks.json"
)


def sample_numeric_tasks(limit: int | None = None) -> list[TaskItem]:
    """Load frozen benchmark tasks from configs/polyphony_benchmark_tasks.json."""
    raw = json.loads(BENCHMARK_TASKS_PATH.read_text(encoding="utf-8"))
    tasks = [TaskItem.model_validate(item) for item in raw]
    return tasks[:limit] if limit is not None else tasks


class StubNumericBackend:
    """Stateless, deterministic offline numeric backend for unit and engine tests."""

    def __init__(self, seed: int = 42, model: str = "openai/gpt-frontier") -> None:
        self.seed = seed
        self.model = model
        self.provider = "stub"
        self.budget_ledger: Any = None

    def for_model(self, model: str) -> StubNumericBackend:
        """Return a routed stub instance pinned to the requested model slug."""
        clone = copy.copy(self)
        clone.model = model
        return clone

    @staticmethod
    def _unit_float(key: str) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def generate_numeric_reply(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        condition_id: str = "C1",
        task: TaskItem | None = None,
        round_idx: int = 1,
        current_x: float = 100.0,
        pop_stats: tuple[float, float, float] | None = None,
        agent: AgentProfile | None = None,
        max_tokens: int = 1024,
        **_extra: Any,
    ) -> dict[str, Any]:
        """Generate a deterministic, parseable numeric reply on the log10 scale."""
        del system_prompt, user_prompt, max_tokens
        truth = float(task.numeric_truth) if (task and task.numeric_truth) else float(current_x)
        log_truth = math.log10(max(truth, 1e-6))
        log_cur = math.log10(max(float(current_x), 1e-6))

        # Model-specific shift produces non-zero between-model variance under multi-model routing.
        model_u = self._unit_float(f"model:{self.seed}:{self.model}")
        model_shift = (model_u - 0.5) * 0.36

        task_id = task.id if task is not None else "unknown"
        agent_id = agent.agent_id if agent is not None else "agent_00"
        cell_u = self._unit_float(
            f"cell:{self.seed}:{self.model}:{condition_id}:{task_id}:{round_idx}:{agent_id}"
        )
        noise = (cell_u - 0.5) * 0.24

        cond_upper = condition_id.upper()
        social_conditions = {
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "P1",
            "P2",
            "P3",
            "P4",
            "P1_REFLECT",
            "P1_DS",
            "P1_QWEN",
            "P1_GLM",
        }
        if pop_stats is not None and cond_upper in social_conditions:
            log_pop = math.log10(max(float(pop_stats[0]), 1e-6))
            target_log = 0.35 * log_truth + 0.35 * log_cur + 0.30 * log_pop + model_shift + noise
        else:
            target_log = 0.55 * log_truth + 0.45 * log_cur + model_shift + noise

        # Stay well inside _parse_reply's 3.0-decade sanity bound around current_x.
        bounded_log = max(log_cur - 1.5, min(log_cur + 1.5, target_log))
        estimate = 10.0**bounded_log
        return {
            "raw_reply": f"Final estimate: {estimate:.6g}",
            "model_slug": self.model,
            "provider": self.provider,
            "input_tokens": 120,
            "output_tokens": 30,
            "reasoning_tokens": 0,
            "finish_reason": "stop",
            "cost_usd": 0.0001,
        }


@pytest.fixture
def sample_agent() -> AgentProfile:
    """Return a reproducible sample agent profile."""
    return AgentProfile(
        agent_id="agent_00",
        name="Ananya Patel",
        region="south_asia",
        profession="civil engineer",
        age_band="25-34",
        values=["empirical_rigor", "fairness"],
        prior_belief=0.55,
        prior_confidence=0.80,
        need_for_cognition=0.75,
        preferred_topics=["climate_science", "public_health"],
    )
