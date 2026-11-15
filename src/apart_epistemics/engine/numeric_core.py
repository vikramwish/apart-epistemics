"""Kevin's crash-resumable C1/C2/C3 numeric estimation engine and Page's Theorem analyzer."""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from apart_epistemics.backends import BackendConfigError, BudgetExceededError
from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.datasets import generate_agent_population
from apart_epistemics.engine.contrasts import compute_paired_holm_contrasts
from apart_epistemics.engine.h4 import compute_h4_slopes
from apart_epistemics.metrics import (
    bootstrap_ci,
    compute_hierarchical_page_decomposition,
    compute_page_decomposition,
    holm_bonferroni,
    paired_wilcoxon_p,
    pearson_corr,
    quantile_sorted,
)
from apart_epistemics.models import AgentProfile, NumericEstimateRecord, TaskItem
from apart_epistemics.parsers import NumericParseResult, parse_numeric_estimate
from apart_epistemics.run_state import backend_identity, configure_budget, validate_manifest

# A parsed value further than this (in log10 units) from the agent's current estimate is
# treated as a parse miss: a factor of 100 catches 10^3 and 10^6 unit slips (e.g. 235,000 vs
# 235,000,000 gave distance 2.898) while preserving genuine estimates (max observed jump 1.013).
MAX_LOG10_JUMP = 2.0

_P_CONTROL_CONDITIONS = frozenset(
    {"P0_POLY", "P1_REFLECT", "P0_DS", "P1_DS", "P0_QWEN", "P1_QWEN", "P0_GLM", "P1_GLM"}
)
_P_SERIES_CONDITIONS = frozenset({"P0", "P1", "P2", "P3", "P4"}) | _P_CONTROL_CONDITIONS
_VALID_NUMERIC_CONDITIONS = frozenset({"C1", "C2", "C3", "C4", "C5", "C6"}) | _P_SERIES_CONDITIONS
_SOCIAL_CONDITIONS = frozenset(
    {
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
)
_COUNTER_ANCHOR_CONDITIONS = frozenset({"C4", "C6", "P2", "P4"})
_REFLECT_CONDITIONS = frozenset({"P1_REFLECT"})
_POLYPHONY_CONDITIONS = frozenset({"C5", "C6", "P3", "P4", "P0_POLY"})
_CONSTITUENT_MODEL_TOKENS: dict[str, tuple[str, int]] = {
    "P0_DS": ("deepseek", 1),
    "P1_DS": ("deepseek", 1),
    "P0_QWEN": ("qwen", 2),
    "P1_QWEN": ("qwen", 2),
    "P0_GLM": ("glm", 3),
    "P1_GLM": ("glm", 3),
}
_COUNTER_ANCHOR_TEXT = (
    "Before estimating, apply dialectical counter-anchoring: in 2-3 concise sentences, "
    "critically examine how both my current estimate and the crowd median could be skewed "
    "(either too high or too low) by shared misconceptions or herding, and check independent "
    "base rates or known reference facts before forming your own independent estimate."
)
_P_COUNTER_ANCHOR_TEXT = (
    "Dialectical counter-anchoring protocol: before estimating, write 2-3 concise sentences "
    "testing both tails \u2014 identify specific physical scaling mechanisms or misconceptions by "
    "which both the prior estimate and peer median could systematically overshoot (too high) OR "
    "undershoot (too low) the true value \u2014 then derive your estimate from independent "
    "physical principles and reference scales."
)
_P_GENERIC_REFLECT_TEXT = (
    "Structured verification protocol: before estimating, write 2-3 concise sentences checking "
    "the quantity carefully \u2014 verify exact unit definitions, confirm any geographical or "
    "historical time bounds, and audit each step of intermediate arithmetic to ensure clear "
    "order-of-magnitude plausibility \u2014 then derive your final estimate directly from "
    "verified physical principles and reference scales."
)
_P_MATRIX_PAIRS = (
    ("P1", "P4"),
    ("P1", "P3"),
    ("P1", "P2"),
    ("P3", "P4"),
    ("P2", "P4"),
    ("P0", "P4"),
    ("P0", "P1"),
    ("P0", "P3"),
    ("P0", "P2"),
)
_P_MATRIX_METRICS = (
    "crowd_error",
    "assistant_crowd_error",
    "diversity",
    "excess_between_model_diversity",
    "between_model_diversity",
    "evrr",
)
_ROUND_METRIC_KEYS = (
    "crowd_error",
    "avg_individual_error",
    "diversity",
    "between_model_diversity",
    "within_model_diversity",
    "excess_between_model_diversity",
    "between_share",
    "iqr",
    "miss_rate",
    "evrr",
    "evrr_between",
    "evrr_within",
    "evrr_cumulative",
    "assistant_crowd_error",
    "assistant_avg_individual_error",
    "assistant_diversity",
    "assistant_between_model_diversity",
    "assistant_within_model_diversity",
    "assistant_excess_between_model_diversity",
    "assistant_between_share",
)


def format_prompt_number(value: float) -> str:
    """Round to 3 significant figures, then to a whole number, with commas and no e-notation."""
    rounded = round(float(f"{value:.3g}"))
    if value > 0 and rounded < 1:
        rounded = 1
    return f"{rounded:,}"


def _parse_reply(raw_reply: str, cur_x: float) -> NumericParseResult:
    """Parse a reply, turning implausibly distant values into misses."""
    parsed = parse_numeric_estimate(raw_reply, require_positive=True)
    if (
        not parsed.miss
        and parsed.log10_value is not None
        and abs(parsed.log10_value - math.log10(cur_x)) > MAX_LOG10_JUMP
    ):
        return NumericParseResult(
            value=None, log10_value=None, miss=True, matched_span=parsed.matched_span
        )
    return parsed


def shared_system_prompt(current_date: str = "") -> str:
    """The system prompt every condition (and the cold selection/movability probes) shares.

    `current_date` (ISO, e.g. 2026-11-13) tells the model today's date so it does not insist
    that events after its training data have not happened. Empty leaves the prompt unchanged.
    """
    prompt = (
        "You are a neutral quantitative estimation assistant. "
        "Reason step by step without web lookup and end with 'Final estimate: <number>'."
    )
    return f"{prompt} Today's date is {current_date}." if current_date else prompt


def p_series_system_prompt(current_date: str = "") -> str:
    """Return the neutral P0..P4 scientific estimator system prompt."""
    prompt = (
        "You are a quantitative scientific estimator. "
        "Answer without web lookup and end with 'Final estimate: <number>'."
    )
    return f"{prompt} Today's date is {current_date}." if current_date else prompt


def _build_p_series_prompts(
    cond: str,
    question: str,
    est_str: str,
    pop_stats: tuple[float, float, float] | None,
    unit: str,
    current_date: str,
) -> tuple[str, str]:
    unit_str = f" {unit}" if unit else ""
    in_unit = f" in {unit}" if unit else ""
    parts = [question, f"Prior estimate: {est_str}{unit_str}."]
    if cond in _SOCIAL_CONDITIONS and pop_stats is not None:
        med, q25, q75 = pop_stats
        parts.append(
            "Peer consensus telemetry from prior estimators reports a median of "
            f"{format_prompt_number(med)} "
            f"(interquartile range {format_prompt_number(q25)} to {format_prompt_number(q75)})."
        )
    if cond in _COUNTER_ANCHOR_CONDITIONS:
        parts.append(_P_COUNTER_ANCHOR_TEXT)
    elif cond in _REFLECT_CONDITIONS:
        parts.append(_P_GENERIC_REFLECT_TEXT)
    parts.append(
        "What is your best estimate of the true quantity? "
        f"End with 'Final estimate: <plain number{in_unit}>' and nothing after the number."
    )
    return (p_series_system_prompt(current_date), "\n".join(parts))


def build_numeric_prompts(
    condition_id: str,
    agent: AgentProfile,
    question: str,
    current_x: float,
    pop_stats: tuple[float, float, float] | None,
    history: list[tuple[str, str]],
    unit: str = "",
    current_date: str = "",
) -> tuple[str, str]:
    """Construct C1..C6 and P0..P4 numeric estimation prompts."""
    cond = condition_id.upper()
    est_str = format_prompt_number(current_x)
    if cond in _P_SERIES_CONDITIONS:
        return _build_p_series_prompts(cond, question, est_str, pop_stats, unit, current_date)

    # One shared system prompt for C1/C2/C4/C5/C6 so conditions differ only in what they add:
    # C2/C5 add social summary, C4/C6 add counter-anchoring, C3 adds profile sentence + history.
    system_prompt = shared_system_prompt(current_date)
    if cond == "C3":
        system_prompt += (
            f" You are speaking with a {agent.profession} in {agent.region} "
            f"(age band {agent.age_band}) who values {', '.join(agent.values) or 'evidence'}."
        )

    parts = [f"Question: {question}"]
    if cond in _SOCIAL_CONDITIONS and pop_stats is not None:
        med, q25, q75 = pop_stats
        parts.append(
            "Other people asking this recently gave a median of "
            f"{format_prompt_number(med)} "
            f"(IQR {format_prompt_number(q25)} to {format_prompt_number(q75)})."
        )
    if cond == "C3" and history:
        recent = history[-2:]
        hist_lines = [
            f"Turn {idx + 1} User: {u} | Assistant: {a}" for idx, (u, a) in enumerate(recent)
        ]
        parts.append("Recent conversation history:\n" + "\n".join(hist_lines))
    if cond in _COUNTER_ANCHOR_CONDITIONS:
        parts.append(_COUNTER_ANCHOR_TEXT)

    unit_str = f" {unit}" if unit else ""
    parts.append(
        f"My current estimate is {est_str}{unit_str}. "
        "What is your best estimate of the true value? "
        f"End with 'Final estimate: <plain number{f' in {unit}' if unit else ''}>' "
        "and nothing after the number."
    )
    return (system_prompt, "\n\n".join(parts))


def _load_existing_numeric_records(
    runs_path: Path,
) -> dict[tuple[str, str, int, int, str], NumericEstimateRecord]:
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord] = {}
    if not runs_path.exists():
        return existing
    raw_lines = runs_path.read_text(encoding="utf-8").splitlines()
    had_truncated_tail = False
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = NumericEstimateRecord.model_validate_json(stripped)
        except ValidationError:
            # Tolerate a single truncated final line if a previous run died mid-write.
            if idx == len(raw_lines) - 1:
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    had_truncated_tail = True
                    continue
            raise
        key = (rec.condition_id, rec.question_id, rec.replicate, rec.round_idx, rec.agent_id)
        existing[key] = rec
    if had_truncated_tail:
        clean_lines = [r.model_dump_json() for r in existing.values()]
        runs_path.write_text(
            "\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8"
        )
    return existing


@dataclass(frozen=True)
class TokenPricing:
    """USD per 1M tokens. Set per model; the defaults are a rough placeholder, not a price list."""

    input_per_m: float = 0.5
    output_per_m: float = 1.5


def _estimate_call_cost_usd(resp: dict[str, Any], pricing: TokenPricing | None = None) -> float:
    """Cost of one call (or one saved record): the provider's real charge when reported.

    Otherwise estimated from tokens. `output_tokens` (completion tokens) already includes hidden
    reasoning tokens, so `reasoning_tokens` is a breakdown and is not added again here.
    """
    real = resp.get("cost_usd")
    if real is not None:
        return float(real)
    price = pricing or TokenPricing()
    in_tok = int(resp.get("input_tokens", 0) or 0)
    out_tok = int(resp.get("output_tokens", 0) or 0)
    return max(1e-5, (in_tok * price.input_per_m + out_tok * price.output_per_m) / 1_000_000.0)


class _SpendTracker:
    """Thread-safe running cost estimate checked against the budget before each call."""

    def __init__(self, budget_usd: float, spent_usd: float = 0.0) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self._lock = threading.Lock()

    def check(self, message: str = "Budget cap exhausted") -> None:
        with self._lock:
            if self.budget_usd <= 0.0 or self.spent_usd >= self.budget_usd:
                raise BudgetExceededError(
                    f"{message} (spent={self.spent_usd:.4f}, cap={self.budget_usd:.4f})."
                )

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd


def _query_with_retry(
    backend: Any,
    sys_p: str,
    usr_p: str,
    cond_id: str,
    task: TaskItem,
    round_idx: int,
    cur_x: float,
    pop_stats: tuple[float, float, float],
    agent: AgentProfile,
    tracker: _SpendTracker,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
) -> tuple[dict[str, Any], Any]:
    """Query backend once, retrying with an explicit format reminder on parse failure."""

    costs: list[float | None] = []

    def call(user_prompt: str) -> tuple[dict[str, Any], Any]:
        resp = backend.generate_numeric_reply(
            system_prompt=sys_p,
            user_prompt=user_prompt,
            condition_id=cond_id,
            task=task,
            round_idx=round_idx,
            current_x=cur_x,
            pop_stats=pop_stats,
            agent=agent,
            max_tokens=max_tokens,
        )
        tracker.add(_estimate_call_cost_usd(resp, pricing))
        costs.append(resp.get("cost_usd"))
        return resp, _parse_reply(str(resp.get("raw_reply", "")), cur_x)

    resp, parsed = call(usr_p)
    if parsed.miss:
        tracker.check("Budget cap exhausted during retry")
        resp, parsed = call(
            f"{usr_p}\n\nRespond with a single positive number prefixed by 'Final estimate: '."
        )
    # The row's cost covers every attempt; unknown if any attempt did not report one.
    row_cost = sum(c for c in costs if c is not None) if all(c is not None for c in costs) else None
    return {**resp, "cost_usd": row_cost}, parsed


@dataclass(frozen=True)
class _RoundContext:
    """Everything that is fixed for one (condition, question, replicate, round) cell."""

    backend: Any
    agents: list[AgentProfile]
    cond_id: str
    task: TaskItem
    round_idx: int
    pop_stats: tuple[float, float, float]
    current_logs: list[float]
    histories: list[list[tuple[str, str]]]
    tracker: _SpendTracker
    max_tokens: int
    pricing: TokenPricing | None
    abort: threading.Event
    current_date: str = ""
    models: tuple[str, ...] = ()
    routed_backends: dict[str, Any] = field(default_factory=dict)


def _resolve_condition_model(
    cond_upper: str, a_idx: int, models: tuple[str, ...] | list[str]
) -> str | None:
    """Resolve the routed model slug for polyphonic or constituent-homogeneous conditions."""
    if cond_upper in _POLYPHONY_CONDITIONS and models:
        return models[a_idx % len(models)]
    if cond_upper in _CONSTITUENT_MODEL_TOKENS and models:
        token, fallback_idx = _CONSTITUENT_MODEL_TOKENS[cond_upper]
        for model_name in models:
            if token in model_name.lower():
                return model_name
        if fallback_idx < len(models):
            return models[fallback_idx]
    return None


def _query_agent(ctx: _RoundContext, a_idx: int) -> tuple[str, Any]:
    """Query one agent from round-start state; never raises, so a pool can collect outcomes."""
    if ctx.abort.is_set():
        return ("skipped", None)
    try:
        ctx.tracker.check()
        cur_x = 10.0 ** ctx.current_logs[a_idx]
        cond_upper = ctx.cond_id.upper()
        sys_p, usr_p = build_numeric_prompts(
            condition_id=ctx.cond_id,
            agent=ctx.agents[a_idx],
            question=ctx.task.question,
            current_x=cur_x,
            pop_stats=ctx.pop_stats if cond_upper in _SOCIAL_CONDITIONS else None,
            history=ctx.histories[a_idx],
            unit=ctx.task.unit,
            current_date=ctx.current_date,
        )
        routed_model = _resolve_condition_model(cond_upper, a_idx, ctx.models)
        agent_backend = (
            ctx.routed_backends[routed_model] if routed_model is not None else ctx.backend
        )
        return (
            "ok",
            _query_with_retry(
                agent_backend,
                sys_p,
                usr_p,
                ctx.cond_id,
                ctx.task,
                ctx.round_idx,
                cur_x,
                ctx.pop_stats,
                ctx.agents[a_idx],
                ctx.tracker,
                ctx.max_tokens,
                ctx.pricing,
            ),
        )
    except Exception as exc:
        ctx.abort.set()
        return ("error", exc)


def _make_record(
    spec: dict[str, Any],
    ctx: _RoundContext,
    rep: int,
    a_idx: int,
    resp: dict[str, Any],
    parsed: Any,
    alpha: float,
) -> NumericEstimateRecord:
    """Apply the damped update for one answered query and return its record."""
    before_log = ctx.current_logs[a_idx]
    if parsed.miss or parsed.log10_value is None:
        after_log, asst_log, parsed_val, is_miss = before_log, None, None, True
    else:
        asst_log, parsed_val, is_miss = parsed.log10_value, parsed.value, False
        after_log = (1.0 - alpha) * before_log + alpha * asst_log
    return NumericEstimateRecord(
        run_spec=spec,
        condition_id=ctx.cond_id,
        question_id=ctx.task.id,
        replicate=rep,
        round_idx=ctx.round_idx,
        agent_id=ctx.agents[a_idx].agent_id,
        model_slug=str(resp.get("model_slug", "synthetic")),
        provider=str(resp.get("provider", "synthetic_sim")),
        raw_reply=str(resp.get("raw_reply", "")),
        parsed_value=parsed_val,
        assistant_log10=asst_log,
        agent_log10_before=before_log,
        agent_log10_after=after_log,
        pop_median_log10=math.log10(ctx.pop_stats[0]),
        miss=is_miss,
        reasoning_tokens=int(resp.get("reasoning_tokens", 0) or 0),
        input_tokens=int(resp.get("input_tokens", 0) or 0),
        output_tokens=int(resp.get("output_tokens", 0) or 0),
        finish_reason=str(resp.get("finish_reason", "") or ""),
        cost_usd=resp.get("cost_usd"),
    )


@dataclass
class _TrajectoryState:
    """Mutable per-(condition, question, replicate) execution state across rounds."""

    cond_id: str
    task: TaskItem
    rep: int
    current_logs: list[float]
    histories: list[list[tuple[str, str]]]
    round_idx: int = 1
    ctx: _RoundContext | None = None
    pending_outcomes: dict[int, tuple[str, Any]] = field(default_factory=dict)
    remaining_pending: int = 0
    records: list[NumericEstimateRecord] = field(default_factory=list)


def _cell_key(
    cond_id: str, question_id: str, rep: int, round_idx: int, agent_id: str
) -> tuple[str, str, int, int, str]:
    return (cond_id, question_id, rep, round_idx, agent_id)


def _build_round_context(
    backend: Any,
    agents: list[AgentProfile],
    traj: _TrajectoryState,
    tracker: _SpendTracker,
    max_tokens: int,
    pricing: TokenPricing | None,
    abort: threading.Event,
    current_date: str,
    models: tuple[str, ...] = (),
    routed_backends: dict[str, Any] | None = None,
) -> _RoundContext:
    sorted_x = sorted(10.0**lg for lg in traj.current_logs)
    return _RoundContext(
        backend=backend,
        agents=agents,
        cond_id=traj.cond_id,
        task=traj.task,
        round_idx=traj.round_idx,
        pop_stats=(
            statistics.median(sorted_x),
            quantile_sorted(sorted_x, 0.25),
            quantile_sorted(sorted_x, 0.75),
        ),
        current_logs=traj.current_logs,
        histories=traj.histories,
        tracker=tracker,
        max_tokens=max_tokens,
        pricing=pricing,
        abort=abort,
        current_date=current_date,
        models=models,
        routed_backends=routed_backends or {},
    )


def _apply_round_outcomes(
    ctx: _RoundContext,
    spec: dict[str, Any],
    rep: int,
    alpha: float,
    outcomes: dict[int, tuple[str, Any]],
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord],
    fh: Any,
) -> list[NumericEstimateRecord]:
    """Apply existing and newly fetched outcomes in deterministic agent order."""
    round_records: list[NumericEstimateRecord] = []
    first_error: Exception | None = None
    unit_str = f" {ctx.task.unit}" if ctx.task.unit else ""
    for a_idx in range(len(ctx.agents)):
        key = _cell_key(ctx.cond_id, ctx.task.id, rep, ctx.round_idx, ctx.agents[a_idx].agent_id)
        if key in existing:
            rec = existing[key]
            est_hist = format_prompt_number(10.0**rec.agent_log10_before)
            ctx.histories[a_idx].append(
                (f"My current estimate is {est_hist}{unit_str}.", rec.raw_reply)
            )
            ctx.current_logs[a_idx] = rec.agent_log10_after
            round_records.append(rec)
            continue
        status, payload = outcomes[a_idx]
        if status == "error":
            first_error = first_error or payload
            continue
        if status == "skipped":
            continue
        resp, parsed = payload
        cur_x = 10.0 ** ctx.current_logs[a_idx]
        rec = _make_record(spec, ctx, rep, a_idx, resp, parsed, alpha)
        ctx.current_logs[a_idx] = rec.agent_log10_after
        ctx.histories[a_idx].append(
            (f"My current estimate is {format_prompt_number(cur_x)}{unit_str}.", rec.raw_reply)
        )
        existing[key] = rec
        round_records.append(rec)
        fh.write(rec.model_dump_json() + "\n")
        fh.flush()

    # Calls that finished are saved above (they were paid for and keep the run resumable);
    # only then surface the first failure so the run stops cleanly.
    if first_error is not None:
        raise first_error
    return round_records


def _run_round(
    ctx: _RoundContext,
    spec: dict[str, Any],
    rep: int,
    alpha: float,
    workers: int,
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord],
    fh: Any,
) -> list[NumericEstimateRecord]:
    """Run one round sequentially or concurrently across pending agents."""
    pending = [
        i
        for i in range(len(ctx.agents))
        if _cell_key(ctx.cond_id, ctx.task.id, rep, ctx.round_idx, ctx.agents[i].agent_id)
        not in existing
    ]
    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda i: _query_agent(ctx, i), pending))
    else:
        results = [_query_agent(ctx, i) for i in pending]
    outcomes = dict(zip(pending, results, strict=True))
    return _apply_round_outcomes(ctx, spec, rep, alpha, outcomes, existing, fh)


def _validate_numeric_inputs(
    tasks: list[TaskItem],
    conditions: list[str],
    n_agents: int,
    n_rounds: int,
    replicates: int,
    backend: Any,
    alpha: float,
    budget_usd: float,
    workers: int,
    models: list[str] | None = None,
) -> list[str]:
    if workers < 1:
        raise BackendConfigError("workers must be at least 1.")
    if not hasattr(backend, "generate_numeric_reply"):
        raise BackendConfigError(
            "This backend does not support numeric replay; use a numeric-capable backend."
        )
    if not 0 <= alpha <= 1 or not math.isfinite(budget_usd) or budget_usd < 0:
        raise BackendConfigError("Invalid alpha or budget.")
    if (
        not tasks
        or len({t.id for t in tasks}) != len(tasks)
        or any(t.numeric_truth is None for t in tasks)
    ):
        raise BackendConfigError("Numeric tasks need unique IDs and numeric truth.")
    if (
        min(n_agents, n_rounds, replicates) < 1
        or not conditions
        or len(set(conditions)) != len(conditions)
    ):
        raise BackendConfigError("Numeric dimensions must be positive and conditions unique.")
    if any(c.upper() not in _VALID_NUMERIC_CONDITIONS for c in conditions):
        raise BackendConfigError(
            f"Numeric conditions must be drawn from {sorted(_VALID_NUMERIC_CONDITIONS)}."
        )
    normalized_models = [m.strip() if isinstance(m, str) else "" for m in (models or [])]
    if any(not m for m in normalized_models):
        raise BackendConfigError("Numeric models must be non-empty strings.")
    if any(c.upper() in _POLYPHONY_CONDITIONS for c in conditions):
        if len(set(normalized_models)) < 2:
            raise BackendConfigError(
                "Conditions C5, C6, P3, P4, and P0_POLY require at least 2 unique non-empty models."
            )
        if not hasattr(backend, "for_model"):
            raise BackendConfigError("Selected backend does not support multi-model routing.")
    constituents = [c.upper() for c in conditions if c.upper() in _CONSTITUENT_MODEL_TOKENS]
    if constituents:
        if not hasattr(backend, "for_model"):
            raise BackendConfigError("Selected backend does not support multi-model routing.")
        for c_upper in constituents:
            if _resolve_condition_model(c_upper, 0, normalized_models) is None:
                raise BackendConfigError(
                    f"Condition {c_upper} requires matching model in numeric_models."
                )
    return normalized_models


def _build_trajectories(
    tasks: list[TaskItem],
    conditions: list[str],
    n_agents: int,
    replicates: int,
    seed: int,
    start_bias_weight: float,
    start_sd: float,
) -> list[_TrajectoryState]:
    trajs: list[_TrajectoryState] = []
    for cond_id in conditions:
        for task in tasks:
            truth = task.numeric_truth or 100.0
            log_truth = math.log10(truth)
            for rep in range(replicates):
                init_rng = random.Random(seed * 313 + sum(ord(c) for c in task.id) * 19 + rep * 7)
                current_logs = [
                    log_truth
                    + start_bias_weight * task.popular_misconception_bias
                    + init_rng.gauss(0.0, start_sd)
                    for _ in range(n_agents)
                ]
                trajs.append(
                    _TrajectoryState(
                        cond_id=cond_id,
                        task=task,
                        rep=rep,
                        current_logs=current_logs,
                        histories=[[] for _ in range(n_agents)],
                    )
                )
    return trajs


def _run_trajectories_parallel(
    trajs: list[_TrajectoryState],
    backend: Any,
    agents: list[AgentProfile],
    spec: dict[str, Any],
    n_rounds: int,
    alpha: float,
    workers: int,
    tracker: _SpendTracker,
    max_tokens: int,
    pricing: TokenPricing | None,
    current_date: str,
    existing: dict[tuple[str, str, int, int, str], NumericEstimateRecord],
    fh: Any,
    models: tuple[str, ...] = (),
    routed_backends: dict[str, Any] | None = None,
) -> None:
    """Execute independent trajectories concurrently via one shared worker pool."""
    run_abort = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight: dict[Future[tuple[str, Any]], tuple[int, int]] = {}

        def enqueue_next(traj_idx: int) -> None:
            traj = trajs[traj_idx]
            while traj.round_idx <= n_rounds:
                ctx = _build_round_context(
                    backend,
                    agents,
                    traj,
                    tracker,
                    max_tokens,
                    pricing,
                    run_abort,
                    current_date,
                    models,
                    routed_backends,
                )
                pending = [
                    i
                    for i in range(len(agents))
                    if _cell_key(
                        traj.cond_id, traj.task.id, traj.rep, traj.round_idx, agents[i].agent_id
                    )
                    not in existing
                ]
                if not pending:
                    traj.records.extend(
                        _apply_round_outcomes(ctx, spec, traj.rep, alpha, {}, existing, fh)
                    )
                    traj.round_idx += 1
                    continue
                traj.ctx = ctx
                traj.pending_outcomes = {}
                traj.remaining_pending = len(pending)
                for a_idx in pending:
                    in_flight[pool.submit(_query_agent, ctx, a_idx)] = (traj_idx, a_idx)
                break

        for t_idx in range(len(trajs)):
            enqueue_next(t_idx)

        first_error: Exception | None = None
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                traj_idx, a_idx = in_flight.pop(fut)
                traj = trajs[traj_idx]
                traj.pending_outcomes[a_idx] = (
                    ("skipped", None) if fut.cancelled() else fut.result()
                )
                traj.remaining_pending -= 1
                if traj.remaining_pending != 0 or traj.ctx is None:
                    continue
                try:
                    round_recs = _apply_round_outcomes(
                        traj.ctx, spec, traj.rep, alpha, traj.pending_outcomes, existing, fh
                    )
                except Exception as exc:
                    first_error = first_error or exc
                    run_abort.set()
                    traj.ctx = None
                    continue
                traj.records.extend(round_recs)
                traj.round_idx += 1
                traj.ctx = None
                if not run_abort.is_set():
                    enqueue_next(traj_idx)
        if first_error is not None:
            raise first_error


def run_numeric_core(
    tasks: list[TaskItem],
    conditions: list[str],
    n_agents: int,
    n_rounds: int,
    replicates: int,
    runs_path: Path,
    backend: Any,
    alpha: float = 0.3,
    budget_usd: float = 50.0,
    seed: int = 42,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
    workers: int = 1,
    start_bias_weight: float = 0.22,
    start_sd: float = 0.24,
    current_date: str = "",
    models: list[str] | None = None,
) -> list[NumericEstimateRecord]:
    """Run crash-resumable C1..C6 numeric estimation sweeps with retry-once and budget cap.

    Independent trajectories and their within-round agents share `workers > 1` concurrently.
    Results are applied and returned in canonical `(condition, question, replicate, round, agent)`
    order so outputs match sequential execution.
    """
    normalized_models = _validate_numeric_inputs(
        tasks,
        conditions,
        n_agents,
        n_rounds,
        replicates,
        backend,
        alpha,
        budget_usd,
        workers,
        models=models,
    )
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    spec: dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "alpha": alpha,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "replicates": replicates,
        "max_tokens": max_tokens,
        "start_bias_weight": start_bias_weight,
        "start_sd": start_sd,
        "current_date": current_date,
        "conditions": conditions,
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "backend": backend_identity(backend),
    }
    if normalized_models:
        spec["numeric_models"] = list(normalized_models)
    validate_manifest(runs_path.with_suffix(".manifest.json"), spec, runs_path)
    ledger = getattr(backend, "budget_ledger", None)
    configure_budget(backend, runs_path.parent, budget_usd, ledger.per_request if ledger else 1.0)
    needs_routing = any(
        c.upper() in _POLYPHONY_CONDITIONS or c.upper() in _CONSTITUENT_MODEL_TOKENS
        for c in conditions
    )
    routed_backends = (
        {m: backend.for_model(m) for m in dict.fromkeys(normalized_models)} if needs_routing else {}
    )
    model_tuple = tuple(normalized_models)
    agents = generate_agent_population(num_agents=n_agents, seed=seed)
    existing = _load_existing_numeric_records(runs_path)

    tracker = _SpendTracker(
        budget_usd, sum(_estimate_call_cost_usd(r.model_dump(), pricing) for r in existing.values())
    )
    if any(r.run_spec != spec for r in existing.values()):
        raise RuntimeError("Record run specification differs from manifest.")
    trajs = _build_trajectories(
        tasks, conditions, n_agents, replicates, seed, start_bias_weight, start_sd
    )

    with runs_path.open("a", encoding="utf-8") as fh:
        if runs_path.stat().st_size and not runs_path.read_bytes().endswith(b"\n"):
            fh.write("\n")
        if workers <= n_agents:
            for traj in trajs:
                while traj.round_idx <= n_rounds:
                    ctx = _build_round_context(
                        backend,
                        agents,
                        traj,
                        tracker,
                        max_tokens,
                        pricing,
                        threading.Event(),
                        current_date,
                        model_tuple,
                        routed_backends,
                    )
                    traj.records.extend(
                        _run_round(ctx, spec, traj.rep, alpha, workers, existing, fh)
                    )
                    traj.round_idx += 1
        else:
            _run_trajectories_parallel(
                trajs,
                backend,
                agents,
                spec,
                n_rounds,
                alpha,
                workers,
                tracker,
                max_tokens,
                pricing,
                current_date,
                existing,
                fh,
                model_tuple,
                routed_backends,
            )

    return [rec for traj in trajs for rec in traj.records]


def _mean_metric(cells: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(m[key]) for m in cells if m.get(key) is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _summarize_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: _mean_metric(cells, key) for key in _ROUND_METRIC_KEYS}


def _compute_cell_metrics(
    by_cell: dict[tuple[str, str, int, int], list[NumericEstimateRecord]],
    truth_by_qid: dict[str, float],
    alpha: float,
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], float]:
    d0_by_traj: dict[tuple[str, str, int], float] = {}
    for (cond_id, qid, rep, r_idx), cell_recs in by_cell.items():
        if r_idx == 1:
            d0_by_traj[(cond_id, qid, rep)] = compute_page_decomposition(
                [r.agent_log10_before for r in cell_recs],
                truth_by_qid[qid],
            )["diversity"]

    cell_metrics: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    max_resid = 0.0
    for key, cell_recs in by_cell.items():
        cond_id, qid, rep, r_idx = key
        truth = truth_by_qid[qid]
        decomp: dict[str, Any] = compute_hierarchical_page_decomposition(
            [r.agent_log10_after for r in cell_recs],
            [r.model_slug for r in cell_recs],
            log_truth=truth,
        )
        decomp["miss_rate"] = sum(1 for r in cell_recs if r.miss) / len(cell_recs)
        d_prev = compute_page_decomposition(
            [r.agent_log10_before for r in cell_recs],
            truth,
        )["diversity"]
        denom = ((1.0 - alpha) ** 2) * d_prev
        if denom <= 1e-15:
            decomp["evrr"] = None
            decomp["evrr_between"] = None
            decomp["evrr_within"] = None
        else:
            decomp["evrr"] = decomp["diversity"] / denom
            decomp["evrr_between"] = decomp["between_model_diversity"] / denom
            decomp["evrr_within"] = decomp["within_model_diversity"] / denom

        d_0 = d0_by_traj.get((cond_id, qid, rep))
        cum_denom = ((1.0 - alpha) ** (2 * r_idx)) * d_0 if d_0 is not None else 0.0
        decomp["evrr_cumulative"] = (
            None if d_0 is None or cum_denom <= 1e-15 else decomp["diversity"] / cum_denom
        )
        max_resid = max(
            max_resid,
            decomp["identity_residual"],
            decomp["hierarchical_identity_residual"],
        )
        asst_pairs = [
            (float(r.assistant_log10), r.model_slug)
            for r in cell_recs
            if not r.miss and r.assistant_log10 is not None
        ]
        if asst_pairs:
            asst_decomp = compute_hierarchical_page_decomposition(
                [v for v, _ in asst_pairs],
                [m for _, m in asst_pairs],
                log_truth=truth,
            )
            decomp["assistant_crowd_error"] = asst_decomp["crowd_error"]
            decomp["assistant_avg_individual_error"] = asst_decomp["avg_individual_error"]
            decomp["assistant_diversity"] = asst_decomp["diversity"]
            decomp["assistant_between_model_diversity"] = asst_decomp["between_model_diversity"]
            decomp["assistant_within_model_diversity"] = asst_decomp["within_model_diversity"]
            decomp["assistant_excess_between_model_diversity"] = asst_decomp[
                "excess_between_model_diversity"
            ]
            decomp["assistant_between_share"] = asst_decomp["between_share"]
            max_resid = max(
                max_resid,
                asst_decomp["identity_residual"],
                asst_decomp["hierarchical_identity_residual"],
            )
        else:
            for key_name in (
                "assistant_crowd_error",
                "assistant_avg_individual_error",
                "assistant_diversity",
                "assistant_between_model_diversity",
                "assistant_within_model_diversity",
                "assistant_excess_between_model_diversity",
                "assistant_between_share",
            ):
                decomp[key_name] = None
        cell_metrics[key] = decomp
    return cell_metrics, max_resid


def _compute_interaction_2x2(
    cell_metrics: dict[tuple[str, str, int, int], dict[str, Any]],
    max_round: int,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {}
    present = {c for (c, _, _, r) in cell_metrics if r == max_round}
    if {"P1", "P2", "P3", "P4"}.issubset(present):
        c00, c01, c10, c11 = "P1", "P2", "P3", "P4"
    elif {"C2", "C4", "C5", "C6"}.issubset(present):
        c00, c01, c10, c11 = "C2", "C4", "C5", "C6"
    else:
        return {}

    metrics: tuple[str, ...] = (
        ("crowd_error", "assistant_crowd_error", "avg_individual_error", "diversity", "evrr")
        if c00 == "P1"
        else ("crowd_error", "avg_individual_error", "diversity", "evrr")
    )
    raw: list[tuple[str, list[float], float, float, float]] = []
    for metric in metrics:
        per_cond = {
            cond: {
                (qid, rep): float(val)
                for (c, qid, rep, r), m in cell_metrics.items()
                if c == cond and r == max_round and (val := m.get(metric)) is not None
            }
            for cond in (c00, c01, c10, c11)
        }
        common = sorted(
            set(per_cond[c00]) & set(per_cond[c01]) & set(per_cond[c10]) & set(per_cond[c11])
        )
        if not common:
            continue
        dids = [
            (per_cond[c11][k] - per_cond[c10][k]) - (per_cond[c01][k] - per_cond[c00][k])
            for k in common
        ]
        ci_lo, ci_hi = bootstrap_ci(dids, seed=42)
        p_val = paired_wilcoxon_p([0.0] * len(dids), dids)
        raw.append((metric, dids, ci_lo, ci_hi, p_val))

    if not raw:
        return {}
    adj_ps = holm_bonferroni([item[4] for item in raw])
    return {
        metric: {
            "mean_interaction": round(sum(dids) / len(dids), 6),
            "ci_low": round(ci_lo, 6),
            "ci_high": round(ci_hi, 6),
            "wilcoxon_p": round(p_val, 6),
            "holm_adjusted_p": round(adj_p, 6),
            "n_cells": len(dids),
        }
        for (metric, dids, ci_lo, ci_hi, p_val), adj_p in zip(raw, adj_ps, strict=True)
    }


_CONTROL_CONTRAST_SPECS = [
    ("p3_vs_p0_poly_crowd_error", "crowd_error", "P0_POLY", "P3"),
    ("p4_vs_p0_poly_crowd_error", "crowd_error", "P0_POLY", "P4"),
    ("p2_vs_p1_reflect_crowd_error", "crowd_error", "P1_REFLECT", "P2"),
    ("p1_reflect_vs_p1_crowd_error", "crowd_error", "P1", "P1_REFLECT"),
    ("p1_ds_vs_p0_ds_crowd_error", "crowd_error", "P0_DS", "P1_DS"),
    ("p1_qwen_vs_p0_qwen_crowd_error", "crowd_error", "P0_QWEN", "P1_QWEN"),
    ("p1_glm_vs_p0_glm_crowd_error", "crowd_error", "P0_GLM", "P1_GLM"),
    ("p3_vs_pooled_k1_social_crowd_error", "crowd_error", "POOLED_K1_SOCIAL", "P3"),
]


def _compute_polyphony_contrast_matrix(
    by_condition_final: dict[str, dict[str, Any]],
    cell_metric_fn: Any,
    complete: bool,
) -> dict[str, list[Any]]:
    core_p = {"P0", "P1", "P2", "P3", "P4"}
    if not complete or not any(c.upper() in core_p for c in by_condition_final):
        return {}
    return {
        metric: compute_paired_holm_contrasts(
            [(f"{b.lower()}_vs_{a.lower()}_{metric}", metric, a, b) for a, b in _P_MATRIX_PAIRS],
            cell_metric_fn,
            seed=42,
            effect_decimals=6,
            include_empty=False,
        )
        for metric in _P_MATRIX_METRICS
    }


def _compute_control_contrasts(
    by_condition_final: dict[str, dict[str, Any]],
    cell_metric_fn: Any,
    complete: bool,
) -> list[Any]:
    if not complete or not any(c.upper() in _P_CONTROL_CONDITIONS for c in by_condition_final):
        return []
    k1_social = ("P1", "P1_DS", "P1_QWEN", "P1_GLM")

    def _control_metric_fn(cond_id: str, metric: str) -> dict[Any, float]:
        if cond_id == "POOLED_K1_SOCIAL":
            if not all(c in by_condition_final for c in k1_social):
                return {}
            maps = [cell_metric_fn(c, metric) for c in k1_social]
            common = sorted(set.intersection(*(set(m) for m in maps))) if maps else []
            return {k: sum(m[k] for m in maps) / len(maps) for k in common}
        return cell_metric_fn(cond_id, metric)

    return compute_paired_holm_contrasts(
        _CONTROL_CONTRAST_SPECS,
        _control_metric_fn,
        seed=42,
        effect_decimals=6,
        include_empty=False,
    )


def _stratum_report(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
    cell_metrics: dict[tuple[str, str, int, int], dict[str, Any]],
    max_round: int,
    complete: bool,
) -> dict[str, Any]:
    """Slopes and final-round metrics for each question stratum (e.g. high/low spread)."""
    by_stratum: dict[str, list[TaskItem]] = defaultdict(list)
    for t in tasks:
        if t.selection_stratum:
            by_stratum[t.selection_stratum].append(t)
    report: dict[str, Any] = {}
    for name, stratum_tasks in sorted(by_stratum.items()):
        qids = {t.id for t in stratum_tasks}
        final: dict[str, dict[str, Any]] = {}
        if complete:
            for cond in sorted({c for c, _, _, _ in cell_metrics}):
                cells = [
                    m
                    for (c, q, _rep, r), m in cell_metrics.items()
                    if c == cond and q in qids and r == max_round
                ]
                if cells:
                    final[cond] = {"n_cells": len(cells), **_summarize_cells(cells)}
        report[name] = {
            "questions": sorted(qids),
            "by_condition_final": final,
            "h4_slopes": compute_h4_slopes(
                [r for r in records if r.question_id in qids], stratum_tasks
            ),
        }
    return report


def _merge_compatible_run_specs(records: list[NumericEstimateRecord]) -> dict[str, Any]:
    """Return unified run_spec when records match on everything except condition subsets."""
    if not records:
        return {}
    raw_specs = [r.run_spec for r in records]
    if len({json.dumps(s, sort_keys=True) for s in raw_specs}) == 1:
        return raw_specs[0]
    base_json = {
        json.dumps(
            {k: v for k, v in s.items() if k not in {"conditions", "numeric_models"}},
            sort_keys=True,
        )
        for s in raw_specs
    }
    if len(base_json) != 1:
        return {}
    merged_conditions = list(dict.fromkeys(c for s in raw_specs for c in s.get("conditions", [])))
    return {**raw_specs[0], "conditions": merged_conditions}


def analyze_numeric_core(
    records: list[NumericEstimateRecord],
    tasks: list[TaskItem],
) -> dict[str, Any]:
    """Compute hierarchical Page decomposition, EVRR, H4 slopes, contrasts, and 2x2 DiD."""
    truth_by_qid = {t.id: math.log10(t.numeric_truth) for t in tasks if t.numeric_truth is not None}
    by_cell: dict[tuple[str, str, int, int], list[NumericEstimateRecord]] = defaultdict(list)
    for rec in records:
        by_cell[(rec.condition_id, rec.question_id, rec.replicate, rec.round_idx)].append(rec)

    spec = _merge_compatible_run_specs(records)
    expected_agents = (
        {a.agent_id for a in generate_agent_population(spec["n_agents"], seed=spec["seed"])}
        if spec
        else set()
    )
    expected_cells = {
        (c, t["id"], rep, rnd)
        for c in spec.get("conditions", [])
        for t in spec.get("tasks", [])
        for rep in range(spec.get("replicates", 0))
        for rnd in range(1, spec.get("n_rounds", 0) + 1)
    }
    complete_cells = {
        key
        for key, rows in by_cell.items()
        if expected_agents
        and len(rows) == len(expected_agents)
        and {r.agent_id for r in rows} == expected_agents
    }
    complete = (
        bool(expected_cells) and set(by_cell) == expected_cells and complete_cells == expected_cells
    )
    if spec and any(
        truth_by_qid.get(t["id"]) != math.log10(t["numeric_truth"]) for t in spec["tasks"]
    ):
        raise RuntimeError("Analysis ground truth differs from the saved run specification.")
    if any(r.question_id not in truth_by_qid for r in records):
        raise RuntimeError("Cannot analyze records without their question ground truth.")

    alpha = float(spec.get("alpha", 0.3))
    cell_metrics, max_resid = _compute_cell_metrics(by_cell, truth_by_qid, alpha)

    valid_recs = [r for r in records if not r.miss and r.assistant_log10 is not None]
    h4_corr = pearson_corr(
        [float(r.assistant_log10) for r in valid_recs if r.assistant_log10 is not None],
        [r.agent_log10_after for r in valid_recs],
    )

    max_round = spec.get("n_rounds", max((r.round_idx for r in records), default=1))
    by_cond_round: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (cond_id, _qid, _rep, r_idx), m in cell_metrics.items():
        by_cond_round[(cond_id, r_idx)].append(m)

    condition_round_metrics: list[dict[str, Any]] = [
        {"condition_id": cond_id, "round_idx": r_idx, **_summarize_cells(ms)}
        for (cond_id, r_idx), ms in sorted(by_cond_round.items())
    ]

    by_condition_final = {
        row["condition_id"]: row
        for row in condition_round_metrics
        if complete and row["round_idx"] == max_round
    }

    core_pairs = [
        ("c2_vs_c1_diversity", "diversity", "C1", "C2"),
        ("c2_vs_c1_crowd_error", "crowd_error", "C1", "C2"),
        ("c3_vs_c1_diversity", "diversity", "C1", "C3"),
        ("c3_vs_c1_crowd_error", "crowd_error", "C1", "C3"),
        ("c4_vs_c2_crowd_error", "crowd_error", "C2", "C4"),
        ("c5_vs_c2_crowd_error", "crowd_error", "C2", "C5"),
        ("c6_vs_c2_crowd_error", "crowd_error", "C2", "C6"),
        ("c6_vs_c1_crowd_error", "crowd_error", "C1", "C6"),
        ("c6_vs_c4_crowd_error", "crowd_error", "C4", "C6"),
        ("c6_vs_c5_crowd_error", "crowd_error", "C5", "C6"),
        ("c5_vs_c2_excess_between_diversity", "excess_between_model_diversity", "C2", "C5"),
        ("c6_vs_c2_excess_between_diversity", "excess_between_model_diversity", "C2", "C6"),
        ("c6_vs_c2_diversity", "diversity", "C2", "C6"),
        ("c6_vs_c2_evrr", "evrr", "C2", "C6"),
        # Standalone P0..P4 confirmatory contrasts (inactive when only C-series ran).
        ("p4_vs_p1_crowd_error", "crowd_error", "P1", "P4"),
        ("p3_vs_p1_crowd_error", "crowd_error", "P1", "P3"),
        ("p2_vs_p1_crowd_error", "crowd_error", "P1", "P2"),
        ("p4_vs_p3_crowd_error", "crowd_error", "P3", "P4"),
        ("p4_vs_p2_crowd_error", "crowd_error", "P2", "P4"),
        ("p4_vs_p0_crowd_error", "crowd_error", "P0", "P4"),
        ("p1_vs_p0_crowd_error", "crowd_error", "P0", "P1"),
        ("p3_vs_p0_crowd_error", "crowd_error", "P0", "P3"),
        ("p2_vs_p0_crowd_error", "crowd_error", "P0", "P2"),
        ("p4_vs_p1_diversity", "diversity", "P1", "P4"),
        ("p3_vs_p1_diversity", "diversity", "P1", "P3"),
        ("p2_vs_p1_diversity", "diversity", "P1", "P2"),
        ("p3_vs_p1_excess_between_model_diversity", "excess_between_model_diversity", "P1", "P3"),
        ("p4_vs_p1_excess_between_model_diversity", "excess_between_model_diversity", "P1", "P4"),
        ("p4_vs_p3_between_model_diversity", "between_model_diversity", "P3", "P4"),
        ("p4_vs_p1_evrr", "evrr", "P1", "P4"),
        ("p4_vs_p1_assistant_crowd_error", "assistant_crowd_error", "P1", "P4"),
        ("p3_vs_p1_assistant_crowd_error", "assistant_crowd_error", "P1", "P3"),
        ("p2_vs_p1_assistant_crowd_error", "assistant_crowd_error", "P1", "P2"),
        ("p4_vs_p3_assistant_crowd_error", "assistant_crowd_error", "P3", "P4"),
    ]

    def _cell_metric(cond_id: str, metric: str) -> dict[tuple[str, int], float]:
        return {
            (qid, rep): float(val)
            for (c, qid, rep, r_idx), m in cell_metrics.items()
            if complete
            and c == cond_id
            and r_idx == max_round
            and (val := m.get(metric)) is not None
        }

    contrasts = compute_paired_holm_contrasts(
        core_pairs,
        _cell_metric,
        seed=42,
        effect_decimals=6,
        include_empty=False,
    )

    control_contrasts = _compute_control_contrasts(by_condition_final, _cell_metric, complete)
    out: dict[str, Any] = {
        "completion_status": "complete" if complete else "incomplete_or_unverified",
        "missing_or_incomplete_cells": len(expected_cells - complete_cells),
        "h4_interpretation": (
            "Legacy pooled correlation, kept for the dashboard: mechanically coupled by the "
            "update rule and pooled across questions. Use h4_slopes instead."
        ),
        "h4_slopes": compute_h4_slopes(records, tasks),
        "by_stratum": _stratum_report(records, tasks, cell_metrics, max_round, complete),
        "max_identity_residual": max_resid,
        "h4_assistant_agent_corr": round(h4_corr, 4),
        "by_condition_final": by_condition_final,
        "condition_round_metrics": condition_round_metrics,
        "interaction_2x2": _compute_interaction_2x2(cell_metrics, max_round, complete),
        "contrasts": contrasts,
        "polyphony_contrast_matrix": _compute_polyphony_contrast_matrix(
            by_condition_final, _cell_metric, complete
        ),
    }
    if control_contrasts:
        out["control_contrasts"] = control_contrasts
    return out
