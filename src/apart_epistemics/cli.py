"""CLI entrypoint for Epistemic Prism polyphony study runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from apart_epistemics.backends import (
    BackendConfigError,
    BudgetExceededError,
    create_backend,
)
from apart_epistemics.backends.base import DEFAULT_NUMERIC_MAX_TOKENS
from apart_epistemics.datasets import validate_candidates
from apart_epistemics.engine import (
    TokenPricing,
    analyze_numeric_core,
    run_numeric_core,
)
from apart_epistemics.models import (
    ExperimentSuiteConfig,
    NumericEstimateRecord,
    TaskItem,
)
from apart_epistemics.run_state import configure_budget


def _load_config(config_path: Path | None) -> ExperimentSuiteConfig:
    target = config_path if config_path is not None else Path("configs/polyphony_p0_p4.yaml")
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    return ExperimentSuiteConfig.model_validate(raw)


def _dump_contrast(c: Any) -> Any:
    return c.model_dump() if hasattr(c, "model_dump") else c


def _serialize_numeric_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    raw_matrix = analysis.get("polyphony_contrast_matrix", {})
    out: dict[str, Any] = {
        "completion_status": analysis.get("completion_status", "incomplete_or_unverified"),
        "missing_or_incomplete_cells": analysis.get("missing_or_incomplete_cells", 0),
        "max_identity_residual": analysis["max_identity_residual"],
        "h4_assistant_agent_corr": analysis["h4_assistant_agent_corr"],
        "h4_slopes": analysis["h4_slopes"],
        "by_stratum": analysis.get("by_stratum", {}),
        "by_condition_final": analysis["by_condition_final"],
        "condition_round_metrics": analysis["condition_round_metrics"],
        "interaction_2x2": analysis.get("interaction_2x2", {}),
        "contrasts": [_dump_contrast(c) for c in analysis["contrasts"]],
        "polyphony_contrast_matrix": {
            metric: [_dump_contrast(c) for c in items] for metric, items in raw_matrix.items()
        },
    }
    if analysis.get("control_contrasts"):
        out["control_contrasts"] = [_dump_contrast(c) for c in analysis["control_contrasts"]]
    return out


def _execute_numeric_track(
    cfg: ExperimentSuiteConfig,
    backend: Any,
    out_dir: Path,
    *,
    n_agents: int,
    budget_usd: float,
    max_tokens: int = DEFAULT_NUMERIC_MAX_TOKENS,
    pricing: TokenPricing | None = None,
    workers: int = 1,
    current_date: str = "",
) -> tuple[dict[str, Any], list[NumericEstimateRecord]]:
    if not cfg.numeric_tasks_file:
        raise BackendConfigError("Config must specify numeric_tasks_file.")
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = _load_task_file(Path(cfg.numeric_tasks_file))
    num_tasks = bank[: cfg.numeric_tasks] if cfg.numeric_tasks else bank
    print(
        f"Numeric track settings: {len(num_tasks)} of {len(bank)} questions, "
        f"{n_agents} agents, {cfg.num_rounds} rounds, {cfg.numeric_replicates} replicates, "
        f"start bias weight {cfg.numeric_start_bias_weight}, start sd {cfg.numeric_start_sd}, "
        f"conditions {cfg.numeric_conditions}, models {cfg.numeric_models}.",
        flush=True,
    )
    num_recs = run_numeric_core(
        tasks=num_tasks,
        conditions=cfg.numeric_conditions,
        n_agents=n_agents,
        n_rounds=cfg.num_rounds,
        replicates=cfg.numeric_replicates,
        runs_path=out_dir / "numeric_runs.jsonl",
        backend=backend,
        alpha=cfg.alpha,
        budget_usd=budget_usd,
        seed=cfg.seed,
        max_tokens=max_tokens,
        pricing=pricing,
        workers=workers,
        start_bias_weight=cfg.numeric_start_bias_weight,
        start_sd=cfg.numeric_start_sd,
        current_date=current_date,
        models=cfg.numeric_models,
    )
    num_analysis = analyze_numeric_core(num_recs, num_tasks)
    (out_dir / "numeric_analysis.json").write_text(
        json.dumps(_serialize_numeric_analysis(num_analysis), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return num_analysis, num_recs


def _resolve_current_date(arg: str | None, out_dir: Path | None = None) -> str:
    """The date told to the model: the flag, else the date saved by a run being resumed, else today.

    Reusing the saved date keeps a run that crosses midnight resumable: the date is part of the
    run manifest, so a different one would otherwise force a new output folder.
    """
    if arg:
        try:
            return date.fromisoformat(arg).isoformat()
        except ValueError:
            raise BackendConfigError(f"--current-date must be YYYY-MM-DD, got {arg!r}.") from None
    if out_dir is not None:
        manifest = out_dir / "numeric_runs.manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8")).get("current_date")
            if saved:
                return str(saved)
    return date.today().isoformat()


def _backend_from_args(args: argparse.Namespace, seed: int) -> Any:
    backend_kwargs: dict[str, Any] = {"seed": seed}
    if args.model:
        backend_kwargs["model"] = args.model
    if args.pin_provider:
        if args.backend != "openrouter":
            raise BackendConfigError("--pin-provider is only supported with --backend openrouter.")
        backend_kwargs["pin_provider"] = args.pin_provider
    if getattr(args, "dry_run", False) and args.backend != "ollama":
        backend_kwargs.setdefault("api_key", "dry-run-placeholder")
    return create_backend(args.backend, **backend_kwargs)


def _load_task_file(path: Path) -> list[TaskItem]:
    """Load saved numeric questions (e.g. selected_tasks.json); every field is validated."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = [TaskItem.model_validate(row) for row in raw]
    validate_candidates(tasks)
    return tasks


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(Path(args.config) if args.config else None)
        backend = _backend_from_args(args, seed=cfg.seed)
        backend.request_allowance = args.request_allowance
        if not cfg.numeric_tasks_file:
            raise BackendConfigError("Config must specify numeric_tasks_file.")
        tasks = _load_task_file(Path(cfg.numeric_tasks_file))
        selected = tasks[: cfg.numeric_tasks] if cfg.numeric_tasks else tasks
    except (BackendConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    try:
        current_date = _resolve_current_date(args.current_date, out_dir)
    except BackendConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("Configuration error: --workers must be at least 1.", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            f"Dry run ok: config={args.config}, backend={args.backend}, "
            f"tasks={len(selected)}/{len(tasks)}, conditions={len(cfg.numeric_conditions)}, "
            f"replicates={cfg.numeric_replicates}, current_date={current_date}."
        )
        return 0
    pricing = TokenPricing(input_per_m=args.price_in, output_per_m=args.price_out)
    try:
        configure_budget(backend, out_dir, args.budget, args.request_allowance)
        _execute_numeric_track(
            cfg,
            backend,
            out_dir,
            n_agents=cfg.num_agents,
            budget_usd=args.budget,
            max_tokens=args.max_tokens,
            pricing=pricing,
            workers=args.workers,
            current_date=current_date,
        )
    except (BudgetExceededError, RuntimeError, BackendConfigError) as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 2

    print(f"Run completed -> {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `apart-epistemics` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="apart-epistemics",
        description="Epistemic Prism CLI: numeric polyphony simulation runner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run configured polyphony numeric suite.")
    p_run.add_argument("--config", default="configs/polyphony_p0_p4.yaml")
    p_run.add_argument(
        "--backend",
        default="adaption",
        choices=["adaption", "openrouter", "openai", "anthropic", "gemini", "ollama"],
    )
    p_run.add_argument("--model", default=None)
    p_run.add_argument(
        "--pin-provider",
        default=None,
        help="OpenRouter only: route every call to this provider with no fallbacks.",
    )
    p_run.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_NUMERIC_MAX_TOKENS,
        help="Output token limit per numeric call (reasoning tokens may count against it).",
    )
    p_run.add_argument("--out", default="tmp/run")
    p_run.add_argument("--budget", type=float, default=15.0)
    p_run.add_argument(
        "--price-in",
        type=float,
        default=TokenPricing().input_per_m,
        help="USD per 1M input tokens, for the budget guard. Set this for your model.",
    )
    p_run.add_argument(
        "--price-out",
        type=float,
        default=TokenPricing().output_per_m,
        help="USD per 1M output tokens (reasoning tokens included), for the budget guard.",
    )
    p_run.add_argument(
        "--request-allowance",
        type=float,
        default=1.0,
        help=(
            "USD reserved before every HTTP attempt (default 1). Set at least "
            "the maximum possible request charge; not actual billing."
        ),
    )
    p_run.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Numeric track: agents queried concurrently within a round. Results are identical "
        "to --workers 1; only speed changes.",
    )
    p_run.add_argument(
        "--current-date",
        default=None,
        help="Today's date (YYYY-MM-DD) put in the shared system prompt. Default: today, or the "
        "date saved by the run being resumed.",
    )
    p_run.add_argument("--numeric-only", action="store_true")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports, config, task file, and backend wiring without calling any API.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
