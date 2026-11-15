#!/usr/bin/env python3
"""Merge and analyze P0-P4 baseline records with the three follow-up control arms.

Reads baseline ``results/polyphony-p0-p4/numeric_runs.jsonl`` (1,280 rows across P0..P4) and
a completed controls run from ``configs/polyphony_controls.yaml`` (2,048 rows across P0_POLY,
P1_REFLECT, P0_DS, P1_DS, P0_QWEN, P1_QWEN, P0_GLM, P1_GLM), runs ``analyze_numeric_core`` on
the union without modifying either source directory, and saves ``combined_numeric_analysis.json``.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/analyze_polyphony_controls.py \
        --controls results/polyphony-controls/numeric_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apart_epistemics.cli import _load_task_file, _serialize_numeric_analysis
from apart_epistemics.engine.numeric_core import analyze_numeric_core
from apart_epistemics.models import NumericEstimateRecord

ROOT = Path(__file__).resolve().parents[1]


def _load_records(path: Path) -> list[NumericEstimateRecord]:
    return [
        NumericEstimateRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "results" / "polyphony-p0-p4" / "numeric_runs.jsonl",
        help="Baseline P0..P4 numeric_runs.jsonl",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=ROOT / "results" / "polyphony-controls" / "numeric_runs.jsonl",
        help="Control arms numeric_runs.jsonl",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "configs" / "polyphony_benchmark_tasks.json",
        help="Benchmark tasks JSON file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <controls-dir>/combined_numeric_analysis.json)",
    )
    args = parser.parse_args()

    tasks = _load_task_file(args.tasks)
    base_records = _load_records(args.baseline)
    ctrl_records = _load_records(args.controls)
    analysis = analyze_numeric_core([*base_records, *ctrl_records], tasks)
    serialized = _serialize_numeric_analysis(analysis)

    out_path = args.out or (args.controls.parent / "combined_numeric_analysis.json")
    out_path.write_text(json.dumps(serialized, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Status: {serialized['completion_status']}")
    print(f"Saved combined analysis -> {out_path}")
    for c in serialized.get("control_contrasts", []):
        print(
            f"  {c['contrast_id']:38s} diff={c['mean_diff']:+.6f} "
            f"[{c['ci_low']:+.6f}, {c['ci_high']:+.6f}] "
            f"p={c['wilcoxon_p']:.6f} p_Holm={c['holm_adjusted_p']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
