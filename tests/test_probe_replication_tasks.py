"""Unit tests for Step 5 cold-probe filtering and resumable runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

from apart_epistemics.backends.providers import AdaptionBackend
from apart_epistemics.engine.numeric_core import p_series_system_prompt
from apart_epistemics.models import TaskItem, TaskKind

ROOT = Path(__file__).resolve().parents[1]


def _load_probe_module():
    path = ROOT / "scripts" / "probe_replication_tasks.py"
    spec = importlib.util.spec_from_file_location("probe_replication_tasks", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe_replication_tasks"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_task(task_id: str, truth: float = 1000.0) -> TaskItem:
    return TaskItem(
        id=task_id,
        kind=TaskKind.NUMERIC_ESTIMATE,
        domain="energy",
        question=f"What was the 2025 metric for {task_id}?",
        numeric_truth=truth,
        unit="units",
        source_note="Primary source 2025",
    )


def test_filter_rules_order_and_strata():
    mod = _load_probe_module()
    tasks = [
        _make_task("drop_parse", 1000.0),
        _make_task("drop_recall", 1000.0),
        _make_task("keep_control", 1000.0),
        _make_task("keep_low", 1000.0),
        _make_task("keep_high", 1000.0),
    ]
    models = mod.MODELS

    def rec(tid: str, model: str, val: float | None, err: float | None, miss: bool = False):
        return {
            "task_id": tid,
            "model_slug": model,
            "parsed_value": val,
            "log10_estimate": (3.0 + err) if err is not None else None,
            "log10_truth": 3.0,
            "log10_error": err,
            "abs_log10_error": abs(err) if err is not None else None,
            "miss": miss,
            "retries_used": 1 if miss else 0,
            "cost_usd": 0.0005,
        }

    records = {
        # Rule (a) fires even though other 3 models are within 0.0086
        ("drop_parse", models[0]): rec("drop_parse", models[0], 1001.0, 0.0004),
        ("drop_parse", models[1]): rec("drop_parse", models[1], 1001.0, 0.0004),
        ("drop_parse", models[2]): rec("drop_parse", models[2], 1001.0, 0.0004),
        ("drop_parse", models[3]): rec("drop_parse", models[3], None, None, miss=True),
        # Rule (b): all 4 models strictly below 0.0086 -> drop
        ("drop_recall", models[0]): rec("drop_recall", models[0], 1000.0, 0.002),
        ("drop_recall", models[1]): rec("drop_recall", models[1], 1000.0, -0.005),
        ("drop_recall", models[2]): rec("drop_recall", models[2], 1000.0, 0.0085),
        ("drop_recall", models[3]): rec("drop_recall", models[3], 1000.0, -0.001),
        # Control: max |err| >= 0.0086 so survives, but variance D_cold < 0.000065
        # est offsets: [0.0087, 0.0087, 0.0087, 0.0087] -> D_cold = 0.0
        ("keep_control", models[0]): rec("keep_control", models[0], 1020.2, 0.0087),
        ("keep_control", models[1]): rec("keep_control", models[1], 1020.2, 0.0087),
        ("keep_control", models[2]): rec("keep_control", models[2], 1020.2, 0.0087),
        ("keep_control", models[3]): rec("keep_control", models[3], 1020.2, 0.0087),
        # Low spread: offsets [-0.02, 0.02, -0.02, 0.02] -> D_cold = 0.0004 in [6.5e-5, 1.95e-3)
        ("keep_low", models[0]): rec("keep_low", models[0], 955.0, -0.02),
        ("keep_low", models[1]): rec("keep_low", models[1], 1047.0, 0.02),
        ("keep_low", models[2]): rec("keep_low", models[2], 955.0, -0.02),
        ("keep_low", models[3]): rec("keep_low", models[3], 1047.0, 0.02),
        # High spread: offsets [-0.1, 0.1, -0.1, 0.1] -> D_cold = 0.01 >= 0.00195
        ("keep_high", models[0]): rec("keep_high", models[0], 794.3, -0.10),
        ("keep_high", models[1]): rec("keep_high", models[1], 1258.9, 0.10),
        ("keep_high", models[2]): rec("keep_high", models[2], 794.3, -0.10),
        ("keep_high", models[3]): rec("keep_high", models[3], 1258.9, 0.10),
    }

    surviving, summary, counts = mod.evaluate_and_filter(tasks, records, models)
    assert [t.id for t in surviving] == ["keep_control", "keep_low", "keep_high"]
    assert [t.selection_stratum for t in surviving] == ["control", "low_spread", "high_spread"]
    assert counts == {"control": 1, "low_spread": 1, "high_spread": 1}
    assert summary[0]["rule"] == "(a) parse_failure"
    assert summary[1]["rule"] == "(b) recall_within_2pct"


def test_probe_runner_retry_resumability_and_min_count_guard(tmp_path: Path):
    mod = _load_probe_module()
    tasks = [_make_task(f"t_{i:02d}", 1000.0) for i in range(26)]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps([t.model_dump(mode="json") for t in tasks]),
        encoding="utf-8",
    )

    lock = threading.Lock()
    calls: list[dict[str, Any]] = []
    retry_state = {"t_01_retry": 0}

    def transport(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        del url, headers
        with lock:
            calls.append(payload)
            prompt = str(payload["input"])
            model = str(payload["model"])
            # Force t_00 deepseek to miss both attempts (drops under rule (a))
            if "t_00" in prompt and model == "deepseek-v4-flash":
                txt = "I refuse to give a number."
            # Force t_01 gemma to miss first attempt and recover on single retry
            elif "t_01" in prompt and model == "gemma-4-26b-a4b-it":
                retry_state["t_01_retry"] += 1
                txt = (
                    "Need clarification."
                    if retry_state["t_01_retry"] == 1
                    else "Final estimate: 1200"
                )
            else:
                val = 1200 if model in ("gemma-4-26b-a4b-it", "qwen3.8-27b") else 900
                txt = f"Final estimate: {val}"
        return {
            "model": model,
            "status": "completed",
            "output_text": txt,
            "usage": {"input_tokens": 50, "output_tokens": 10, "cost": 0.0002},
        }

    backend = AdaptionBackend(
        model=mod.MODELS[0],
        api_key="test-key",
        min_output_tokens=8192,
        transport=transport,
    )
    probe_dir = tmp_path / "probe"
    filtered_out = tmp_path / "filtered.json"
    table_out = tmp_path / "errors.md"
    argv = [
        "--tasks",
        str(tasks_path),
        "--filtered-out",
        str(filtered_out),
        "--probe-dir",
        str(probe_dir),
        "--table-out",
        str(table_out),
        "--current-date",
        "2026-11-15",
        "--workers",
        "1",
        "--budget",
        "5.0",
        "--request-allowance",
        "0.015369",
    ]
    rc = mod.main(argv, base_backend=backend)
    assert rc == 0
    # 26 * 4 = 104 base calls + 2 format retries (t_00 deepseek + t_01 gemma)
    assert len(calls) == 106
    expected_sys = p_series_system_prompt("2026-11-15")
    assert all(c["instructions"] == expected_sys for c in calls)
    assert all(c["max_output_tokens"] >= 8192 for c in calls)
    assert all("reasoning" not in c for c in calls if c["model"] == "gemma-4-26b-a4b-it")
    assert all(
        c["reasoning"] == {"effort": "none"}
        for c in calls
        if c["model"] != "gemma-4-26b-a4b-it"
    )
    assert (probe_dir / "provider_requests.jsonl").exists()
    assert (probe_dir / "budget.json").exists()

    filtered = json.loads(filtered_out.read_text(encoding="utf-8"))
    assert len(filtered) == 25
    assert "t_00" not in {t["id"] for t in filtered}
    assert "t_01" in {t["id"] for t in filtered}

    # Re-running against same probe_dir must never re-sample completed cells
    rc2 = mod.main(argv, base_backend=backend)
    assert rc2 == 0
    assert len(calls) == 106

    # < 25 surviving returns stop code 2
    short_tasks_path = tmp_path / "short.json"
    short_tasks_path.write_text(
        json.dumps([t.model_dump(mode="json") for t in tasks[:24]]),
        encoding="utf-8",
    )
    short_argv = [*argv]
    short_argv[1] = str(short_tasks_path)
    short_argv[3] = str(tmp_path / "short_filtered.json")
    short_argv[5] = str(tmp_path / "short_probe")
    short_argv[7] = str(tmp_path / "short_errors.md")
    assert mod.main(short_argv, base_backend=backend) == 2
