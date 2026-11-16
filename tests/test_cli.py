"""CLI tests for `apart-epistemics run` dry-run validation, secrets, and pricing flags."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from apart_epistemics.cli import build_parser, main


@pytest.mark.parametrize(
    ("config_path", "expected_tasks", "expected_conditions"),
    [
        ("configs/polyphony_p0_p4.yaml", "16/16", "5"),
        ("configs/polyphony_controls.yaml", "16/16", "8"),
        ("configs/polyphony_replication_2025.yaml", "40/40", "13"),
    ],
)
def test_cli_run_dry_run_validates_all_study_configs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    config_path: str,
    expected_tasks: str,
    expected_conditions: str,
) -> None:
    rc = main(
        [
            "run",
            "--config",
            config_path,
            "--backend",
            "adaption",
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"tasks={expected_tasks}" in out
    assert f"conditions={expected_conditions}" in out


def test_cli_run_missing_secret_returns_clean_error_and_module_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out_dir = tmp_path / "run_out"
    rc = main(
        [
            "run",
            "--config",
            "configs/polyphony_p0_p4.yaml",
            "--backend",
            "openrouter",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 2

    proc = subprocess.run(
        [sys.executable, "-m", "apart_epistemics", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "run" in proc.stdout


def test_cli_run_pin_provider_requires_openrouter_and_max_tokens_is_parsed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    ns = build_parser().parse_args(["run", "--pin-provider", "OpenAI", "--max-tokens", "2048"])
    assert ns.pin_provider == "OpenAI"
    assert ns.max_tokens == 2048
    assert build_parser().parse_args(["run"]).max_tokens == 1024

    rc = main(
        [
            "run",
            "--config",
            "configs/polyphony_p0_p4.yaml",
            "--backend",
            "openai",
            "--pin-provider",
            "OpenAI",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 2
    assert "only supported with --backend openrouter" in capsys.readouterr().err


def test_cli_run_price_flags_default_and_parse() -> None:
    default = build_parser().parse_args(["run"])
    assert (default.price_in, default.price_out) == (0.5, 1.5)
    ns = build_parser().parse_args(["run", "--price-in", "0.1", "--price-out", "0.5"])
    assert (ns.price_in, ns.price_out) == (0.1, 0.5)
