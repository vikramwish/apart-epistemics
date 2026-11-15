"""Tests for SVG figure generation and self-contained editorial static dashboard builder."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from apart_epistemics.report import build_static_site, generate_figures, generate_polyphony_figures


def test_generate_figures_produces_valid_svgs(
    tmp_path: Path, mini_demo_bundle: tuple[dict[str, Any], dict[str, Any], list[Any]]
) -> None:
    suite_res, num_analysis, _ = mini_demo_bundle
    fig_dir = tmp_path / "figures"
    paths = generate_figures(suite_res, num_analysis, fig_dir)

    expected = {
        "dissociation_decoding.svg",
        "epistemic_quadrant.svg",
        "round_trajectories.svg",
        "pages_headline.svg",
    }
    assert {p.name for p in paths} == expected
    for svg_path in paths:
        raw = svg_path.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "undefined" not in raw
        root = ET.fromstring(raw)
        assert root.tag.endswith("svg")
        assert root.attrib.get("viewBox")


def test_build_static_site_self_contained_and_safe(
    tmp_path: Path, mini_demo_bundle: tuple[dict[str, Any], dict[str, Any], list[Any]]
) -> None:
    suite_res, num_analysis, num_recs = mini_demo_bundle
    # Inject XSS-like sequence into one record to ensure safe JSON embedding.
    suite_res["records"][0].agent_response_text += " </script><script>window.__xss=1</script>"

    fig_dir = tmp_path / "figures"
    generate_figures(suite_res, num_analysis, fig_dir)
    site_dir = tmp_path / "site"
    index_path = build_static_site(
        suite_result=suite_res,
        numeric_analysis=num_analysis,
        site_dir=site_dir,
        figures_dir=fig_dir,
        numeric_records=num_recs,
    )

    assert index_path.exists()
    html = index_path.read_text(encoding="utf-8")

    # Zero external CDNs
    assert "https://cdn." not in html
    assert "fonts.googleapis" not in html
    assert '<script src="http' not in html
    assert "window.__xss=1" not in html or "\\u003c/script\\u003e" in html

    for section in (
        "masthead-banner",
        "headline-findings",
        "epistemic-quadrant",
        "decoding-dissociation",
        "round-trajectory-explorer",
        "condition-score-table",
        "trace-specimen-drawer",
    ):
        assert f'data-section="{section}"' in html

    m = re.search(
        r'<script id="epistemic-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert m is not None
    embedded = json.loads(m.group(1))
    assert len(embedded["summaries"]) == 6
    assert embedded["numeric_analysis"]["max_identity_residual"] < 1e-11
    assert len(embedded["trace_specimens"]) > 0
    assert (site_dir / "figures" / "pages_headline.svg").exists()
    assert (site_dir / "data" / "benchmark_summary.json").exists()

    # The results view should explain its scales and expose readable navigation.
    for anchor in ("compare", "decoding", "rounds", "scores"):
        assert f'href="#{anchor}"' in html
        assert f'id="{anchor}"' in html
    assert 'aria-label="Results sections"' in html
    assert 'aria-label="Numeric chart legend"' in html
    assert "Lower Brier is better" in html
    assert "Synthetic runs demonstrate simulator assumptions" in html
    assert 'tabindex="0" role="region" aria-label="Condition scores' in html
    assert "prefers-reduced-motion" in html


def test_generate_polyphony_figures_produces_valid_svgs(tmp_path: Path) -> None:
    from apart_epistemics.backends import SyntheticEpistemicBackend
    from apart_epistemics.cli import _serialize_numeric_analysis
    from apart_epistemics.datasets import build_numeric_core_tasks
    from apart_epistemics.engine import (
        analyze_numeric_core,
        run_numeric_core,
    )

    tasks = build_numeric_core_tasks()[:2]
    models = ["gemma-4-26b-a4b-it", "deepseek-v4-flash", "qwen3.8-27b", "glm-5.2"]
    recs = run_numeric_core(
        tasks=tasks,
        conditions=["P0", "P1", "P2", "P3", "P4"],
        n_agents=4,
        n_rounds=2,
        replicates=1,
        runs_path=tmp_path / "poly_runs.jsonl",
        backend=SyntheticEpistemicBackend(seed=19),
        alpha=0.4,
        models=models,
    )
    analysis = analyze_numeric_core(recs, tasks)
    serialized = json.loads(json.dumps(_serialize_numeric_analysis(analysis), allow_nan=False))

    for sub_name, payload in (("raw", analysis), ("json", serialized)):
        fig_dir = tmp_path / f"figs_{sub_name}"
        paths = generate_polyphony_figures(payload, fig_dir)
        assert {p.name for p in paths} == {
            "polyphony_crowd_error.svg",
            "polyphony_hierarchical_page.svg",
            "polyphony_2x2_factorial.svg",
        }
        for svg_path in paths:
            raw = svg_path.read_text(encoding="utf-8")
            assert "NaN" not in raw
            assert "undefined" not in raw
            root = ET.fromstring(raw)
            assert root.tag.endswith("svg")
            assert root.attrib.get("viewBox")

    # Zero-EVRR fallback (alpha=1.0) still generates clean valid XML without NaN.
    z_recs = run_numeric_core(
        tasks=tasks,
        conditions=["P0", "P1", "P2", "P3", "P4"],
        n_agents=4,
        n_rounds=2,
        replicates=1,
        runs_path=tmp_path / "poly_zero.jsonl",
        backend=SyntheticEpistemicBackend(seed=23),
        alpha=1.0,
        models=models,
    )
    z_analysis = analyze_numeric_core(z_recs, tasks)
    z_paths = generate_polyphony_figures(z_analysis, tmp_path / "figs_zero")
    for svg_path in z_paths:
        raw = svg_path.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "undefined" not in raw
        ET.fromstring(raw)
    assert "Metric unavailable" in (
        tmp_path / "figs_zero" / "polyphony_2x2_factorial.svg"
    ).read_text(encoding="utf-8")
