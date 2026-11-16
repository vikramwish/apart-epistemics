# Mad in Herds

**AI Assistants Amplify a Biased Crowd** — Vikram Viswanath, University of Sydney,
with Apart Research. Kevin O’Shaughnessy contributed experiment-harness code.

[Read the final paper](reports/manuscript/main.pdf) ·
[Manuscript source and verification](reports/manuscript/README.md) ·
[Recorded evidence](results/README.md)

## Research

Eight simulated agents update numerical estimates using advice from LLM assistants.
A 16-task discovery study is followed by a 40-task, three-replicate study under a
committed analysis plan. Showing assistants a biased crowd’s median and IQR increased
final crowd error relative to withholding telemetry in all four model families tested
(24–115%). Structured critique helped, but recovered only about half of Gemma’s added
error. Mixing families did not eliminate the penalty.

The paper distinguishes exact error accounting from causal explanation. Agents are
scalar updaters; transfer to human users and AI judging remains untested. Study 3 is
unfinished and is not included in this submission.

## Reproduce offline

Python 3.11+ is required. These commands make no model calls:

```sh
make install       # package, test and manuscript dependencies
make check         # lint, tests and configuration dry runs
make paper-audit   # exact algebra and task-level statistical checks
make reproduce     # full Study 1 audit from retained records
make paper         # figures and final PDF; requires XeLaTeX + latexmk
```

`make paper-reanalyze` additionally requires the two local replication inputs described
in the [manuscript README](reports/manuscript/README.md). The large replication logs
are not in Git; their hashes are recorded in
[PROVENANCE.md](results/polyphony-replication-2025/PROVENANCE.md).
The included compact task data reproduce the new task-level sensitivity analyses.

## Contents

| Path | Purpose |
|---|---|
| `reports/manuscript/` | Final PDF, TeX, figures, fonts, compact data and mathematical/statistical checks |
| `results/` | Frozen discovery/control evidence, replication summaries and provenance |
| `configs/` | Task banks and experimental configurations |
| `docs/` | Committed analysis plans for controls and replication |
| `scripts/` | Original statistical analyses, audits and task-screening script |
| `src/` | Numeric experiment harness, parsers, metrics and provider adapters |
| `tests/` | Offline regression tests |
| `reports/DECISIONS.md` | Decision log cited by the paper’s resampling disclosure |

The superseded 12-task archive is retained only because the original audit reads it.
Older manuscripts, dashboards, exploratory experiments and Study 3 remain in historical
or development branches. There is one current manuscript in this submission.

## New experiments

Live runs cost money and are separate from reproduction. Export the appropriate
provider credentials using `.env.example` as a guide; `.env` is not loaded automatically.
Validate configuration first:

```sh
.venv/bin/apart-epistemics run --config configs/polyphony_p0_p4.yaml --dry-run
```

Remove `--dry-run` only when intentionally authorizing model calls. Output defaults to
ignored `tmp/`. Request allowances are a conservative budget guard, not actual billing.

## License

Code: [MIT](LICENSE). Bundled font notices are in
[reports/manuscript/fonts/NOTICES.txt](reports/manuscript/fonts/NOTICES.txt).
