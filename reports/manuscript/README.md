# Revised manuscript: Mad in Herds

This is the working paper incorporating the 16-task discovery study and the
40-task, three-replicate follow-up. It was imported from the author's
`paper_v4_source` bundle. Its TeX was byte-identical to the separately supplied
`APART_SUBMISSION_PAPER_v4 (1).tex`; the supplied PDF represented an older draft.
The original 16-task submission remains in the historical `study3-judge-panels` branch.

## Build

From the repository root, install the `paper` extra (`make install` also installs
it), then use:

```sh
make paper              # regenerate figures and build reports/manuscript/main.pdf
make paper-reanalyze    # recompute Study 2 supplemental JSON from local raw logs
make paper-audit        # exact math checks and statistical sensitivity
```

The revised manuscript requires XeLaTeX and latexmk. The bundled Old Standard
fonts preserve the source's typography; `fonts/NOTICES.txt` extracts their embedded
copyright and license metadata, and `fonts/README` identifies the upstream package.
NumPy, Matplotlib and SciPy are declared in the `paper` optional dependency group.
`PYTHON` can be overridden with an absolute interpreter path.

`make paper` works from the tracked Study 1 raw log and committed Study 2 summary
files. `make paper-reanalyze` additionally needs the large replication raw run log
and the cold-probe records under `results/polyphony-replication-2025/`. The main
raw log is excluded from Git; its SHA-256 is recorded in that directory's
`PROVENANCE.md`. The scripts also accept an explicit `.jsonl` or `.jsonl.gz` path;
`s2_extra.py` accepts the cold-probe JSONL as its second positional argument.
No model calls are made by these commands.

## Files and provenance

- `main.tex` / `main.pdf`: revised manuscript and compiled output.
- `IMPORT_PROVENANCE.json`: SHA-256 of every file in the supplied source bundle.
- `verify.py`: Study 1 plotting inputs reconstructed from full-precision raw data;
  replaces the imported rounded table and checks the error identity.
- `fig0.py`, `figs.py`, `fig_main_v3.py`, `fig_replication.py`: figure generators.
- `s2_analysis.py` / `s2_results.json`: independent Study 2 recomputation,
  10,000 task-bootstrap samples and 200,000 sign flips, with seed 42.
- `s2_extra.py` / `s2_extra.json`: 15 topic-domain cluster bootstrap and exploratory
  spread, starting-error and difficulty analyses.
- `VERIFICATION.json`: import parity and recorded-result comparisons from this revision.

The original analysis scripts in `scripts/` and all experiment logs remain
unchanged. Supplemental analyses are explicitly post hoc. Topic domains are the
benchmark's `domain` field, not independent source institutions. Parse-miss
sensitivity removes affected condition/task/replicate panels before averaging
remaining replicates within each condition; it does not resample answers.

## Editorial changes

- Restricted the headline and recommendations to the biased-consensus setting.
- Retained the supplied source's distinction between error accounting and causation,
  and its acknowledgment that two-sidedness has not been isolated.
- Corrected the request-setting description: temperature/top-p are omitted from
  requests, although response metadata can report resolved defaults.
- Replaced the total-compute claim with the measured output-token comparison.
- Distinguished Study 1's transformed log RMSE from Study 2's mean multiplicative
  deviation; neither is ordinary absolute percentage error for underestimates.
- Clarified that nonsignificance is not equivalence and that extra sensitivity
  analyses are post hoc; preserved all primary effect estimates.
- Corrected raw-data availability language and bounded wide tables to page width.

## Verification scope

The supplied Study 2 summaries were reproduced from the local raw logs to floating
point tolerance, including bootstrap intervals, permutation tests, parse-miss
sensitivity, and topic-domain analysis. Their point estimates and Wilcoxon p-values were compared with the repository's frozen preregistered output.
The frozen output rounds raw p-values before Holm correction; the supplemental
analysis adjusts full-precision values. These small differences change no significance
decisions. Study 1 figures now use full-precision retained observations. Input hashes and
reproduction tolerances are recorded in `VERIFICATION.json`.

The primary-source ground truths have not undergone a fresh external audit.
Selected bibliography records were checked against primary sources (SOURCE_CHECKS.md).
The supplied 20-dataset null-simulation summary has been removed because its exact
generator/seed record was absent. The repository's `scripts/synth_null_test.py` is
a separate fixture and is not treated as a reproduction of that result.

## Mathematical and statistical revision

Run `make paper-audit` with the normal NumPy-enabled environment. It requires no
API access and no large raw log. The following files are included:

- `audit_math.py` / `math_checks.json`: exact rational verification of the update
  identity, hierarchical decomposition, all 105 pairings on 36 quadratic
  coefficient vectors, weighted Jensen identity and a nonconvergent EVRR=1 example.
- `task_metrics.json`: full-precision task-level E2 for 13 arms and two virtual
  panels, with IDs, domains and the source file's SHA-256. Regenerated by
  `s2_analysis.py`; these are derived observations, not raw model transcripts.
- `audit_statistics.py` / `statistical_checks.json`: reproduces all 14 original
  point estimates and exact Wilcoxon p-values, then runs global Holm, leave-one-task
  and leave-one-domain sensitivity, and paired-bootstrap recovery intervals.
- `sensitivity_table.tex` / `recovery_summary.tex`: generated directly by the
  statistical audit and included in the PDF.
- `SOURCE_CHECKS.md`: bounded primary-source checks, not a full novelty audit.

The revision expands both propositions into complete proofs, states balanced-group
and matched-family-mean assumptions, derives EVRR and the cross-term accounting,
and distinguishes finite-panel identities from probabilistic claims. It removes the
unreproduced 20-dataset simulation figures; the exact algebra checks do **not**
replace a calibration experiment. The main replication figure now shows the
primary squared-log-error outcome and paired effect intervals.

All sensitivity analyses are post hoc. The compact dataset can reproduce task-level
statistics; checking extraction, parse handling and construction of virtual panels
still requires raw trajectories. No live experiments were run for this revision.
