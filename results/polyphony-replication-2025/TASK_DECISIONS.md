# Task Decisions — `polyphony-replication-2025` (Step 2 Primary-Source Verification)

Source candidate file: `~/Downloads/polyphony_replication_tasks_2025.json` (`45` candidate tasks; `min(numeric_truth) = 119.33 >= 100`; zero ID overlap with the `16` tasks in `configs/polyphony_benchmark_tasks.json`).

## 1. Resolution of the 8 `"VERIFY"`-Flagged Tasks

| Task ID | Original Value | Primary Source Check | Decision | Reason & Details |
|---|---|---|---|---|
| `nrg_france_consumption_2025_twh` | `451.0` TWh | `rte-france.com` (*Bilan électrique 2025*) | **KEEP (disambiguated question)** | RTE directly reports weather- and calendar-corrected consumption in metropolitan France = `451 TWh` (`451.2 TWh`, `+0.4%` vs 2024) vs raw/gross `446.2 TWh`. Updated the question text as requested by `source_note` to specify `"weather- and calendar-corrected"`. |
| `nrg_us_coal_power_2025_twh` | `731.0` TWh | `eia.gov` (*Electric Power Monthly* / *Annual*) | **DROP** | Candidate value `731` came from EIA *Short-Term Energy Outlook* (early 2026). Primary EIA *Electric Power Monthly* utility-scale data for 2025 reports Total (All Sectors) = `734,928` thousand MWh (`734.928` TWh), Electric Utilities = `574,357` thousand MWh, Independent Power Producers = `156,102` thousand MWh (`730,459` thousand MWh sum before any CHP adjustment). Dropped rather than guessing or revising without the full Table 1.1.A PDF row. |
| `nrg_germany_public_net_2025_twh` | `440.0` TWh | `energy-charts.info` / `ise.fraunhofer.de` | **DROP** | Candidate `440` came from secondary `strom-report.com`. Primary `energy-charts.info` gives Total Public Load = `465.8 TWh` and per-source public net generation components that vary slightly between ENTSO-E and energetically corrected tracking (~`427`–`438` TWh), with no direct headline `440 TWh` figure verified on `energy-charts.info` / `ise.fraunhofer.de`. Dropped per Hard Rule 5. |
| `env_wmo_ch4_2025_ppb` | `1948.0` ppb | `wmo.int` (WMO Greenhouse Gas Bulletin No. 22 / State of the Climate Update) | **KEEP** | Directly confirmed on `wmo.int`: 2025 global mean surface $\text{CH}_4$ = `1,948` ($\pm 2$) ppb. |
| `env_wmo_n2o_2025_ppb` | `339.1` ppb | `wmo.int` (WMO Greenhouse Gas Bulletin No. 22 / State of the Climate Update) | **KEEP** | Directly confirmed on `wmo.int`: 2025 global mean surface $\text{N}_2\text{O}$ = `339.1` ($\pm 0.1$) ppb. |
| `cry_antarctic_max_2026_kkm2` | `17590.0` thousand km² | `nsidc.org` (Sea Ice Today, through Oct–Nov 2026) | **KEEP** | Directly confirmed on `nsidc.org`: Antarctic sea ice maximum extent on 14 Sept 2026 = `17.59 million km²` (`17,590` thousand km², 3rd lowest in satellite record), consistent in subsequent Oct–Nov 2026 NSIDC tracking. |
| `trn_suez_transits_2025` | `12758.0` ships | `suezcanal.gov.eg` (Suez Canal Authority annual statistics) | **KEEP** | Directly confirmed on `suezcanal.gov.eg`: `12,758` ships transited the Suez Canal in 2025 (`-3.4%`, down 455 from `13,213` in 2024). |
| `spc_satellites_deployed_2025` | `4517.0` satellites | `payloadspace.com` (*2025 Orbital Launch Attempts by Country*) | **KEEP** | Directly confirmed on `payloadspace.com`: `4,517` satellites deployed on orbit in 2025 (`+58%` vs 2024); the question explicitly names Payload's annual analysis. |

## 2. Summary Entering Step 5 Cold Probe

- **Total candidates:** `45`
- **Dropped in Step 2:** `2` (`nrg_us_coal_power_2025_twh`, `nrg_germany_public_net_2025_twh`)
- **Unflagged primary/audit-annotated tasks retained:** `37` (plus `6` kept `"VERIFY"` tasks = **`43` tasks retained** in `configs/polyphony_replication_tasks_2025.json`).
