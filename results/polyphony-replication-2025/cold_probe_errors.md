# Cold-Probe Results & Per-Model Errors (Step 5)

- **Candidates probed:** `43` tasks x `4` models (`172` cell records, `0` format retries).
- **Prompt:** `p_series_system_prompt("2026-11-15")` + question-only user prompt (no prior estimate, no telemetry), `min_output_tokens=8192`, `reasoning={"effort": "none"}` on DeepSeek/Qwen/GLM.
- **Total probe API cost:** `$0.086311`.
- **Dropped under rule (a) parse failure:** `0`
- **Dropped under rule (b) recall check (`|log10 err| < 0.0086` for all 4 models):** `3`
- **Surviving tasks (`>= 25` required):** **`40`**
- **Stratum counts (`D_cold` fixed midpoint cutoffs):** `control`: `5`, `low_spread`: `18`, `high_spread`: `17`.

| # | Task ID | Status | Stratum | `D_cold` | `E_cold` | Gemma err | DeepSeek err | Qwen err | GLM err | `max|err|` | Rule / Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `nrg_france_nuclear_2025_twh` | KEPT | low_spread | 0.000907 | 0.000442 | -0.04666 | -0.03389 | +0.03035 | -0.03389 | 0.04666 | assigned low_spread (D_cold=0.000907) |
| 2 | `nrg_france_total_production_2025_twh` | KEPT | low_spread | 0.001062 | 0.001479 | -0.09002 | -0.02238 | -0.03941 | -0.00199 | 0.09002 | assigned low_spread (D_cold=0.001062) |
| 3 | `nrg_france_consumption_2025_twh` | KEPT | low_spread | 0.000260 | 0.000054 | -0.00875 | +0.01792 | -0.02681 | -0.01171 | 0.02681 | assigned low_spread (D_cold=0.000260) |
| 4 | `nrg_us_wind_2025_twh` | KEPT | low_spread | 0.000953 | 0.000561 | +0.05776 | -0.00565 | +0.05115 | -0.00851 | 0.05776 | assigned low_spread (D_cold=0.000953) |
| 5 | `nrg_us_utility_solar_2025_twh` | KEPT | high_spread | 0.009530 | 0.045342 | -0.24598 | -0.35735 | -0.10956 | -0.13885 | 0.35735 | assigned high_spread (D_cold=0.009530) |
| 6 | `nrg_germany_wind_2025_twh` | KEPT | low_spread | 0.000459 | 0.003604 | +0.06127 | +0.06976 | +0.08355 | +0.02555 | 0.08355 | assigned low_spread (D_cold=0.000459) |
| 7 | `nrg_china_solar_added_2025_gw` | KEPT | high_spread | 0.002067 | 0.001679 | +0.00674 | -0.10924 | -0.00705 | -0.05436 | 0.10924 | assigned high_spread (D_cold=0.002067) |
| 8 | `nrg_china_wind_added_2025_gw` | KEPT | high_spread | 0.002395 | 0.020648 | -0.21942 | -0.14733 | -0.12251 | -0.08552 | 0.21942 | assigned high_spread (D_cold=0.002395) |
| 9 | `nrg_china_total_capacity_2025_gw` | KEPT | low_spread | 0.001226 | 0.000828 | -0.06490 | -0.01246 | +0.02071 | -0.05847 | 0.06490 | assigned low_spread (D_cold=0.001226) |
| 10 | `nrg_china_thermal_capacity_2025_gw` | KEPT | low_spread | 0.001075 | 0.003017 | -0.09910 | -0.01845 | -0.07330 | -0.02888 | 0.09910 | assigned low_spread (D_cold=0.001075) |
| 11 | `nrg_global_solar_gen_2025_twh` | KEPT | low_spread | 0.000838 | 0.000552 | +0.01111 | -0.00440 | -0.03719 | -0.06352 | 0.06352 | assigned low_spread (D_cold=0.000838) |
| 12 | `nrg_global_demand_2025_twh` | KEPT | low_spread | 0.000215 | 0.000001 | +0.01373 | -0.00938 | -0.01784 | +0.01637 | 0.01784 | assigned low_spread (D_cold=0.000215) |
| 13 | `nrg_global_wind_added_2025_gw` | KEPT | low_spread | 0.001350 | 0.004983 | -0.02715 | -0.11027 | -0.04139 | -0.10354 | 0.11027 | assigned low_spread (D_cold=0.001350) |
| 14 | `nrg_global_wind_cumulative_2025_gw` | KEPT | control | 0.000018 | 0.000114 | -0.01152 | -0.00471 | -0.01670 | -0.00981 | 0.01670 | assigned control (D_cold=0.000018) |
| 15 | `nrg_global_wind_turbines_2025` | KEPT | high_spread | 0.002890 | 0.001909 | -0.10106 | -0.09151 | -0.00608 | +0.02388 | 0.10106 | assigned high_spread (D_cold=0.002890) |
| 16 | `env_noaa_co2_2025_ppm` | DROPPED | - | 0.000001 | 0.000000 | +0.00090 | -0.00063 | -0.00094 | -0.00114 | 0.00114 | all 4 models |log10 err| < 0.0086 (max=0.001144) |
| 17 | `env_wmo_ch4_2025_ppb` | KEPT | control | 0.000034 | 0.000004 | +0.00045 | +0.00532 | -0.00246 | -0.01084 | 0.01084 | assigned control (D_cold=0.000034) |
| 18 | `env_wmo_n2o_2025_ppb` | DROPPED | - | 0.000001 | 0.000001 | -0.00141 | -0.00051 | -0.00013 | -0.00205 | 0.00205 | all 4 models |log10 err| < 0.0086 (max=0.002054) |
| 19 | `cry_arctic_min_2026_kkm2` | KEPT | low_spread | 0.000090 | 0.002250 | -0.03437 | -0.04471 | -0.04997 | -0.06070 | 0.06070 | assigned low_spread (D_cold=0.000090) |
| 20 | `cry_arctic_max_2026_kkm2` | KEPT | low_spread | 0.000976 | 0.000215 | +0.00333 | +0.00030 | -0.06867 | +0.00634 | 0.06867 | assigned low_spread (D_cold=0.000976) |
| 21 | `cry_antarctic_min_2026_kkm2` | KEPT | high_spread | 0.157847 | 0.017382 | +0.81883 | -0.13287 | -0.06920 | -0.08940 | 0.81883 | assigned high_spread (D_cold=0.157847) |
| 22 | `cry_antarctic_max_2026_kkm2` | KEPT | low_spread | 0.001615 | 0.000369 | -0.08390 | -0.00974 | +0.02658 | -0.00974 | 0.08390 | assigned low_spread (D_cold=0.001615) |
| 23 | `dem_japan_births_2025` | KEPT | low_spread | 0.000335 | 0.000001 | -0.00733 | +0.02131 | +0.00882 | -0.02753 | 0.02753 | assigned low_spread (D_cold=0.000335) |
| 24 | `dem_us_births_2025` | KEPT | control | 0.000016 | 0.000035 | +0.00043 | -0.00685 | -0.01053 | -0.00685 | 0.01053 | assigned control (D_cold=0.000016) |
| 25 | `dem_us_population_2025` | DROPPED | - | 0.000002 | 0.000001 | -0.00036 | +0.00255 | +0.00217 | -0.00011 | 0.00255 | all 4 models |log10 err| < 0.0086 (max=0.002553) |
| 26 | `dem_texas_population_2025` | KEPT | low_spread | 0.000258 | 0.000019 | +0.03092 | -0.00288 | -0.01194 | +0.00123 | 0.03092 | assigned low_spread (D_cold=0.000258) |
| 27 | `trn_panama_transits_fy2025` | KEPT | high_spread | 0.066496 | 0.027225 | -0.00502 | -0.03032 | -0.61136 | -0.01329 | 0.61136 | assigned high_spread (D_cold=0.066496) |
| 28 | `trn_panama_tonnage_fy2025` | KEPT | high_spread | 0.231985 | 0.044376 | +0.03076 | +0.03076 | +1.02241 | -0.24131 | 1.02241 | assigned high_spread (D_cold=0.231985) |
| 29 | `trn_panama_revenue_fy2025` | KEPT | high_spread | 0.011309 | 0.014635 | -0.12279 | -0.04025 | -0.29386 | -0.02701 | 0.29386 | assigned high_spread (D_cold=0.011309) |
| 30 | `trn_suez_transits_2025` | KEPT | high_spread | 0.006432 | 0.015007 | +0.22666 | +0.00143 | +0.13726 | +0.12467 | 0.22666 | assigned high_spread (D_cold=0.006432) |
| 31 | `trn_heathrow_pax_2025` | KEPT | control | 0.000061 | 0.000139 | -0.00778 | -0.01304 | -0.02377 | -0.00258 | 0.02377 | assigned control (D_cold=0.000061) |
| 32 | `tou_japan_visitors_2025` | KEPT | low_spread | 0.001046 | 0.000996 | -0.08619 | -0.00187 | -0.01748 | -0.02067 | 0.08619 | assigned low_spread (D_cold=0.001046) |
| 33 | `tou_nps_visits_2025` | KEPT | control | 0.000031 | 0.000326 | +0.02480 | +0.01969 | +0.00929 | +0.01841 | 0.02480 | assigned control (D_cold=0.000031) |
| 34 | `tou_great_smoky_visits_2025` | KEPT | low_spread | 0.000371 | 0.005795 | +0.09054 | +0.05882 | +0.09962 | +0.05552 | 0.09962 | assigned low_spread (D_cold=0.000371) |
| 35 | `ind_world_steel_2025_mt` | KEPT | low_spread | 0.000107 | 0.000083 | +0.01672 | -0.00103 | +0.02189 | -0.00103 | 0.02189 | assigned low_spread (D_cold=0.000107) |
| 36 | `ind_china_steel_2025_mt` | KEPT | high_spread | 0.006158 | 0.004199 | +0.20064 | +0.01519 | +0.02169 | +0.02169 | 0.20064 | assigned high_spread (D_cold=0.006158) |
| 37 | `ind_india_steel_2025_mt` | KEPT | high_spread | 0.155058 | 0.060693 | -0.92719 | -0.03253 | +0.01831 | -0.04403 | 0.92719 | assigned high_spread (D_cold=0.155058) |
| 38 | `aut_tesla_deliveries_2025` | KEPT | high_spread | 0.002158 | 0.007440 | +0.16639 | +0.06034 | +0.06494 | +0.05335 | 0.16639 | assigned high_spread (D_cold=0.002158) |
| 39 | `aut_tesla_storage_2025_mwh` | KEPT | high_spread | 0.069141 | 0.003961 | +0.02965 | +0.01192 | -0.49906 | +0.20574 | 0.49906 | assigned high_spread (D_cold=0.069141) |
| 40 | `aut_europe_ev_sales_2025_k` | KEPT | high_spread | 0.007657 | 0.000015 | -0.11137 | -0.01579 | +0.13263 | -0.02119 | 0.13263 | assigned high_spread (D_cold=0.007657) |
| 41 | `spc_falcon9_launches_2025` | KEPT | high_spread | 0.023333 | 0.022387 | -0.02715 | -0.13112 | -0.03564 | -0.40457 | 0.40457 | assigned high_spread (D_cold=0.023333) |
| 42 | `spc_starlink_falcon9_2025` | KEPT | high_spread | 0.003563 | 0.005471 | -0.06872 | -0.13566 | +0.02068 | -0.11218 | 0.13566 | assigned high_spread (D_cold=0.003563) |
| 43 | `spc_satellites_deployed_2025` | KEPT | high_spread | 0.005554 | 0.013320 | -0.23160 | -0.12337 | -0.07507 | -0.03160 | 0.23160 | assigned high_spread (D_cold=0.005554) |
