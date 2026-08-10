# Data generation for the FASTMIP project with MESMER v1

This folder documents and reproduces the MESMER-based temperature emulations used
throughout the FASTMIP project. It covers (1) generating the raw MESMER emulations
from pre-calibrated parameters (`emulation/emulate_FASTMIP_MESMER.py`,
`emulation/emulate_FASTMIP_MESMER-M.py`), and (2) the post-processing pipeline
(`process/01_build_ensemble.py`, `02_process_gridcell.py`, `03_process_regional.py`)
that turns those emulations into the gridcell- and regional-level FASTMIP outputs
(quantiles, uncertainty decomposition, cross-scenario uncertainty).

## What is MESMER?

MESMER (Modular Earth System Model Emulator with spatially Resolved output) is a
Python software package for spatially resolved climate model emulation, developed
and maintained at ETH Zurich. It translates global mean temperature trajectories
into gridcell-level realizations that mimic ESM output, including both the forced
response and internal variability.

Full description:

> Bauer, V. M., Hauser, M., Quilcaille, Y., Schöngart, S., Gudmundsson, L., and
> Seneviratne, S. I.: MESMER v1.0.0: consolidating the modular Earth system model
> emulator into a sustainable research software package, Geosci. Model Dev., 19,
> 5669–5688, [https://doi.org/10.5194/gmd-19-5669-2026](https://doi.org/10.5194/gmd-19-5669-2026), 2026.

Source code: [github.com/MESMER-group/mesmer](https://github.com/MESMER-group/mesmer)
Documentation: [mesmer-emulator.readthedocs.io](https://mesmer-emulator.readthedocs.io/)

### Modules used in FASTMIP

MESMER v1 bundles several previously separate emulation components. FASTMIP uses:

| Module | Emulates | Reference |
|---|---|---|
| **MESMER** (base) | Annual mean temperature | Beusch, L., Gudmundsson, L., and Seneviratne, S. I.: Emulating Earth system model temperatures with MESMER: from global mean temperature trajectories to grid-point-level realizations on land, Earth Syst. Dynam., 11, 139–159, [doi:10.5194/esd-11-139-2020](https://doi.org/10.5194/esd-11-139-2020), 2020. |
| **MESMER-M** | Monthly mean temperature | Nath, S., Lejeune, Q., Beusch, L., Seneviratne, S. I., and Schleussner, C.-F.: MESMER-M: an Earth system model emulator for spatially resolved monthly temperature, Earth Syst. Dynam., 13, 851–877, [doi:10.5194/esd-13-851-2022](https://doi.org/10.5194/esd-13-851-2022), 2022. |
| **MESMER-M-TP** *(not yet used in this pipeline)* | Monthly temperature & precipitation jointly | Schöngart, S., Gudmundsson, L., Hauser, M., Pfleiderer, P., Lejeune, Q., Nath, S., Seneviratne, S. I., and Schleussner, C.-F.: Introducing the MESMER-M-TPv0.1.0 module: spatially explicit Earth system model emulation for monthly precipitation and temperature, Geosci. Model Dev., 17, 8283–8320, [doi:10.5194/gmd-17-8283-2024](https://doi.org/10.5194/gmd-17-8283-2024), 2024. |
| **MESMER-X** *(not yet used in this pipeline)* | Annual temperature extremes, soil moisture, fire weather index | Quilcaille, Y., Gudmundsson, L., Beusch, L., Hauser, M., and Seneviratne, S. I.: Showcasing MESMER-X: Spatially Resolved Emulation of Annual Maximum Temperatures of Earth System Models, Geophys. Res. Lett., 49, e2022GL099012, [doi:10.1029/2022GL099012](https://doi.org/10.1029/2022GL099012), 2022; and Quilcaille, Y., Gudmundsson, L., and Seneviratne, S. I.: Extending MESMER-X: a spatially resolved Earth system model emulator for fire weather and soil moisture, Earth Syst. Dynam., 14, 1333–1362, [doi:10.5194/esd-14-1333-2023](https://doi.org/10.5194/esd-14-1333-2023), 2023. |

This pipeline currently produces **annual** (MESMER) and **monthly** (MESMER-M)
temperature output only.

## Pre-calibrated parameters

Rather than calibrating MESMER ourselves, we use the pre-calibrated parameter set
published alongside MESMER v1:

> Quilcaille, Y., Bauer, V., Hauser, M., Schöngart, S., Gudmundsson, L., and
> Seneviratne, S. I.: Parametrizations for MESMER v1.0.0, ETH Zurich [data set],
> [https://doi.org/10.3929/ethz-c-000798034](https://doi.org/10.3929/ethz-c-000798034), 2026.

This covers annual mean temperature, monthly mean temperature, annual maximum
temperature, soil moisture, and fire weather indicators, calibrated on 58 CMIP6
models. FASTMIP uses the annual and monthly temperature parameters for the models
listed as `TIER1_MODELS`/`TIER2_MODELS` in `mesmer_config.py`.

**Setup:**
1. Install MESMER following the instructions in the
   [MESMER repository](https://github.com/MESMER-group/mesmer) (this pipeline was
   built against the environment produced by that install process; MESMER itself
   is not vendored here).
2. Download the parameter set from
   [https://doi.org/10.3929/ethz-c-000798034](https://doi.org/10.3929/ethz-c-000798034)
   and place it at the location you'll set as `PARAM_PATH` in `paths_local.py`
   (see the repo root `SETUP.md` for the `paths_local.py` setup step — this folder
   consumes `PARAM_PATH`, `PARAM_PATH_MESMER`, and `PARAM_PATH_MESMERM` as defined
   in `mesmer_config.py`).
3. Run the emulation step (`mesmer_for_fastmip.emulation`, see inline docs there)
   to generate `tas_emulations_<ESM>_<SCENARIO>.nc` files per model/scenario.
4. Run the post-processing pipeline described below to turn those emulations into
   FASTMIP's gridcell- and regional-level outputs.

## Post-processing pipeline

Once the raw emulations exist (step 3 above), the scripts in `process/` build the
combined ensemble and derive FASTMIP's standard outputs:

| Script | Produces |
|---|---|
| `01_build_ensemble.py` | Combines per-ESM emulation files into per-scenario ensemble files (`_phase2_cache/`) |
| `02_process_gridcell.py` | Gridcell-level selected realisations, quantiles (by/across ESM), and uncertainty decomposition |
| `03_process_regional.py` | Same as above, aggregated to AR6 land regions |

Both `ann` (annual, MESMER) and `mon` (monthly, MESMER-M) resolutions are
supported; toggle which ones run via `TEMPORAL_RESOLUTIONS` in `fastmip_config.py`.
For monthly data, the forced-response fit and uncertainty decomposition are done
independently per calendar month (the raw monthly anomalies still carry the
seasonal cycle, since they're relative to a single 1850–1900 reference point rather
than a per-month climatology) — see `estimate_forced_response(..., group_by_month=True)`
in `process/functions.py`.

See the repo root `SETUP.md` for the `paths_local.py` configuration step required
before running any of these scripts.