"""
01_build_ensemble.py
==================
Combines fastmip phase-2 (CMIP7) MESMER emulations into a single file.
These can be deleted once processing steps 2 and 3 were carried out.

Input files:  tas_emulations_<ESM>_<SCENARIO>.nc
              Each file has dimensions: (gridcell, time, member) on an unstructured grid
              with lat/lon coordinate arrays (i.e., gridcell = [(lat_val, lon_val)]).

Output files: Generates data for all CMIP7 scenario over the projection period,
              here: (2022, 2100); Historic data (HIST) is taken as
              (1850,2022) from SSP1-VL

Output storage layout:
  CACHE_DIR / tas / temporal_resolution / g025 /
    tas_<temporal_resolution>_<scenario>_mesmer_g025.nc
"""

import xarray as xr
import time
from mesmer_for_fastmip.mesmer_config import (
    PATH_MESMER_RAW,
    ALL_MODELS,
    PROJ_PERIOD,
    HIST_PERIOD,
    FULL_PERIOD,
)
from mesmer_for_fastmip.fastmip_config import SCENARIOS, PATH_MESMER_PROCESSED, TEMPORAL_RESOLUTIONS
from mesmer_for_fastmip.process.functions import (
    build_ensemble,
    CACHE_DIR
)

# ------------------------------------------------------------------ #
# Definitions
# ------------------------------------------------------------------ #

target_variable     = 'tas'
spatial_resolution  = 'g025'

# -------------------,----------------------------------------------- #
# Ensemble building per scenario
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    for temporal_resolution in TEMPORAL_RESOLUTIONS:
        if temporal_resolution == 'mon':
            suffix = '_monthly'
        else:
            suffix = ''
        OUTPUT_DIR = CACHE_DIR / target_variable / temporal_resolution / spatial_resolution
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        INPUT_DIR = PATH_MESMER_RAW / target_variable / temporal_resolution / spatial_resolution

        for scenario in SCENARIOS[1:]:
            start_time = time.perf_counter()
            print(f"[build_ensemble/{temporal_resolution}] {scenario} ...")

            if scenario == 'SSP1-VL':
                ds = build_ensemble(scenario, ALL_MODELS, INPUT_DIR, period=FULL_PERIOD, suffix = suffix)
                ds.sel(time=HIST_PERIOD).to_netcdf(
                    OUTPUT_DIR / f"{target_variable}_{temporal_resolution}_HIST_mesmer_{spatial_resolution}.nc")
                ds.sel(time=PROJ_PERIOD).to_netcdf(
                    OUTPUT_DIR / f"{target_variable}_{temporal_resolution}_{scenario}_mesmer_{spatial_resolution}.nc")
            else:
                ds = build_ensemble(scenario, ALL_MODELS, INPUT_DIR, period=PROJ_PERIOD, suffix = suffix)
                ds.to_netcdf(
                    OUTPUT_DIR / f"{target_variable}_{temporal_resolution}_{scenario}_mesmer_{spatial_resolution}.nc")

            del ds

            elapsed = time.perf_counter() - start_time

            print(
                f"[build_ensemble/{temporal_resolution}] "
                f"{scenario} done ({elapsed:.2f} s)"
            )