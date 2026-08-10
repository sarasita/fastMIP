"""
02_process_gridcell.py
==================
Generates Gridcell-level FASTMIP outputs from cached MESMER tas emulations.

For each scenario, produces:
    tas_<scen>_MESMER_gridcell_selected-realisations.nc      (output 1)
    tas_<scen>_MESMER_gridcell_quantiles-by-ESM.nc           (output 2)
    tas_<scen>_MESMER_gridcell_quantiles-across-ESM.nc       (output 3)
    tas_<scen>_MESMER_gridcell_uncertainty.nc                (output 7b)

Once, across all scenarios:
    tas_cross-scenario_MESMER_gridcell_across-scenario-uncertainty.nc  (output 7a)

"""

import xarray as xr
from mesmer_for_fastmip.mesmer_config import (
    ALL_MODELS,
    TIER1_MODELS,
    TIER2_MODELS,
)
from mesmer_for_fastmip.fastmip_config import (
    SCENARIOS,
    PATH_MESMER_PROCESSED,
    N_REALISATIONS,
    SEED,
    QUANTILES
    )
from mesmer_for_fastmip.process.functions import (
    select_realisations,
    quantiles_by_esm,
    quantiles_across_esm,
    estimate_forced_response,
    uncertainty_decomposition,
    cache_forced_mean,
    scenario_uncertainty,
    CACHE_DIR
)

# ------------------------------------------------------------------ #
# Definitions
# ------------------------------------------------------------------ #

target_variable     = 'tas'
temporal_resolution = 'ann'
spatial_resolution  = 'g025'

EMULATOR_NAME = "mesmer"

INPUT_DIR = CACHE_DIR / target_variable / temporal_resolution / spatial_resolution

OUTPUT_DIR = PATH_MESMER_PROCESSED / target_variable / temporal_resolution / spatial_resolution
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tiers = {'tier1': TIER1_MODELS,
         'tier1-and-2': TIER1_MODELS + TIER2_MODELS,
         'tier2': TIER2_MODELS
         }

# ------------------------------------------------------------------ #
# Per scenario output generation
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for temporal_resolution in TEMPORAL_RESOLUTIONS:
        group_by_month = (temporal_resolution == "mon")

        INPUT_DIR = CACHE_DIR / target_variable / temporal_resolution / spatial_resolution
        OUTPUT_DIR = PATH_MESMER_PROCESSED / target_variable / temporal_resolution / spatial_resolution
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for scenario in SCENARIOS + ["HIST"]:
            print(f"[gridcell/{temporal_resolution}] {scenario} ...")
            scen_dir = OUTPUT_DIR / scenario
            scen_dir.mkdir(parents=True, exist_ok=True)

            ds = xr.load_dataset(INPUT_DIR / f"{target_variable}_{temporal_resolution}_{scenario}_mesmer_{spatial_resolution}.nc")

            member_counts = len(ds.member)
            selected = select_realisations(ds, ALL_MODELS, member_counts, N_REALISATIONS, SEED)
            selected.set_index(gridcell=["lat", "lon"]).unstack("gridcell").sortby("lon").to_netcdf(
                scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_gridcell_selected-realisations.nc")
            del selected

            quantiles_by_esm(ds, "tas", QUANTILES).set_index(gridcell=["lat", "lon"]).unstack("gridcell").sortby("lon").to_netcdf(
                scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_gridcell_quantiles-by-ESM.nc")

            quantiles_across_esm(ds, "tas", QUANTILES, tiers=tiers).set_index(gridcell=["lat", "lon"]).unstack("gridcell").sortby("lon").to_netcdf(
                scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_gridcell_quantiles-across-ESM.nc")

            ds_processed = estimate_forced_response(ds, variable="tas", method="polyfit", group_by_month=group_by_month)

            unc = uncertainty_decomposition(
                ds_processed["forced"], ds_processed["residual"], TIER1_MODELS, TIER2_MODELS
            )
            unc.set_index(gridcell=["lat", "lon"]).unstack("gridcell").sortby("lon").to_netcdf(
                scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_gridcell_uncertainty.nc")

            cache_forced_mean(
                ds_processed["forced"].set_index(gridcell=["lat", "lon"]).unstack("gridcell").sortby("lon"),
                scenario, temporal_resolution, label="gridcell"
            )

            print(f"[gridcell/{temporal_resolution}] {scenario} done")

        var_scenario = scenario_uncertainty(SCENARIOS, temporal_resolution, label="gridcell")
        var_scenario.to_dataset().to_netcdf(
            OUTPUT_DIR / f"tas_cross-scenario_{EMULATOR_NAME}_gridcell_across-scenario-uncertainty.nc"
        )
        print(f"[gridcell/{temporal_resolution}] cross-scenario uncertainty done")