"""
03_process_regional.py
==================
Regional (AR6 land region) FASTMIP outputs from cached MESMER tas emulations.

For each scenario, produces:
    tas_<scen>_MESMER_regional_quantiles-by-ESM.nc          (output 4)
    tas_<scen>_MESMER_regional_quantiles-across-ESM.nc      (output 5)
    tas_<scen>_MESMER_regional_uncertainty.nc               (output 6b)

Once, across all scenarios (per temporal resolution):
    tas_cross-scenario_MESMER_regional_across-scenario-uncertainty.nc  (output 6a)
"""

import xarray as xr
from mesmer_for_fastmip.mesmer_config import (
    TIER1_MODELS,
    TIER2_MODELS,
)
from mesmer_for_fastmip.fastmip_config import (
    SCENARIOS,
    PATH_MESMER_PROCESSED,
    QUANTILES,
    TEMPORAL_RESOLUTIONS,
    )
from mesmer_for_fastmip.process.functions import (
    quantiles_by_esm,
    quantiles_across_esm,
    estimate_forced_response,
    uncertainty_decomposition,
    cache_forced_mean,
    scenario_uncertainty,
    CACHE_DIR
)
from utils.regions.aggregate_to_regions import compute_regional_means

# ------------------------------------------------------------------ #
# Definitions
# ------------------------------------------------------------------ #

target_variable    = 'tas'
spatial_resolution = 'g025'

aggregation = "regional"

EMULATOR_NAME = "mesmer"

tiers = {'tier1': TIER1_MODELS,
         'tier1-and-2': TIER1_MODELS + TIER2_MODELS,
         'tier2': TIER2_MODELS
         }

# ------------------------------------------------------------------ #
# Per resolution, per scenario output generation
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    for temporal_resolution in TEMPORAL_RESOLUTIONS:
        group_by_month = (temporal_resolution == "mon")

        INPUT_DIR = CACHE_DIR / target_variable / temporal_resolution / spatial_resolution
        OUTPUT_DIR = PATH_MESMER_PROCESSED / target_variable / temporal_resolution / spatial_resolution
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for scenario in SCENARIOS + ["HIST"]:
            print(f"[regional/{temporal_resolution}] {scenario} ...")
            scen_dir = OUTPUT_DIR / scenario
            scen_dir.mkdir(parents=True, exist_ok=True)

            # 1. Load everything for this scenario into memory
            ds = xr.load_dataset(INPUT_DIR / f"{target_variable}_{temporal_resolution}_{scenario}_mesmer_{spatial_resolution}.nc")

            # 2. Aggregate gridcell -> AR6 regions (+ global land) once, up front;
            #    everything below operates on the much smaller regional dataset.
            ds_regional = compute_regional_means(ds, var="tas")

            # 3. Outputs 4 & 5 -- quantiles by / across ESM
            quantiles_by_esm(ds_regional, "tas", QUANTILES).to_netcdf(
                scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_{aggregation}_quantiles-by-ESM.nc"
            )
            quantiles_across_esm(ds_regional, "tas", QUANTILES, tiers=tiers).to_netcdf(
                scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_{aggregation}_quantiles-across-ESM.nc"
            )

            # 4. Forced response / residual, estimated directly on the regional means.
            #    For monthly data, fit independently per calendar month (the
            #    seasonal cycle is still intact -- see functions.py docstring).
            ds_processed = estimate_forced_response(
                ds_regional, variable="tas", method="polyfit", group_by_month=group_by_month
            )

            # 5. Per-scenario uncertainty decomposition (output 6b)
            unc = uncertainty_decomposition(
                ds_processed["forced"], ds_processed["residual"], TIER1_MODELS, TIER2_MODELS
            )
            unc.to_netcdf(scen_dir / f"tas_{scenario}_{EMULATOR_NAME}_{aggregation}_uncertainty.nc")

            # 6. Cache the multi-model-mean forced response for the cross-scenario step below
            cache_forced_mean(
                ds_processed["forced"], scenario, temporal_resolution, label="regional"
            )

            print(f"[regional/{temporal_resolution}] {scenario} done")

        # ------------------------------------------------------------------ #
        #  Cross-scenario uncertainty (output 6a)                             #
        # ------------------------------------------------------------------ #
        var_scenario = scenario_uncertainty(SCENARIOS, temporal_resolution, label="regional")
        var_scenario.to_dataset().to_netcdf(
            OUTPUT_DIR / f"tas_cross-scenario_{EMULATOR_NAME}_{aggregation}_across-scenario-uncertainty.nc"
        )
        print(f"[regional/{temporal_resolution}] cross-scenario uncertainty done")