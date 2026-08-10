import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import mesmer
import filefisher
import secrets
import cftime
import datetime

from mesmer_for_fastmip.mesmer_config import (scenarios_to_keys,
                                       fair_variable,
                                       input_version,
                                       FILE_FAIR_SELIND,
                                        FILE_FAIR_FORCING,
                                        PATH_FASTMIP_LOCAL,
                                        ALL_MODELS,
                                        training_scenarios,
                                        time_y,
                                        YEAR_START,
                                        CUTOFF_YEAR,
                                        PARAM_PATH_MESMER,
                                        n_members,
                                        buffer_global_variability,
                                        buffer_local_variability,
                                        HIST_PERIOD
)


### MESMER-SPECIFIC SETTIMGS
scenarios           = list(scenarios_to_keys.keys())
target_variable     = 'tas'
temporal_resolution = 'ann'
spatial_resolution  = 'g025'
all_modules = [
    "volcanic",
    "global-variability",
    "local-trends",
    "local-variability",
    "covariance",
    "grid-orig"
]

### FUNCTIONS
def scenario_key(name):
    return scenarios_to_keys[name]

def build_scenario_ds(GMT_df, time = time_y, use_cftime=True):
    if time is None:
        years = GMT_df.columns.astype(int)

        if use_cftime:
            time = [cftime.DatetimeNoLeap(y, 7, 1) for y in years]
        else:
            time = pd.to_datetime(years.astype(str)) + pd.DateOffset(months=6)

    data = GMT_df.values  # shape: (member, time)

    ds = xr.Dataset(
        data_vars={
            "tas": (["member", "time"], data)
        },
        coords={
            "member": np.arange(data.shape[0]),
            "time": time
        }
    )
    return ds

def load_forced_GMT(scenarios):
    selected_indices = pd.read_csv(FILE_FAIR_SELIND, index_col = 0)
    fair_df = pd.read_csv(FILE_FAIR_FORCING)

    tree_dict = {}

    for scenario in scenarios:
        indices = selected_indices[scenario].values

        GMT_df = (fair_df[(fair_df.scenario == scenario) &
                        (fair_df.variable == fair_variable)
                        ]
                .iloc[indices, :]
                .loc[:, slice(YEAR_START, CUTOFF_YEAR)]
                .reset_index(drop=True)
        )

        ds = build_scenario_ds(GMT_df, time = time_y)

        tree_dict[scenario_key(scenario)] = ds
    return(xr.DataTree.from_dict(tree_dict))


### MAIN
if __name__ == '__main__':
    # LOADING FAIR INPUT
    tas_globmean_forcing = load_forced_GMT(scenarios)

    # SETTING MESMER PARAMETERS
    PARAM_FILEFINDER = filefisher.FileFinder(
        path_pattern = PARAM_PATH_MESMER / "{esm}_{scen}",
        file_pattern = "params_{module}_{esm}_{scen}.nc",
    )

    # EXECUTING CODE FOR EACH MODEL
    for model in ALL_MODELS[1:2]:
        print(model)
        param_files = PARAM_FILEFINDER.find_files(esm=model)

        params = xr.DataTree()
        for module in all_modules:
            params[module] = xr.DataTree(
                xr.open_dataset(param_files.search(module=module).paths.pop()), name=module
            )

        # RANDOM SEEEDS
        seed_global_variability = xr.DataTree.from_dict({
            scenario_key(s): xr.Dataset(
                data_vars={
                    "seed": ((), secrets.randbits(32))
                },
                coords={}
            )
            for s in scenarios
        })
        seed_local_variability = xr.DataTree.from_dict({
            scenario_key(s): xr.Dataset(
                data_vars={
                    "seed": ((), secrets.randbits(32))
                },
                coords={}
            )
            for s in scenarios
        })

        # # GMT - VOLC
        # tas_globmean_forcing_volc = mesmer.volc.superimpose_volcanic_influence(
        #     tas_globmean_forcing,
        #     params["volcanic"].ds,
        #     hist_period=HIST_PERIOD,
        # )

        # GMT - VAR
        global_variability = mesmer.stats.draw_auto_regression_uncorrelated(
            params["global-variability"].ds,
            realisation=n_members,
            time=time_y,
            seed=seed_global_variability,
            buffer=buffer_global_variability,
            realisation_dim = 'member'
        )

        global_variability = mesmer.datatree.map_over_datasets(
            lambda ds: ds.rename({"samples": "tas_resids"}), global_variability
        )

        # TAS - LFR
        predictors = xr.merge([tas_globmean_forcing, global_variability])

        lr_params = params["local-trends"].ds
        lr = mesmer.stats.LinearRegression.from_params(lr_params)

        # uses ``exclude`` to split the linear response
        local_forced_response = lr.predict(predictors, exclude={"tas_resids"})

        # local variability part driven by global variabilty - only from `tas_resids`
        local_variability_from_global_var = lr.predict(predictors, only={"tas_resids"})

        # TAS - LOCAL VAR
        local_variability = mesmer.stats.draw_auto_regression_correlated(
            params["local-variability"].ds,
            params["covariance"].localized_covariance_adjusted,
            time=time_y,
            realisation=n_members,
            seed=seed_local_variability,
            buffer=buffer_local_variability,
            realisation_dim = 'member'
        )
        local_variability = mesmer.datatree.map_over_datasets(
            lambda ds: ds.rename({"samples": "prediction"}), local_variability
        )

        local_variability_total = local_variability_from_global_var + local_variability
        emulations = local_forced_response + local_variability_total

        emulations = mesmer.datatree.map_over_datasets(
            lambda ds: ds.rename({"prediction": "tas"}),
            emulations,
        )

        # store
        PATH_OUT = PATH_FASTMIP_LOCAL / target_variable / temporal_resolution / spatial_resolution
        PATH_OUT.mkdir(parents=True, exist_ok=True)

        selected_indices = pd.read_csv(FILE_FAIR_SELIND, index_col = 0)
        selected_indices = selected_indices.rename(columns = scenarios_to_keys)
        for scen in scenarios_to_keys.values():
            local_seed = seed_local_variability[scen].seed.rename("seed_local_variability")
            global_seed = seed_global_variability[scen].seed.rename("seed_global_variability")
            ds = xr.merge([emulations[scen].ds, local_seed, global_seed])
            ds.attrs.update({
                "author": "Sarah",
                "date": datetime.datetime.now(datetime.UTC).isoformat(),
                "fair_input_version": input_version,
                "mesmer_version": 'v.0.10.0',
                "description": "Temperature emulations for FASTMIP phase 2"
             })
            ds["tas"].attrs.update({
                "standard_name": "air_temperature",
                "long_name": "Near-Surface Air Temperature Anomaly",
                "units": "K",
                "cell_methods": "time: mean",
                "comment": "Temperature expressed as anomaly relative to a reference period",
                "reference_period": "1850-1900",
                "frequency": temporal_resolution
            })
            ds["member"].attrs.update({
                "long_name": "ensemble member",
                "description": "Index represents a FAIR configuration selected via Subsampling by Jonas Schwaab"
            })
            ds = ds.assign_coords(
                fair_config=("member", selected_indices[scen].values)
            )
            ds["fair_config"].attrs.update({
                "long_name": "FAIR configuration ID",
                "description": "Original FAIR configuration number"
            })

            emulations[scen] = xr.DataTree(ds)

            # update metadae and
            emulations[scen].to_netcdf(PATH_OUT / f"tas_emulations_{model}_{scen}.nc")
