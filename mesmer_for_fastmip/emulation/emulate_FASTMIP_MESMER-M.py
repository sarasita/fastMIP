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
import secrets

from tqdm.auto import tqdm

from mesmer_for_fastmip.mesmer_config import (scenarios_to_keys,
                                       input_version,
                                       ALL_MODELS,
                                       time_m,
                                       PARAM_PATH_MESMERM,
                                       buffer_local_variability
)

#============================================================
# 0. OPTIONS
#============================================================
# original names of scenarios
scenarios = list(scenarios_to_keys.keys())

# name of variable
target_variable = "tas"

# temporal resolutions of predictor and target
time_res_targ = "mon"
time_res_pred = "ann"

# spatial resolution (common to predictor and target)
spat_res = "g025"

# moduels included in MESMER-M parameter set
all_modules = [
    "harmonic-model",
    "power-transformer",
    "local-variability",
    "covariance"
]

PATH_IN = PATH_FASTMIP_LOCAL / target_variable / time_res_pred / spat_res
PATH_OUT = PATH_FASTMIP_LOCAL / target_variable / time_res_targ / spat_res
PATH_OUT.mkdir(parents = True, exist_ok=True)

#============================================================
# FUNCTIONS
#============================================================

def scenario_key(name):
    return scenarios_to_keys[name]

#============================================================
# 1. PREPARATION OF FILES
#============================================================
# setting immediately the priority of the job
# SETTING MESMER PARAMETERS
PARAM_FILEFINDER = filefisher.FileFinder(
    path_pattern = PARAM_PATH_MESMERM / "{esm}_{scen}",
    file_pattern = "params_{module}_{esm}_{scen}.nc",
)

if __name__ == '__main__':
    for model in ALL_MODELS:
        print(model)
        param_files = PARAM_FILEFINDER.find_files(esm=model)

        # load parameters
        params = xr.DataTree()
        for module in all_modules:
            params[module] = xr.DataTree(
                xr.open_dataset(param_files.search(module=module).paths.pop()), name=module
            )

        harmonic_coeffs = params['harmonic-model'].to_dataset()['coeffs'].load()
        local_var_ds = params['local-variability'].to_dataset().load()
        covariance = params['covariance'].to_dataset().localized_covariance.load()
        lambda_coeffs = params['power-transformer'].to_dataset()['lambda_coeffs'].load()

        for scen in scenarios_to_keys.values():
            # output file
            FILE_OUT = PATH_OUT / f"tas_emulations_{model}_{scen}_monthly.nc"

            # open lazily
            tas_y = xr.open_dataset(
                PATH_IN / f"tas_emulations_{model}_{scen}.nc"
            )['tas']
            members = tas_y.member.values

            # pre-generate seeds (reproducibility)
            seeds = [secrets.randbits(32) for _ in members]

            def build_member_emulation(member, seed):
                tas_member = tas_y.sel(member=member).load()

                # Harmonic model
                monthly_harmonic_emu = mesmer.stats.predict_harmonic_model(
                    tas_member,
                    harmonic_coeffs,
                    time_m
                )

                # AR(1) variability
                local_variability_transformed = mesmer.stats.draw_auto_regression_monthly(
                    local_var_ds,
                    covariance,
                    time=time_m,
                    n_realisations=1,
                    seed=seed,
                    buffer=buffer_local_variability,
                    realisation_dim="member"
                )

                # inverse transform
                yj_transformer = mesmer.stats.YeoJohnsonTransformer("logistic")
                local_variability_inverted = yj_transformer.inverse_transform(
                    tas_member,
                    local_variability_transformed.isel(member=0).samples,
                    lambda_coeffs,
                )

                # FINAL NUMPY OUTPUT ONLY
                emu = (
                    monthly_harmonic_emu + local_variability_inverted.inverted
                ).values

                return emu

            tas_out = []
            seed_out = []

            for i, (member, seed) in tqdm(enumerate(zip(members, seeds)), total = len(members)):
                print(i)

                emu = build_member_emulation(member, seed)

                tas_out.append(emu)
                seed_out.append(seed)
                del emu

            tas_out = np.stack(tas_out, axis=0)
            seed_out = np.array(seed_out)

            ds_out = xr.Dataset(
                data_vars={
                    "tas": (("member", "time", "gridcell"), tas_out),
                    "monthly_variability_seed": (("member",), seed_out),
                },
                coords={
                    "member": tas_y.member,
                    "time": time_m,
                    "gridcell": tas_y.gridcell
                }
            )

            ds_out.attrs.update({
                "author": "Sarah",
                "date": datetime.datetime.now(datetime.UTC).isoformat(),
                "fair_input_version": input_version,
                "mesmer_version": "v0.10.0",
                "description": "Temperature emulations for FASTMIP phase 2"
            })

            ds_out["tas"].attrs.update({
                "standard_name": "air_temperature",
                "long_name": "Near-Surface Air Temperature Anomaly",
                "units": "K",
                "cell_methods": "time: mean",
                "comment": "Temperature expressed as anomaly relative to a reference period",
                "reference_period": "1850-1900",
                "frequency": time_res_targ
            })

            ds_out["monthly_variability_seed"].attrs.update({
                "long_name": "Random seed used for AR(1) monthly variability generation",
                "description": "Seed ensures reproducibility of stochastic variability per ensemble member"
            })

            ds_out.to_netcdf(FILE_OUT)



        # # looping over members
        # for i, member in enumerate(tas_y['member'].values):
        #     seed = seeds[i]
        #     tas_member = tas_y.sel(member=member).load()

        #     # Harmonic model
        #     monthly_harmonic_emu = mesmer.stats.predict_harmonic_model(
        #         tas_member,
        #         harmonic_coeffs,
        #         time_m
        #     )

        #     # AR(1) variability
        #     local_variability_transformed = mesmer.stats.draw_auto_regression_monthly(
        #         local_var_ds,
        #         covariance,
        #         time=time_m,
        #         n_realisations=1,
        #         seed=seed,
        #         buffer=buffer_local_variability,
        #         realisation_dim='member'
        #     )

        #     # inverse transform
        #     yj_transformer = mesmer.stats.YeoJohnsonTransformer("logistic")
        #     local_variability_inverted = yj_transformer.inverse_transform(
        #         tas_member,
        #         local_variability_transformed.isel(member=0).samples,
        #         lambda_coeffs,
        #     )

        #     emu_member = monthly_harmonic_emu + local_variability_inverted.inverted
        #     emu_member = emu_member.expand_dims(member=[member])

        #     # build dataset for this member
        #     ds_out = emu_member.to_dataset(name="tas")
        #     ds_out["monthly_variability_seed"] = xr.DataArray(
        #         [seed],
        #         dims="member",
        #         coords={"member": [member]}
        #     )

        #     ds_out["tas"].attrs.update({
        #         "standard_name": "air_temperature",
        #         "long_name": "Near-Surface Air Temperature Anomaly",
        #         "units": "K",
        #         "cell_methods": "time: mean",
        #         "comment": "Temperature expressed as anomaly relative to a reference period",
        #         "reference_period": "1850-1900",
        #         "frequency": time_res_targ
        #     })

        #     # optional but good practice
        #     ds_out["monthly_variability_seed"].attrs.update({
        #         "long_name": "Random seed used for AR(1) monthly variability generation",
        #         "description": "Seed ensures reproducibility of stochastic variability per ensemble member"
        #     })

        #     # write incrementally
        #     if i == 0:
        #         ds_out.attrs.update({
        #             "author": "Sarah",
        #             "date": datetime.datetime.now(datetime.UTC).isoformat(),
        #             "fair_input_version": input_version,
        #             "mesmer_version": "v0.10.0",
        #             "description": "Temperature emulations for FASTMIP phase 2"
        #         })

        #         ds_out.to_netcdf(FILE_OUT, mode="w")
        #     else:
        #         ds_out.to_netcdf(FILE_OUT, mode="a")

        #     del (ds_out,
        #          emu_member,
        #          local_variability_inverted,
        #          monthly_harmonic_emu,
        #          local_variability_transformed
        #     )