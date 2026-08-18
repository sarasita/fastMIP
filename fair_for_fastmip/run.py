"""
Running the CMIP6 SSP scenarios through FaIR v2.2.4
using the v1.6.0 calibration + v1.6.0 species set.

Similar to
https://github.com/chrisroadmap/fastmip-phase-1/blob/main/notebooks/run.py
but I didn't have access to any CMIP6 land use &
irrigation data --> mapped CMIP6 scenarios onto CMIP7 data
(SSP_TO_CMIP7_BUCKET)

Probably makes sense to review the section that leverages
RCMIP emissions data (CMIP6 EMISSIONS/CONCENTRATIONS FROM RCMIP)
and the two subsequent sections on Irrigation & Land use
as well as on volcanic & solar


CMIP6 scenarios:
    SSP1-1.9
    SSP1-2.6
    SSP2-4.5
    SSP3-7.0
    SSP5-8.5
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise


# ============================================================
# USER SETTINGS
# ============================================================

PARAMETER_FILE = Path(
    "./fair_for_fastmip/data/fair_parameters_1.6.0/calibrated_constrained_parameters.csv"
)

# NEW: the full species configuration that matches v1.6.0
# (ships alongside calibrated_constrained_parameters.csv in the
# same fair-calibrate v1.6.0 release / Zenodo record).
SPECIES_CONFIG_FILE = Path(
    "./fair_for_fastmip/data/fair_parameters_1.6.0/species_configs_properties.csv"
)

# NEW: directory holding your downloaded CMIP7 forcing files
# for species that RCMIP (CMIP6) cannot provide.
CMIP7_FORCING_DIR = Path(
     "./fair_for_fastmip/data/forcing"
)

# Land use & Irrigation: one column per CMIP7 ScenarioMIP bucket
# (H, HL, M, ML, L, LN, VL)
SCENARIO_DEPENDENT_CMIP7_FILES = {
    "Land use": CMIP7_FORCING_DIR / "land_use_forcing_timebounds_cmip7.csv",
    "Irrigation": CMIP7_FORCING_DIR / "irrigation_forcing_timebounds_cmip7.csv",
}

# Solar & Volcanic: one global file
GLOBAL_CMIP7_FILES = {
    "Solar": {
        "path": CMIP7_FORCING_DIR / "solar_forcing_timebounds_cmip7.csv",
        "year_col": "year",
        "value_col": None,  # auto-detect; set explicitly if needed, e.g. "solar_erf_rel_1850-2019"
    },
    "Volcanic": {
        "path": CMIP7_FORCING_DIR / "volcanic_forcing_timebounds_cmip7.csv",
        "year_col": "Year",
        "value_col": None,  # auto-detect; set explicitly if needed, e.g. "volcanic_erf_rel_1850-2021"
    },
}


SSP_TO_CMIP7_BUCKET = {
    "ssp119": "VL",
    "ssp126": "L",
    "ssp245": "M",
    "ssp370": "H",
    "ssp585": "H",   # no CMIP7 bucket is as extreme as ssp585, not sure what to do here

}

# Year-column auto-detection for the scenario-dependent files
# (tries 'year', else falls back to the first/unnamed column).
# Override per-specie here if needed.
SCENARIO_DEPENDENT_YEAR_COL_OVERRIDES = {}

OUTPUT_DIR = Path(
    "./fair_for_fastmip/output/fair_cmip6"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SCENARIOS / TIME
# ============================================================

SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]

START_YEAR = 1750
END_YEAR = 2100
STEP = 1


# ============================================================
# CHECK FILES
# ============================================================

for path in (PARAMETER_FILE, SPECIES_CONFIG_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Could not find:\n{path}")

for specie, path in SCENARIO_DEPENDENT_CMIP7_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"No CMIP7 file for '{specie}': {path}")

for specie, cfg in GLOBAL_CMIP7_FILES.items():
    if not cfg["path"].exists():
        raise FileNotFoundError(f"No CMIP7 file for '{specie}': {cfg['path']}")


# ============================================================
# READ CALIBRATION PARAMETERS
# ============================================================

print("\nReading v1.6.0 calibration...")

df_configs = pd.read_csv(PARAMETER_FILE, index_col=0)
valid_all = df_configs.index

print(f"Found {len(df_configs)} calibration members.")
print(f"Found {len(df_configs.columns)} parameters.")


# ============================================================
# CREATE FAIR + DECLARE SPECIES
# ============================================================

print("\nCreating FaIR v2.2.4 instance...")

f = FAIR(ch4_method="Thornhill2021")

f.define_time(START_YEAR, END_YEAR, STEP)
f.define_scenarios(SCENARIOS)

species, properties = read_properties(filename=str(SPECIES_CONFIG_FILE))
f.define_species(species, properties)

f.define_configs(valid_all)
print(f"Number of configurations: {len(f.configs)}")

f.allocate()


# ============================================================
# CMIP6 EMISSIONS/CONCENTRATIONS FROM RCMIP
# ============================================================
#
# fill_from_rcmip() tries to fetch driver data for all declared
# species, and fails when one isn't in the RCMIP database
# (Land use / Irrigation aren't). --> run on a throwaway
# instance restricted to subset of v1.6.0 species that are in
# RCMIP (detected by diffing against built-in default species list),
# then copy the results in. Land use / Irrigation / Solar / Volcanic
# are filled separately below, from CMIP7 files.
#
# ============================================================

print("\nDetecting which v1.6.0 species exist in RCMIP...")

default_species, default_properties = read_properties()
rcmip_compatible = set(default_species)

manual_species = set(SCENARIO_DEPENDENT_CMIP7_FILES) | set(GLOBAL_CMIP7_FILES)

non_rcmip_species = [
    s for s in species
    if properties[s]["input_mode"] != "calculated" and s not in rcmip_compatible
]
rcmip_species = [s for s in species if s not in non_rcmip_species]

unexpected_non_rcmip = set(non_rcmip_species) - manual_species
if unexpected_non_rcmip:
    raise RuntimeError(
        f"These v1.6.0 species aren't in RCMIP and aren't covered by "
        f"SCENARIO_DEPENDENT_CMIP7_FILES / GLOBAL_CMIP7_FILES either: "
        f"{sorted(unexpected_non_rcmip)}. Add a CMIP7 source for them "
        f"before running."
    )

print(f"  {len(rcmip_species)} species will be filled from RCMIP.")
print(f"  {len(non_rcmip_species)} species need CMIP7 forcing: {non_rcmip_species}")

print("\nLoading CMIP6 SSP data from RCMIP (helper instance)...")

f_rcmip = FAIR(ch4_method="Thornhill2021")
f_rcmip.define_time(START_YEAR, END_YEAR, STEP)
f_rcmip.define_scenarios(SCENARIOS)
f_rcmip.define_configs(["dummy"])  # driver data doesn't vary by config

rcmip_properties = {s: properties[s] for s in rcmip_species}
f_rcmip.define_species(rcmip_species, rcmip_properties)
f_rcmip.allocate()
f_rcmip.fill_species_configs(filename=str(SPECIES_CONFIG_FILE))
f_rcmip.fill_from_rcmip()

print("Copying RCMIP driver data into the main run...")

n_main_configs = len(f.configs)

for specie in rcmip_species:
    mode = properties[specie]["input_mode"]

    if mode == "emissions":
        target, source = f.emissions, f_rcmip.emissions
    elif mode == "concentration":
        target, source = f.concentration, f_rcmip.concentration
    elif mode == "forcing":
        target, source = f.forcing, f_rcmip.forcing
    else:
        continue

    values = source.sel(specie=specie, config="dummy").values  # (time, scenario)
    broadcast = np.repeat(values[:, :, np.newaxis], n_main_configs, axis=2)
    fill(target, broadcast, specie=specie)

del f_rcmip

print("CMIP6 RCMIP data loaded successfully.")


# ============================================================
# LAND USE & IRRIGATION FORCING (scenario/bucket-dependent, from CMIP7)
# ============================================================
#
# Pre-multiplied by each config's forcing_scale[<specie>], matching
# the reference routine -- FaIR doesn't apply forcing_scale to
# input_mode="forcing" species automatically at runtime.
#
# ============================================================

print("\nFilling Land use / Irrigation forcing from CMIP7 (per-scenario bucket)...")


def load_scenario_dependent_cmip7(csv_path, start_year, end_year, year_col=None):
    """
    Read a CMIP7 forcing CSV with one column per bucket (H, HL, M,
    ML, L, LN, VL) and a year column. Returns df indexed by year
    with columns = bucket names, aligned/interpolated onto an
    annual grid start_year..end_year inclusive.
    """
    df = pd.read_csv(csv_path)
    if year_col is None:
        year_col = "year" if "year" in df.columns else df.columns[0]
    df = df.set_index(year_col).sort_index()

    target_years = np.arange(start_year, end_year + 1)
    df = df.reindex(df.index.union(target_years)).sort_index().interpolate(method="index")
    df = df.reindex(target_years).ffill().bfill()
    return df


n_timebounds = len(f.timebounds)

for specie, csv_path in SCENARIO_DEPENDENT_CMIP7_FILES.items():

    if specie not in f.species:
        raise KeyError(
            f"'{specie}' not found in f.species -- check exact capitalisation "
            f"against species_configs_properties.csv."
        )

    scale_col = f"forcing_scale[{specie}]"
    if scale_col not in df_configs.columns:
        raise KeyError(
            f"Expected column '{scale_col}' in {PARAMETER_FILE.name} to scale "
            f"'{specie}' forcing, but it isn't there. Available forcing_scale "
            f"columns: {[c for c in df_configs.columns if c.startswith('forcing_scale')]}"
        )

    df = load_scenario_dependent_cmip7(
        csv_path, START_YEAR, END_YEAR,
        year_col=SCENARIO_DEPENDENT_YEAR_COL_OVERRIDES.get(specie),
    )

    forcing_scale = df_configs[scale_col].values.squeeze()  # (n_configs,)

    for scenario in SCENARIOS:
        bucket = SSP_TO_CMIP7_BUCKET.get(scenario)
        if bucket is None:
            raise KeyError(f"No CMIP7 bucket mapping for scenario '{scenario}'.")
        if bucket not in df.columns:
            raise KeyError(
                f"Scenario '{scenario}' maps to bucket '{bucket}', not found in "
                f"{csv_path.name}. Available columns: {df.columns.tolist()}"
            )

        raw = df[bucket].values  # (n_years,)
        if len(raw) < n_timebounds:
            raw = np.append(raw, raw[-1])
        elif len(raw) > n_timebounds:
            raw = raw[:n_timebounds]

        scaled = raw[:, None] * forcing_scale[None, :]  # (timebounds, n_configs)

        f.forcing.loc[dict(scenario=scenario, specie=specie)] = scaled

    print(f"  Filled '{specie}' forcing from {csv_path.name} (pre-scaled by {scale_col})")


# ============================================================
# SOLAR & VOLCANIC FORCING (global, non-scenario-dependent, from CMIP7)
# ============================================================

print("\nFilling Solar / Volcanic forcing from CMIP7 (global trajectory)...")

for specie, cfg in GLOBAL_CMIP7_FILES.items():

    if specie not in f.species:
        raise KeyError(
            f"'{specie}' not found in f.species -- check exact capitalisation "
            f"against species_configs_properties.csv."
        )

    scale_col = f"forcing_scale[{specie}]"
    if scale_col not in df_configs.columns:
        raise KeyError(
            f"Expected column '{scale_col}' in {PARAMETER_FILE.name} to scale "
            f"'{specie}' forcing, but it isn't there."
        )

    df = pd.read_csv(cfg["path"])
    year_col = cfg["year_col"] or ("year" if "year" in df.columns else df.columns[0])

    value_col = cfg["value_col"]
    if value_col is None:
        candidates = [c for c in df.columns if c != year_col]
        if len(candidates) != 1:
            raise ValueError(
                f"Can't auto-detect the value column for '{specie}' in "
                f"{cfg['path'].name} -- found {candidates}, expected exactly "
                f"one. Set GLOBAL_CMIP7_FILES['{specie}']['value_col'] explicitly."
            )
        value_col = candidates[0]

    df = df.set_index(year_col).sort_index()

    target_years = np.arange(START_YEAR, END_YEAR + 1)
    series = df[value_col].reindex(df.index.union(target_years)).sort_index().interpolate(method="index")
    series = series.reindex(target_years).ffill().bfill()

    raw = series.values
    if len(raw) < n_timebounds:
        raw = np.append(raw, raw[-1])
    elif len(raw) > n_timebounds:
        raw = raw[:n_timebounds]

    forcing_scale = df_configs[scale_col].values.squeeze()  # (n_configs,)
    scaled = raw[:, None, None] * forcing_scale[None, None, :]  # (timebounds, 1, n_configs) -> broadcasts over scenario

    fill(f.forcing, scaled, specie=specie)

    print(f"  Filled '{specie}' forcing from {cfg['path'].name} (column '{value_col}', pre-scaled by {scale_col})")


# ============================================================
# SANITY CHECK: NOTHING LEFT UNFILLED
# ============================================================

print("\nChecking for unfilled (NaN) driver data...")

for da, label in ((f.emissions, "emissions"), (f.concentration, "concentration"), (f.forcing, "forcing")):
    n_nan = int(np.isnan(da.values).sum())
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values remain in f.{label}.")
    else:
        print(f"  OK: f.{label} fully populated.")


# ============================================================
# SPECIES CONFIGS + CALIBRATION
# ============================================================
#
# fill_species_configs() loads the v1.6.0 defaults; override_defaults()
# then applies calibrated_constrained_parameters.csv on top (climate_configs
# AND species_configs, including bracketed per-specie columns like
# "forcing_scale[Land use]")
#
# ============================================================

print("\nLoading v1.6.0 species configuration values...")

f.fill_species_configs(filename=str(SPECIES_CONFIG_FILE))

print("Applying v1.6.0 calibration via override_defaults()...")

f.override_defaults(str(PARAMETER_FILE))


# ============================================================
# STOCHASTIC RUN + INITIAL CONDITIONS
# ============================================================

fill(f.climate_configs["stochastic_run"], True)

print("\nInitialising model...")

initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)
initialise(f.ocean_heat_content_change, 0)


# ============================================================
# RUN
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FaIR")
print("=" * 70)
print(f"{len(SCENARIOS)} scenarios x {len(f.configs)} calibration members")

f.run(progress=True)


# ============================================================
# SAVE NETCDF
# ============================================================

print("\nSaving NetCDF...")

output_file = OUTPUT_DIR / "cmip6_ssps_fair_v224_v16_cmip7species.nc"
f.to_netcdf(output_file)
print(f"Saved:\n{output_file}")

# ============================================================
# PLOT GMT
# ============================================================

print("\nProcessing GMT...")

temperature = f.temperature.sel(layer=0)
temperature_df = temperature.to_dataframe(name="temperature").reset_index()
temperature_df.to_csv(OUTPUT_DIR / "cmip6_ssps_gmt_all_members.csv", index=False)

plt.figure(figsize=(11, 7))
for scenario in SCENARIOS:
    data = temperature.sel(scenario=scenario)
    years = data.timebounds.values
    median = data.median(dim="config").values
    lower = data.quantile(0.05, dim="config").values
    upper = data.quantile(0.95, dim="config").values
    plt.plot(years, median, label=scenario)
    plt.fill_between(years, lower, upper, alpha=0.15)

plt.axhline(0, linewidth=0.8)
plt.xlabel("Year")
plt.ylabel("Global mean surface temperature change (K)")
plt.title("FaIR v2.2.4 - CMIP6 SSPs\nv1.6.0 calibration + species set, CMIP7-sourced natural/land forcing")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gmt_cmip6_ssps_v224_v16.png", dpi=300)
plt.show()

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("RUN COMPLETE")
print("=" * 70)
print("\nOutput directory:")
print(OUTPUT_DIR.resolve())
print("\nFiles created:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {file.name}")
