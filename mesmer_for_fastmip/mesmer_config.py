"""
mesmer_config.py
=================
Describes the MESMER emulator itself: what it was trained on (FAIR
forcing, training scenarios, ESM tiers) and the time/period conventions
the emulations follow. Pipeline-level choices (which scenarios to output,
quantile levels, realisation counts, temporal resolutions to process) live
in fastmip_config.py instead.

All real filesystem paths come from paths_local.py (gitignored -- see
paths_local.example.py for setup instructions). Nothing in this file
should ever contain a hardcoded absolute path.
"""

from pathlib import Path
import cftime
import xarray as xr
import numpy as np

from mesmer_for_fastmip.paths_local import (
    FAIR_ROOT,
    PARAM_PATH,
    PATH_FASTMIP_UPLOAD,
    PATH_FASTMIP_DOWNLOAD,
    PATH_FASTMIP_LOCAL,
)

# ---- FAIR INPUT
input_version     = 'v20260325' # data verion used

FILE_FAIR_SELIND  = FAIR_ROOT  / 'subselected' / f'subselection_forced_{input_version}.csv' # 100 subselected FAIR indices
FILE_FAIR_FORCING = FAIR_ROOT  / f'climate_assessment_forced_{input_version}.csv' # Forced FAIR response

fair_variable     = 'Climate Assessment|Surface Temperature (GSAT)' # forcing variable


# ---- MESMER Settings
training_scenarios  = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp460", "ssp585", "ssp534-over"] # scenarios used for training MESMER
THRESHOLD_LAND      = 0.1 # threshold for separating land and ocean cells
YEAR_START          = '1850' # year the emulations start
CUTOFF_YEAR         = '2100' # year the emulations end
REFERENCE_PERIOD    = slice("1850", "1900") # reference period for emulations
HIST_PERIOD         = slice("1850", "2020") # historic period, can currently only do until 2022 in line with CMIP7 harmonization
PROJ_PERIOD         = slice("2020", "2100")
FULL_PERIOD         = slice("1850", "2100")

n_members                 = 100
buffer_global_variability = 50
buffer_local_variability  = 20

# ---- Paths derived from paths_local -- not raw paths themselves, so they
# stay here rather than in paths_local.py (single source of truth for the
# root, everything else derives from it).
PARAM_PATH_MESMER     = PARAM_PATH / 'parameters_mesmer_v1' / 'tas'
PARAM_PATH_MESMERM    = PARAM_PATH / 'parameters_mesmer_m_v1'
PATH_MESMER_RAW       = PATH_FASTMIP_LOCAL / "mesmer"


time_y = xr.Coordinates({
    'time': [cftime.DatetimeNoLeap(y, 7, 1)
             for y in np.arange(int(YEAR_START), int(CUTOFF_YEAR)+1)
             ]
    })['time'] # yearly timeseries
time_m = xr.Coordinates({
    'time': [
        cftime.DatetimeNoLeap(y, m, 1)
        for y in range(int(YEAR_START), int(CUTOFF_YEAR) + 1)
        for m in range(1, 13)
        ]
    })['time'] # monthly timeseries


# ---- SCENARIOS included in emulation and naming convention
scenarios_to_keys = {
    'SSP1 - Very Low Emissions': 'SSP1-VL',
    'SSP2 - Low Emissions': 'SSP2-L',
    'SSP2 - Medium Emissions': 'SSP2-M',
    'SSP5 - Medium-Low Emissions_a': 'SSP5-ML',
    'SSP2 - Low Overshoot_a': 'SSP2-LOS',
    'SSP2 - Medium-Low Emissions': 'SSP2-ML',
    'SSP3 - High Emissions': 'SSP3-H'
}

# ----- ESMs that will be emulated
TIER1_MODELS = [
    "ACCESS-ESM1-5",
    "CanESM5",
    "IPSL-CM6A-LR",
    "MPI-ESM1-2-LR",
    "MIROC6"
]
TIER2_MODELS = [
    "ACCESS-CM2",
    "AWI-CM-1-1-MR",
    "BCC-CSM2-MR",
    "CanESM5-1",
    "CESM2",
    "CESM2-WACCM",
    "CNRM-CM6-1",
    "CNRM-ESM2-1",
    "EC-Earth3",
    "FGOALS-g3",
    "FIO-ESM-2-0",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "MIROC-ES2L",
    "MRI-ESM2-0",
    "NorESM2-LM",
    "UKESM1-0-LL",
    "MPI-ESM1-2-HR"
]
ALL_MODELS  = TIER1_MODELS + TIER2_MODELS
