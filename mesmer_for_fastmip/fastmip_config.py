"""
fastmip_config.py
==================
Pipeline-level choices for the FASTMIP post-processing scripts (01/02/03):
which scenarios to produce output for, quantile levels, realisation
counts, and which temporal resolutions to process. Emulator/input-data
properties (ESM tiers, training scenarios, FAIR forcing) live in
mesmer_config.py instead.

Real filesystem paths come from paths_local.py (gitignored -- see
paths_local.example.py for setup instructions).
"""

from mesmer_for_fastmip.paths_local import PATH_FASTMIP_LOCAL

SCENARIOS = [
    "SSP1-VL",
    "SSP2-L",
    "SSP2-M",
    "SSP3-H",
    "SSP2-LOS",
    "SSP2-ML",
    "SSP5-ML",
]

QUANTILES       = [0.01, 0.025, 0.05, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 0.975, 0.99]
N_REALISATIONS  = 10
SEED            = 1430

TEMPORAL_RESOLUTIONS = ["mon"]
# TEMPORAL_RESOLUTIONS = ["ann", "mon"]

# ---- Path derived from paths_local -- not a raw path itself, so it stays
# here rather than in paths_local.py (PATH_FASTMIP_LOCAL is the single
# source of truth; everything else derives from it).
PATH_MESMER_PROCESSED = PATH_FASTMIP_LOCAL / "fastmip_outputs"
