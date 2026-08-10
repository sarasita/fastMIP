"""
paths_local.py (template)
==========================
Machine-/cluster-specific paths. Copy this file to `paths_local.py` in the
same directory and fill in real paths for your environment.

    cp paths_local.example.py paths_local.py

`paths_local.py` is gitignored and never committed -- it's the only place
in this repo that should contain real filesystem paths. Everything else
(mesmer_config.py, fastmip_config.py, and all processing scripts) imports
from here rather than hardcoding paths; this is the single file you need
to edit to run the pipeline locally.
"""

from pathlib import Path

# Root of the FAIR forcing input data (climate assessment CSVs).
FAIR_ROOT = Path('/path/to/fair_input')

# Root of the trained MESMER parameter files (tas + MESMER-M).
PARAM_PATH = Path('/path/to/mesmer_v1_training')

# FASTMIP phase-2 shared/network locations (upload, download, local scratch).
PATH_FASTMIP_UPLOAD   = Path('/path/to/FASTMIP_upload/FASTMIP_phase2')
PATH_FASTMIP_DOWNLOAD = Path('/path/to/FASTMIP_phase2')

# Local working directory: raw MESMER files, caches, and processed outputs
# all live under here (see mesmer_config.py / fastmip_config.py).
PATH_FASTMIP_LOCAL = Path('/path/to/local/data/fastMIP')
