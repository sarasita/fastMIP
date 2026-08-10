## Setup

### 1. Install dependencies

This pipeline uses two separate environments, since generating emulations and
post-processing them have different requirements.

**a) Emulation environment (generating raw MESMER output)**

Needed to run `mesmer_for_fastmip.emulation` and produce
`tas_emulations_<ESM>_<SCENARIO>.nc` files. Follow the install instructions in the
[MESMER repository](https://github.com/MESMER-group/mesmer) to set this up — it
installs the `mesmer` package itself plus its dependencies.

**b) Post-processing environment (`process/01`–`03`)**

Needed to run the scripts in `process/`. Does not require the `mesmer` package
itself — only:

```
numpy
xarray
regionmask
statsmodels
cftime
netCDF4       # or h5netcdf — xarray's netCDF backend
matplotlib    # optional only if plotting is wanted
```

```bash
pip install numpy xarray regionmask statsmodels cftime netCDF4
```

or via `mesmer_for_fastmip/requirements.txt` if present in the repo.

(`03_process_regional.py` also imports `matplotlib` for an ad-hoc diagnostic plot
at the end of the script — add it if you keep that plot, or drop the import if
you remove it before running headless/batch.)

### 2. Configure local paths

This repo keeps machine-specific data paths out of version control. Before running
anything, copy the path template and fill in your own locations:

```bash
cp mesmer_for_fastmip/paths_local.example.py mesmer_for_fastmip/paths_local.py
```

Then edit `mesmer_for_fastmip/paths_local.py` and set each path to where the
corresponding data lives on your machine / cluster:

| Variable | What it points to |
|---|---|
| `FAIR_ROOT` | FAIR forcing input data (climate assessment CSVs) |
| `PARAM_PATH` | Trained MESMER parameter files (tas + MESMER-M) |
| `PATH_FASTMIP_UPLOAD` | FASTMIP phase-2 shared upload location |
| `PATH_FASTMIP_DOWNLOAD` | FASTMIP phase-2 shared download location |
| `PATH_FASTMIP_LOCAL` | Local working directory — raw MESMER files, caches, and processed outputs all live under here |

`paths_local.py` is gitignored and never committed — it's the only file in the
repo that should contain real filesystem paths. `mesmer_config.py` and
`fastmip_config.py` import from it, so this is the one file you need to edit
per machine; nothing else needs to change to run the pipeline locally.

### 3. Run the pipeline

```bash
python -m mesmer_for_fastmip.process.01_build_ensemble
python -m mesmer_for_fastmip.process.02_process_gridcell
python -m mesmer_for_fastmip.process.03_process_regional
```

(adjust `TEMPORAL_RESOLUTIONS` in `fastmip_config.py` to control which
resolutions — `ann`, `mon`, or both — get processed by 01/02/03)
