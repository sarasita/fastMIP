# mesmer_for_fastmip

Data generation & post-processing for FASTMIP's MESMER-based temperature
emulations. Full context: see `README.md` (what MESMER is, citations, module
overview) and `SETUP.md` (environment + path setup steps).

## Structure

- `mesmer_config.py` — MESMER emulator/input properties (ESM tiers, training
  scenarios, FAIR forcing, time coordinates). Not pipeline choices.
- `fastmip_config.py` — pipeline choices (scenarios to output, quantiles,
  realisation counts, temporal resolutions). Not emulator properties.
- `paths_local.py` — **gitignored, machine-specific paths only.** Never add a
  real filesystem path anywhere else in this folder — import it from
  `paths_local` instead. If it's missing, copy `paths_local.example.py` and
  fill in real paths; don't invent placeholder values elsewhere as a workaround.
- `process/functions.py` — shared building blocks. `01`/`02`/`03` call these
  explicitly in sequence; nothing in `functions.py` runs end-to-end on its own.
- `process/01_build_ensemble.py` → `02_process_gridcell.py` → `03_process_regional.py`
  — run in this order. `01` must complete for a scenario before `02`/`03` can
  read its cached output.

## Two separate environments — don't assume one covers both

- **Emulation** (generating raw `tas_emulations_*.nc` files): needs the actual
  `mesmer` package, installed per the
  [MESMER repo](https://github.com/MESMER-group/mesmer) instructions.
- **Post-processing** (`process/01`–`03`): does *not* need `mesmer` installed —
  only `numpy`, `xarray`, `regionmask`, `statsmodels`, `cftime`, `netCDF4`
  (see `requirements.txt`).

Don't add a `mesmer` import to anything under `process/` — it's deliberately
decoupled from the emulation environment.

## Known conventions to preserve

- `TEMPORAL_RESOLUTIONS` in `fastmip_config.py` controls which of `ann`/`mon`
  get processed — respect it rather than hardcoding a resolution in a script.
- Monthly (`mon`) forced-response fitting is done **per calendar month**
  (`estimate_forced_response(..., group_by_month=True)`), because the raw
  monthly anomalies still carry the seasonal cycle (reference period is a
  single 1850–1900 baseline, not a per-month climatology). Don't fit monthly
  data as one continuous series without this flag.
- Cache paths under `_phase2_cache/forced_mean/` are nested by
  `temporal_resolution` — if you add a new cache/output path, nest it the
  same way, or `ann` and `mon` runs will silently overwrite each other.
- This folder is one of several sibling model folders in the repo
  (`meteor_for_fastmip`, etc.) — keep it self-contained; don't import from or
  add dependencies on another model's folder.

## When editing

- Prefer paraphrasing/checking against `README.md`'s citation list rather than
  inventing new citations for MESMER components.
- Flag (don't silently fix) anything that looks like it would change scientific
  output — e.g. quantile lists, seeds, tier membership, window/degree constants.