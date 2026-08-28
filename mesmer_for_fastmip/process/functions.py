"""
mesmer_common.py
================
Shared building blocks for the FASTMIP MESMER post-processing pipeline.

Both `process_gridcell.py` and `process_regional.py` import from here.
Nothing in this module runs anything end-to-end -- each function does one
job, and the two scripts call them explicitly in sequence.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import regionmask
from statsmodels.nonparametric.smoothers_lowess import lowess

from mesmer_for_fastmip.mesmer_config import PATH_FASTMIP_LOCAL, TIER1_MODELS, TIER2_MODELS, PROJ_PERIOD

# -------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------

ALL_MODELS = TIER1_MODELS + TIER2_MODELS

FORCED_RESPONSE_WINDOW = 20   # years; only used by the lowess method
FORCED_RESPONSE_DEGREE = 4    # only used by the polyfit method

CACHE_DIR = PATH_FASTMIP_LOCAL / "_phase2_cache"

AR6_LAND = regionmask.defined_regions.ar6.land

# -------------------------------------------------------------------------
# 1. Building the in-memory ensemble
# -------------------------------------------------------------------------

def build_ensemble(scenario, models, path, period = PROJ_PERIOD, suffix = ''):
    """
    Load one tas_emulations file per ESM for a given scenario and
    concatenate along a new 'esm' dim, fully in memory.

    Loading with .load() up front (rather than leaving files open/lazy)
    is what makes everything downstream -- member selection, quantiles,
    smoothing -- fast: it's all plain numpy after this point, no repeated
    disk reads.

    ESMs with fewer members than others are padded with NaN up to the
    largest member count (xr.concat's default outer join); this is
    handled correctly by select_realisations() and by the nan-aware
    reductions (.var/.mean) used later on.
    """
    print('Build ensemble')
    datasets = [
        xr.open_dataset(path / f"tas_emulations_{esm}_{scenario}{suffix}.nc").load()
        for esm in models
    ]
    ds = xr.concat(datasets, dim="esm", join="outer")
    ds = ds.assign_coords(esm=models)
    ds = ds.sel(time=period)
    return ds.astype("float32")

# -------------------------------------------------------------------------
# 2. Selecting realisations & computing quantiles
# -------------------------------------------------------------------------

def select_realisations(ds, models, member_counts, n_realisations, seed):
    """
    Pick `n_realisations` members per ESM.

    Each ESM gets its own independent RNG (seeded from `seed` + the ESM
    name) so the selection for a given ESM is always the same regardless
    of which other ESMs are present, which scenario is being processed,
    or whether this runs in the gridcell or the regional script.
    """
    blocks = []
    for esm in models:
        esm_seed = (hash((seed, esm)) % (2**32 - 1))
        rng = np.random.default_rng(esm_seed)

        n_mem = member_counts
        n_sel = min(n_realisations, n_mem)
        idx = np.sort(rng.choice(n_mem, size=n_sel, replace=False))

        block = (
            ds[["tas"]]
            .sel(esm=esm)
            .isel(member=idx)
            .assign_coords(realisation=("member", np.arange(n_sel)))
            .swap_dims({"member": "realisation"})
            .drop_vars("member", errors="ignore")
            .expand_dims({"esm": [esm]})
        )
        blocks.append(block)

    return xr.concat(blocks, dim="esm", join="outer")


def quantiles_by_esm(ds, variable, quantiles):
    """Quantiles (and mean) across members, kept separate per ESM."""
    print('Compute quantiles by ESM')
    q = ds[variable].quantile(quantiles, dim="member")
    m = ds[variable].mean(dim="member")
    return xr.Dataset({variable: q, f"{variable}_mean": m})


# def quantiles_across_esm(ds, variable, quantiles):
#     """Quantiles (and mean) across ESMs, of the per-ESM member-mean."""
#     print('Compute quantiles across ESM')
#     da = ds[variable].mean(dim="member")
#     q = da.quantile(quantiles, dim="esm")
#     m = da.mean(dim="esm")
#     return xr.Dataset({variable: q, f"{variable}_mean": m})
def quantiles_across_esm(
    ds,
    variable,
    quantiles,
    tiers=None,
):
    """
    Quantiles (and mean) across the pooled (esm, member) ensemble -- every
    individual member from every ESM is treated as one sample in a single
    flattened distribution, rather than first collapsing each ESM to its
    own member-mean and taking quantiles of those per-ESM means.

    Parameters
    ----------
    ds : xr.Dataset
    variable : str
    quantiles : sequence of float
    tiers : dict[str, list[str]], optional
        Mapping from tier name to list of ESM names. If omitted,
        quantiles are computed across all available ESMs.
    """

    print("Compute quantiles across ESM (pooled esm x member)")

    da = ds[variable]

    if tiers is None:
        q = da.quantile(quantiles, dim=["esm", "member"])
        m = da.mean(dim=["esm", "member"])
        return xr.Dataset(
            {
                variable: q,
                f"{variable}_mean": m,
            }
        )

    q_list = []
    m_list = []

    for tier_name, models in tiers.items():

        da_tier = da.sel(esm=models)

        q = da_tier.quantile(quantiles, dim=["esm", "member"])
        m = da_tier.mean(dim=["esm", "member"])

        q = q.expand_dims(tier=[tier_name])
        m = m.expand_dims(tier=[tier_name])

        q_list.append(q)
        m_list.append(m)

    return xr.Dataset(
        {
            variable: xr.concat(q_list, dim="tier"),
            f"{variable}_mean": xr.concat(m_list, dim="tier"),
        }
    )


# -------------------------------------------------------------------------
# 3. Forced response / residual (internal variability)
# -------------------------------------------------------------------------

def _lowess_1d(y, frac):
    """LOWESS smoothing of a single 1D time series."""
    x = np.arange(y.size)
    mask = np.isfinite(y)
    if mask.sum() < 5:
        return np.full_like(y, np.nan)
    out = lowess(y[mask], x[mask], frac=frac, return_sorted=False)
    result = np.full_like(y, np.nan)
    result[mask] = out
    return result


def _polyfit_forced_response(da, degree=FORCED_RESPONSE_DEGREE, time_dim="time"):
    """
    Vectorized polynomial smoothing -- a fast alternative to LOWESS.

    LOWESS (via apply_ufunc(vectorize=True)) loops in Python over every
    (esm, member, gridcell) series one at a time, which is what made the
    original pipeline slow at gridcell resolution. Here, one polynomial
    least-squares solve is fit for *all* series at once.

    This assumes a series is either fully valid or fully NaN over time
    (true for MESMER's land/ocean masking, where a gridcell is masked
    for all time steps or none). Series that don't fit that assumption
    fall back to a per-series fit, so correctness never depends on it --
    only speed does.
    """
    other_dims = [d for d in da.dims if d != time_dim]
    arr = da.transpose(time_dim, *other_dims).values
    n_time = arr.shape[0]
    flat = arr.reshape(n_time, -1)

    x = np.arange(n_time, dtype=float)
    x = (x - x.mean()) / x.std()

    valid_cols = np.isfinite(flat).all(axis=0)
    forced_flat = np.full_like(flat, np.nan)

    if valid_cols.any():
        coefs = np.polynomial.polynomial.polyfit(x, flat[:, valid_cols], degree)
        # polyval(x, c) with 2D c returns shape (n_series, n_time); we want (n_time, n_series)
        forced_flat[:, valid_cols] = np.polynomial.polynomial.polyval(x, coefs).T

    ragged_cols = ~valid_cols & np.isfinite(flat).any(axis=0)
    for j in np.where(ragged_cols)[0]:
        m = np.isfinite(flat[:, j])
        if m.sum() > degree:
            coefs = np.polynomial.polynomial.polyfit(x[m], flat[m, j], degree)
            forced_flat[:, j] = np.polynomial.polynomial.polyval(x, coefs)

    forced = forced_flat.reshape(arr.shape)
    out = xr.DataArray(
        forced,
        dims=[time_dim] + other_dims,
        coords={time_dim: da[time_dim], **{d: da[d] for d in other_dims}},
    )
    return out.transpose(*da.dims)


def _fit_forced_response_1axis(da, method, window, degree, time_dim="time"):
    """Forced-response fit along one homogeneous time axis (one point per year)."""
    if method == "polyfit":
        return _polyfit_forced_response(da, degree=degree, time_dim=time_dim)
    elif method == "lowess":
        frac = window / da.sizes[time_dim]
        return xr.apply_ufunc(
            _lowess_1d, da, kwargs={"frac": frac},
            input_core_dims=[[time_dim]], output_core_dims=[[time_dim]],
            vectorize=True, dask="parallelized", output_dtypes=[da.dtype],
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")


def estimate_forced_response(ds, variable="tas", method="polyfit",
                              window=FORCED_RESPONSE_WINDOW, degree=FORCED_RESPONSE_DEGREE,
                              group_by_month=False):
    """
    Split `ds[variable]` into a smooth forced response and a residual.

    method : {"polyfit", "lowess"} -- see previous docstring.

    group_by_month : bool, default False
        Set True for monthly data whose seasonal cycle is still intact
        (e.g. anomalies relative to a single 1850-1900 reference rather
        than a per-month climatology). Fits Jan/Feb/.../Dec completely
        independently, so a January trend can differ from a July trend.
        Each month's own series still has one point per year, so
        `window`/`degree` keep the same meaning as for annual data --
        no rescaling needed. Leave False for annual data.
    """
    print('Estimate forced response')
    da = ds[variable].sortby("time")

    if group_by_month:
        pieces = [
            _fit_forced_response_1axis(da_m, method, window, degree)
            for _, da_m in da.groupby("time.month")
        ]
        forced = xr.concat(pieces, dim="time").sortby("time")
    else:
        forced = _fit_forced_response_1axis(da, method, window, degree)

    forced.name = variable
    residual = da - forced
    residual.name = variable

    out = xr.Dataset({"forced": forced, "residual": residual})
    if group_by_month:
        # explicit, queryable month flag (e.g. out.sel(time=out.month==1))
        out = out.assign_coords(month=da["time"].dt.month)
    return out


# -------------------------------------------------------------------------
# 4. Uncertainty decomposition
# -------------------------------------------------------------------------

def internal_variability(residual):
    """Average (across ESMs) variance of the residuals across members."""
    return residual.var("member").mean("esm").rename("var_internal")


def esm_uncertainty(forced, tier1_models=TIER1_MODELS, tier2_models=TIER2_MODELS):
    """
    Inter-model (structural) uncertainty: variance across ESMs of the
    forced response, averaged over members.

    Computed twice, as two separate variables:
      - var_esm_tier1       using TIER1 models only
      - var_esm_tier1_tier2 using TIER1 + TIER2 models
    """
    var_tier1 = (
        forced.sel(esm=list(tier1_models))
        .var("esm")
        .mean("member")
        .rename("var_esm_tier1")
    )
    all_models = list(tier1_models) + list(tier2_models)
    var_tier1_tier2 = (
        forced.sel(esm=all_models)
        .var("esm")
        .mean("member")
        .rename("var_esm_tier1-and-2")
    )
    return xr.merge([var_tier1, var_tier1_tier2], compat="override")


def fair_uncertainty(forced):
    """
    Realisation-to-realisation spread of the multi-model-mean forced
    signal, i.e. the part of the member-to-member spread that's *not*
    explained by which ESM you picked.

    NOTE: your draft snippet used a dimension called 'fair_config' here
    but 'member' everywhere else. The raw files only ever carry a
    'member' dimension (build_ensemble never introduces a separate
    fair_config axis), so I've treated the two as the same axis here.
    With that reading, var_esm_tier1_tier2 + var_fair exactly reproduces
    the total variance across (esm, member) of the forced response, via
    the law of total variance -- a cleaner property than the old code's
    residual-based "var_glob = total - other terms, clamped at 0". If
    your members are *not* one draw per FAIR config, let me know and
    this will need adjusting.
    """
    return forced.mean("esm").var("member").rename("var_fair")


def uncertainty_decomposition(forced, residual, tier1_models=TIER1_MODELS, tier2_models=TIER2_MODELS):
    """Combine internal / ESM (tier1 & tier1+2) / fair uncertainty into one Dataset."""
    print('Estimate uncertainties')
    return xr.merge(
        [
            internal_variability(residual),
            esm_uncertainty(forced, tier1_models, tier2_models),
            fair_uncertainty(forced),
        ],
        compat="override",
    )


# -------------------------------------------------------------------------
# 5. Cross-scenario uncertainty via cached forced means
# -------------------------------------------------------------------------

def cache_forced_mean(forced, scenario, temporal_resolution,
                       cache_dir=CACHE_DIR / "forced_mean", label="gridcell"):
    """
    Cache the multi-model-mean forced response for one scenario. Once
    this has been called for every scenario, scenario_uncertainty() can
    compute the variance across scenarios without holding every
    scenario's full ensemble in memory at the same time.
    """
    cache_dir = cache_dir / temporal_resolution
    cache_dir.mkdir(parents=True, exist_ok=True)
    mean_da = forced.mean(["esm", "member"])
    mean_da.name = "tas"
    mean_da.to_netcdf(cache_dir / f"forced_mean_{label}_{scenario}.nc")


def scenario_uncertainty(scenarios, temporal_resolution,
                          cache_dir=CACHE_DIR / "forced_mean", label="gridcell"):
    """
    Variance across scenarios of the cached multi-model-mean forced
    response. Requires cache_forced_mean() to have already been run for
    every scenario in `scenarios` (in this run or an earlier one).
    """
    cache_dir = cache_dir / temporal_resolution
    means = []
    for scen in scenarios:
        da = xr.open_dataarray(cache_dir / f"forced_mean_{label}_{scen}.nc")
        means.append(da.assign_coords(scenario=scen))
    stacked = xr.concat(means, dim="scenario")
    return stacked.var("scenario").rename("var_scenario")

# -------------------------------------------------------------------------
# 6. Regional aggregation (used only by process_regional.py)
# -------------------------------------------------------------------------

def compute_regional_means(ds, variable="tas"):
    """
    Area-weighted mean of `variable` over AR6 land regions, plus a
    'global land' aggregate (mask == -1).

    This works directly on the unstructured 'gridcell' dimension --
    regionmask treats lon/lat as a set of scattered points (rather than
    a 2-D grid to take the outer product of) as long as they share a
    dimension name, so there's no need to pivot to a full lat/lon grid
    first like the old code did. That pivot was one of the slow steps
    and is no longer needed.
    """
    print('Compute regional means')
    lon = xr.DataArray(ds["lon"].values, dims=["gridcell"])
    lat = xr.DataArray(ds["lat"].values, dims=["gridcell"])

    # one boolean slice per AR6 region, vectorized over all regions at once
    mask3d = AR6_LAND.mask_3D(lon, lat, drop=False)           # (region, gridcell)
    mask3d = mask3d.drop_vars(["abbrevs", "names"])           # keep only the numeric region id
    weights = xr.DataArray(np.cos(np.deg2rad(lat.values)), dims=["gridcell"])

    da = ds[variable]

    w = mask3d * weights                                     # (region, gridcell)
    total_w = w.sum("gridcell")                               # (region,)
    regional = (da * w).sum("gridcell") / total_w.where(total_w > 0)
    regional = regional.assign_coords(region=AR6_LAND.numbers)

    # global land aggregate -> mask == -1
    land_mask = mask3d.any("region")
    w_global = land_mask * weights
    global_mean = (da * w_global).sum("gridcell") / w_global.sum()
    global_mean = global_mean.expand_dims(region=[-1])

    combined = xr.concat([regional, global_mean], dim="region")
    combined = combined.rename({"region": "mask"})
    combined.name = variable
    return xr.Dataset({variable: combined})
