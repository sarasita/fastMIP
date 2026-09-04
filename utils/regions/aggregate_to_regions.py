import numpy as np
import xarray as xr
import regionmask


import numpy as np
import regionmask
import xarray as xr

from mesmer.mask import mask_antarctica, mask_ocean_fraction


def compute_regional_means(
    obj, var=None, mask_ocean=False, compute_global_means=False, threshold=1 / 3
):
    """
    Compute AR6 regional and global means from gridded data.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Input data on a regular latitude-longitude grid, with coordinates
        "lat" and "lon". Accepts either layout:
          - already stacked, with a "gridcell" dimension (lat/lon as
            auxiliary coordinates on it), or
          - unstacked, with separate "lat" and "lon" dimensions -- these
            are stacked into "gridcell" internally before aggregating.

        If a Dataset is passed, the variable to apply the aggregation on
        must be specified, i.e. var != None

    mask_ocean : bool, default False
        If True, mask out ocean grid points (via mesmer's
        `mask_ocean_fraction`, using `threshold`) and Antarctica (via
        mesmer's `mask_antarctica`) before computing the AR6 regional
        means and the "-1" global-land mean below. Note: this requires
        `obj` to have regularly-spaced 1-D "lat"/"lon" dims -- fractional
        land overlap can't be computed on data that's already stacked
        into an irregular "gridcell" dim (mesmer will raise a ValueError
        pointing you to `mask_land` in that case).

    compute_global_means : bool, default False
        If True, additionally compute a genuine whole-globe (land +
        ocean, all latitudes) area-weighted mean, independent of
        `mask_ocean` -- i.e. this is always taken over every original
        grid cell, not the ocean/Antarctica-masked subset used for the
        AR6 regions. Added on top of the regions/global-land mean, under
        the flag value -2.

    threshold : float, default 1/3
        Passed through to mesmer's `mask_ocean_fraction` as the minimum
        land fraction for a grid point to be kept as land. Only used
        when `mask_ocean=True`.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Same type as the input.
    """

    # --------------------------------------------------------------
    # Accept either Dataset or DataArray
    if isinstance(obj, xr.Dataset):
        if var is None:
            raise ValueError(
                "When passing a Dataset, specify the variable name "
                "using `var`, e.g. var='tas'."
            )

        if var not in obj:
            raise ValueError(
                f"'{var}' is not a variable in the Dataset. "
                f"Available variables are {list(obj.data_vars)}."
            )

        da = obj[var]

    elif isinstance(obj, xr.DataArray):
        da = obj
        var = da.name

    else:
        raise TypeError(
            "Input must be an xarray.DataArray or xarray.Dataset."
        )

    # --------------------------------------------------------------
    # Longitude convention -- regionmask's AR6 land regions are defined on
    # -180/180. Normalize defensively here (idempotent if already -180/180)
    # rather than assuming the caller already did this -- a 0-360 input
    # (e.g. raw cmip6-ng) would otherwise cause ar6.mask() to silently
    # misassign/drop regions for roughly half the grid. Done before any
    # stacking below, while "lon" is still guaranteed to be a plain
    # coordinate rather than a MultiIndex level.
    # --------------------------------------------------------------
    da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
    da = da.sortby("lon")

    # Keep a reference to the data *before* any ocean/Antarctica masking,
    # for the optional whole-globe (land + ocean) mean below -- that mean
    # should reflect every original grid cell regardless of mask_ocean.
    da_all_cells = da

    # --------------------------------------------------------------
    # Optional ocean / Antarctica masking. Must happen here, before the
    # "stack to gridcell" step below -- mask_ocean_fraction computes
    # fractional land overlap and needs regularly-spaced 1-D lat/lon
    # dims to do that; it can't be computed once lat/lon have been
    # collapsed into a single (possibly irregular) "gridcell" dim.
    # --------------------------------------------------------------
    if mask_ocean:
        da = mask_ocean_fraction(da, threshold=threshold)
        da = mask_antarctica(da)

    # --------------------------------------------------------------
    # Normalize to a "gridcell" dim, whichever layout was passed in.
    # Must happen before `lat` is pulled out below, so it lines up
    # 1:1 with the (now 1-D) gridcell points either way.
    # --------------------------------------------------------------
    if "gridcell" not in da.dims:
        if "lat" not in da.dims or "lon" not in da.dims:
            raise ValueError(
                "Expected either a 'gridcell' dim or separate 'lat'/'lon' "
                f"dims to stack -- got dims {list(da.dims)}."
            )
        da = da.stack(gridcell=("lat", "lon"))

    if compute_global_means and "gridcell" not in da_all_cells.dims:
        da_all_cells = da_all_cells.stack(gridcell=("lat", "lon"))

    # --------------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------------
    lat = da["lat"]

    # --------------------------------------------------------------
    # AR6 region mask
    # --------------------------------------------------------------
    ar6 = regionmask.defined_regions.ar6.land
    ar6_mask = ar6.mask(da.gridcell)

    # --------------------------------------------------------------
    # Area weights
    # --------------------------------------------------------------
    weights = np.cos(np.deg2rad(lat))
    weights.name = "area_weight"

    # --------------------------------------------------------------
    # Regional means
    # --------------------------------------------------------------
    # NOTE: previously computed as (da * weights).groupby(...).sum("gridcell")
    # / weights.groupby(...).sum("gridcell"). That silently breaks whenever
    # one group (e.g. one ESM's gridcells within one AR6 region) is entirely
    # NaN: .sum(skipna=True) on an all-NaN group returns 0.0 (not NaN), while
    # the weight-sum denominator is unaffected by da's missingness (it's
    # purely mask * cos(lat)) and stays a genuine positive number -- so the
    # result is a hard, clean-looking 0.0 sitting undetected among real
    # values, rather than a propagated NaN. da.weighted(weights).mean(...) is
    # correctly skipna-aware (as already relied on for the global mean
    # below): it normalizes only by the weights of the *valid* cells, so an
    # all-NaN group correctly returns NaN instead.
    def _weighted_group_mean(da_group):
        w = np.cos(np.deg2rad(da_group["lat"]))
        return da_group.weighted(w).mean("gridcell", skipna=True)

    da_ar6 = da.groupby(ar6_mask).map(_weighted_group_mean)

    # --------------------------------------------------------------
    # Global mean (over whatever `da` currently is -- i.e. land-only if
    # mask_ocean=True, land+ocean if mask_ocean=False -- flag -1, as before)
    # --------------------------------------------------------------
    da_global = (
        da.weighted(weights)
        .mean("gridcell", skipna=True)
        .expand_dims(mask=[-1])
    )

    pieces = [da_ar6, da_global]
    flag_values = list(np.concatenate([ar6.numbers, [-1]]))
    flag_meanings = list(ar6.names) + ["GLOBAL"]

    # --------------------------------------------------------------
    # Optional whole-globe (land + ocean, unmasked) mean -- flag -2
    # --------------------------------------------------------------
    if compute_global_means:
        weights_all = np.cos(np.deg2rad(da_all_cells["lat"]))
        da_global_all = (
            da_all_cells.weighted(weights_all)
            .mean("gridcell", skipna=True)
            .expand_dims(mask=[-2])
        )
        pieces.append(da_global_all)
        flag_values.append(-2)
        flag_meanings.append("GLOBAL_ALL")

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------
    da_regions = xr.concat(pieces, dim="mask")

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------
    da_regions["mask"].attrs = {
        "standard_name": "region",
        "flag_values": np.array(flag_values),
        "flag_meanings": " ".join(flag_meanings),
    }

    # --------------------------------------------------------------
    # Return same type as input
    # --------------------------------------------------------------
    if isinstance(obj, xr.Dataset):
        return da_regions.to_dataset(name=var)

    return da_regions