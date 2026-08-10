import numpy as np
import xarray as xr
import regionmask


def compute_regional_means(obj, var=None):
    """
    Compute AR6 regional and global means from gridded data.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Input data on a regular latitude-longitude grid.
        Must contain a dimension "gridcell" and coordinates
        "lat" and "lon".

        If a Dataset is passed, the variable to apply the
        aggregation on must be specified, i.e. var != None

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
    da_ar6 = (
        (da * weights)
        .groupby(ar6_mask)
        .sum("gridcell")
        / weights.groupby(ar6_mask).sum("gridcell")
    )

    # --------------------------------------------------------------
    # Global mean
    # --------------------------------------------------------------
    da_global = (
        da.weighted(weights)
        .mean("gridcell")
        .expand_dims(mask=[-1])
    )

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------
    da_regions = xr.concat(
        [da_ar6, da_global],
        dim="mask",
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------
    flag_values = np.concatenate([ar6.numbers, [-1]])
    flag_meanings = " ".join(ar6.names + ["GLOBAL"])

    da_regions["mask"].attrs = {
        "standard_name": "region",
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
    }

    # --------------------------------------------------------------
    # Return same type as input
    # --------------------------------------------------------------
    if isinstance(obj, xr.Dataset):
        return da_regions.to_dataset(name=var)

    return da_regions