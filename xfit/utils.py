from itertools import product
from typing import (
    TYPE_CHECKING,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Hashable,
    TypeVar
)


import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from xarray import DataArray, Dataset
    T_DSorDA = TypeVar("T_DSorDA", DataArray, Dataset)


def coordinate_slice(
    ds: T_DSorDA, 
    dim: str, 
    llim: float = -np.inf, 
    ulim: float = np.inf
    ) -> "np.ndarray":
    """
    Returns a slice of coordinate values of a dataset
    
    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        xarray objet to take slice from
    dim : str
        Name of dimension to slice
    llim : float
        lower limit of slice
    ulim : float
        upper limit of slice
    
    Returns
    -------
    array
    """
    
    mask = np.logical_and(ds[dim]>=llim, ds[dim]<=ulim)
    
    return ds[dim].where(mask,drop=True).values


def da_filter(
    da: T_DSorDA, 
    selections: Mapping[str: Union[Hashable, Sequence[Hashable]]] = {}, 
    omissions: Mapping[str: Union[Hashable, Sequence[Hashable]]] = {}, 
    ranges: Mapping[str: Tuple[float, float]] = {}
    ) -> T_DSorDA:
    """
    Filters an xarray object subject to some criteria on its coordinates. Keeps
    the intersection of all criteria.
    
    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        xarray object to filter
    selections : dict
        Key is dimension name, values are coordinate values which we want to keep
    omissions : dict
        Key is dimension name, values are coordinate values which we want to omit
    ranges : dict
        Key is dimension name, values are a 2-iterable of a lower and upper limit
        (inclusive) which we want to keep.
        
    Returns
    -------
    xr.Dataset or xr.DataArray
        Filtered xarray object
    """
    
    
    selections = {dim: np.atleast_1d(sel) for dim, sel in selections.items()}
    
    filter1 = da.sel(selections)
    
    coord_ranges = {}
    for dim, lims in ranges.items():
        coord_ranges[dim] = coordinate_slice(filter1, dim, lims[0], lims[1])
    filter2 = filter1.sel(coord_ranges)
    
    omitted_coords = {}
    for dim, omission in omissions.items():
        omitted_coords[dim] = [x for x in filter2[dim].values if x not in np.atleast_1d(omission)]
    final = filter2.sel(omitted_coords)
    
    return final


def gen_coord_combo(
    ds: T_DSorDA, 
    drop_dims: Sequence[str] = [], 
    selections: Mapping[str: Union[Hashable, Sequence[Hashable]]] = {}, 
    omissions: Mapping[str: Union[Hashable, Sequence[Hashable]]] = {}, 
    ranges: Mapping[str: Tuple[float, float]] = {}
    ) -> product:
    """
    Generates a cartesian product of combinations of all coordinates of the
    given dataset, subject to coordinate constraints and excluded dimensions.
    First selects, then omits, then finds ranges.
    
    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        xarray object whose coordinates to consider
    drop_dims : list of str
        Dimensions to exclude from the combination
    selections : dict
        Coordinate selections to draw from. Key should be dimension name,
        values a list of coordinate values to select along that dimension
    omissions : dict
        Coordinate omissions. Key should be dimension name,
        values a list of coordinate values to omit along that dimension
    ranges : dict
        Coordinate ranges to select. Key should be a dimension name, values
        should be a 2-tuple of lower and upper coordinate limits to use for
        that dimension
        
    Returns
    -------
    itertools.product
        A cartesian product of the remaining dimensions of `ds`
    """

    remain = da_filter(ds, selections, omissions, ranges)

    # Making a cartesian product of all of the coord vals to loop over
    coord_vals = [np.atleast_1d(remain[dim])
                  for dim in remain.dims]
    
    return product(*coord_vals)


def gen_copy_ds(
    ds: T_DSorDA, 
    new_dvar_names: Sequence[str], 
    drop_dims: Sequence[str] = []
    ) -> Dataset:
    """
    Generates a copy of `ds` with identical coordinates but given
    data variable names
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to base new one on
    new_dvar_names : list of str
        Data variable names of the new dataset
    drop_dims : list of str
        Dimensions to leave off of the new dataset
    
    Returns
    -------
    xarray.Dataset
        New dataset with desired data varialbes and subset of coordinates.
        Entries are meaningless and cannot be relied upon.
    """
   
    remaining_dims = tuple(dim for dim in ds.dims if dim not in drop_dims)
    
    # generate data vars with random data
    sizes = tuple(ds.coords[dim].size for dim in remaining_dims)
    variables = {dv: (remaining_dims, np.empty(sizes)) for dv in new_dvar_names}
        
    new_coords = {d: v.values for d, v in ds.coords.items() if d in remaining_dims}
    
    return xr.Dataset(data_vars=variables, coords=new_coords)


def combine_new_ds_dim(
    ds_dict: Mapping[str, T_DSorDA], 
    new_dim: str
    ) -> T_DSorDA:
    """
    Combines a dictionary of datasets along a new dimension using dictionary keys
    as the new coordinates.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of xarray Datasets or dataArrays
    new_dim : str
        The name of the newly created dimension

    Returns
    -------
    xarray.Dataset
        Merged Dataset or DataArray

    Raises
    ------
    ValueError
        If the values of the input dictionary were of an unrecognized type
    """

    expanded_dss = []

    for coord, ds in ds_dict.items():
        expanded_dss.append(ds.expand_dims(new_dim))
        expanded_dss[-1][new_dim] = [coord]
    new_ds = xr.concat(expanded_dss, new_dim)

    return new_ds
