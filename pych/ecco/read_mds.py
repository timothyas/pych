"""
Script to load mds datasets
Essentially a wrapper for xmitgcm routines
"""

import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
import xgcm
import MITgcmutils as mut
from ecco_v4_py import llc_compact_to_tiles


def read_mds(data_dir,grid_dir=None,
             prefix=None, iters='all', 
             ignore_unknown_vars=False,
             llc_method="smallchunks"):
    """A simple wrapper for xmitgcm.open_mdsdataset to reduce my typing
       Set xmitgcm 'face' coordinate -> 'tile' to work with ECCOv4-py

    Parameters
    ----------

    data_dir : string
        location of the *.meta/.data files with ECCO fields
    grid_dir : string, optional
        location of grid variables. If not provided, look in data_dir
    prefix : string, optional
        file prefixes to look for
    iters : list, optional
        see xmitgcm.open_mdsdataset. Set to None to just read the grid
    ignore_unknown_vars : bool, optional
        pretty self explanatory
    llc_method : string, optional
        memmap for how to load llc. 

    Output
    ------

    ds : xarray Dataset
        dataset with all ECCO grid and data variables
        see xmitgcm.open_mdsdataset for details on loading, 
        xarray.DataSet on file format, and the ECCOv4-py
        tutorial for general help
        http://xarray.pydata.org/en/stable/
        https://xmitgcm.readthedocs.io/en/latest/
        https://ecco-v4-python-tutorial.readthedocs.io/index.html#
    """

    # Open up the dataset
    ds = open_mdsdataset(
            data_dir=data_dir,
            grid_dir=grid_dir,
            prefix=prefix,
            iters=iters,
            ignore_unknown_vars=ignore_unknown_vars,
            delta_t=3600,
            ref_date='1992-01-1',
            geometry='llc',
            llc_method=llc_method
    )

    # Swap 'face' coordinate with 'tiles' for use with ecco_v4_py
    ds = ds.rename(name_dict={'face' : 'tile'})

    return ds

def read_single_mds(fnamebase,coords=None,dims=None):
    """Read meta/data that are not grid files but do not have time stamps.
    If coords, dims are provided, create xarray DataArray. 
    Otherwise, return data as numpy ndarray

    Parameters
    ----------
    fnamebase : string
        the base filename <fname>.meta/data
    coords : dictionary, optional
        specify coordinates of the data in the file

    Returns
    -------
    data : numpy ndarray or xarray DataArray
    """

    if (coords is None and dims is not None) and (dims is None and coords is not None):
        return ValueError('Must provide both coords and dims')

    # Read in as n_timesteps x n_vertical_levels x 1170 x 90 numpy.ndarray
    data = mut.rdmds(fnamebase)

    # Convert to the 13 tile format
    data = llc_compact_to_tiles(data, less_output=True)

    # Potentially convert to xarray.DataArray
    if coords is not None:
        data = xr.DataArray(data,coords=coords,dims=dims)

    return data
