"""some quick and dirty io routines"""

import numpy as np
import xarray as xr
from MITgcmutils import rdmds, wrmds
from xmitgcm.utils import read_raw_data

def read_pigbin_xy(fname,ds):
    """Read a binary file with dimensions (YC,XC)
    and return dataarray

    Parameters
    ----------
    fname : str
        full path filename to binary file
    ds : xarray Dataset
        with coordinates to create the DataArray

    Returns
    -------
    xda : xarray DataArray
    """
    arr = read_raw_data(fname,np.float64(),ds.Depth.shape).byteswap()
    return xr.DataArray(arr,ds.Depth.coords,ds.Depth.dims)

def read_pigbin_yz(fname,ds):
    """Read a binary file with dimensions (Z,YC)
    and return dataarray

    Parameters
    ----------
    fname : str
        full path filename to binary file
    ds : xarray Dataset
        with coordinates to create the DataArray

    Returns
    -------
    xda : xarray DataArray
    """

    arr = read_raw_data(fname,np.float64(),[len(ds.Z),len(ds.YC)]).byteswap()
    return xr.DataArray(arr,{'Z':ds.Z,'YC':ds.YC},('Z','YC'))

def read_mds(fname,xdalike,rec=None,name=None):
    """Read a meta/data file pair and return
    an xarray DataArray similar to xdalike

    Parameters
    ----------
    fname : str
        path and filename base (i.e. not include suffix .meta / .data)
    xdalike : xarray DataArray
        DataArray with the same coordinates as the file to be read in
    rec : int, optional
        if specified, grabs a specific record from the file
    name : str, optional
        returns named DataArray with this very name

    Returns
    -------
    xda : xarray DataArray
    """
    xda= xr.DataArray(rdmds(fname,rec=rec),coords=xdalike.coords,dims=xdalike.dims)
    if name is not None:
        xda.name=name
    return xda
