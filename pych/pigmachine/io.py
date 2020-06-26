"""some quick and dirty io routines"""

import numpy as np
import xarray as xr
from xmitgcm.utils import read_raw_data

def read_pigbin_xy(fname,ds):
    arr = read_raw_data(fname,np.float64(),ds.Depth.shape).byteswap()
    return xr.DataArray(arr,ds.Depth.coords,ds.Depth.dims)
def read_pigbin_yz(fname,ds):
    arr = read_raw_data(fname,np.float64(),[len(ds.Z),len(ds.YC)]).byteswap()
    return xr.DataArray(arr,{'Z':ds.Z,'YC':ds.YC},('Z','YC'))
