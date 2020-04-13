"""
Interpolate and return points along section
"""

import numpy as np
import xarray as xr


def get_section_tracers(fldc,left,right,nx=100):
    """
    Interpolate a tracer field to a section line

    Parameters
    ----------
    fldc : xarray DataArray
        Containing tracer field to grab along section
    left, right : tuple or list of 2 floats
        Containing lon/lat bounding points
    nx : int, optional
        Number of interpolation points 

    Returns
    -------
    fldi : xarray DataArray
        with interpolated result along section, dimension i
        and xc/yc as lon/lat along section, dim i
    """

    # Create x/y coords for line
    x = np.linspace(left[0],right[0],nx+1)
    y = np.linspace(left[1],right[1],nx+1)

    # interp to mid point
    # create an index variable: i
    # interpolated result will live along this coordinate
    xc = xr.DataArray(_mov_avg(x),dims='i')
    yc = xr.DataArray(_mov_avg(y),dims='i')

    # Look for a mask for valid points
    maskC = fldc.maskC if 'maskC' in fldc.coords else True*xr.ones_like(fldc)

    # do the interpolation
    fldi = fldc.where(maskC,np.NAN).interp(XC=xc,YC=yc).to_dataset()

    # add lon/lat as coordinates
    fldi['xc']=xc
    fldi['yc']=yc
    fldi=fldi.set_coords('xc')
    fldi=fldi.set_coords('yc')

    return fldi[fldc.name]

def _mov_avg(arr,n=2):
    out = np.cumsum(arr)
    out[n:] -= out[:-n]
    return out[n - 1:] / n
