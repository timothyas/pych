"""
Interpolate and return points along section
"""

import numpy as np
import xarray as xr


def get_section_tracers(fldc,left,right,nx=100,mask_field=True):
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
    mask_field : bool, optional
        Mask out "non-wet" points, if True 'maskC'
        must be in fldc.coords

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

    # do the interpolation
    if mask_field:
        fldi = fldc.where(fldc.maskC,np.NAN).interp(XC=xc,YC=yc).to_dataset()
    else:
        fldi = fldc.interp(XC=xc,YC=yc).to_dataset()


    # add lon/lat as coordinates
    fldi['xc']=xc
    fldi['yc']=yc
    fldi=fldi.set_coords('xc')
    fldi=fldi.set_coords('yc')

    return fldi[fldc.name]

def get_section_trsp(fldx,fldy,grid,left,right,nx=100):
    """
    Interpolate a vector field to a section line, returning
    the normal component

    Note: DIRECTION NEEDS TO BE VERIFIED!

    Parameters
    ----------
    fldx, fldy : xarray DataArray
        Containing vector field to grab along section
    left, right : tuple or list of 2 floats
        Containing lon/lat bounding points
    nx : int, optional
        Number of interpolation points 

    Returns
    -------
    q : xarray DataArray
        with interpolated vector field into section, dimension i
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
    maskW = fldx.maskW if 'maskW' in fldx.coords else True*xr.ones_like(fldx)
    maskS = fldy.maskS if 'maskS' in fldy.coords else True*xr.ones_like(fldy)

    # interpolate U and V to this point
    uvel = grid.interp(fldx.where(maskW,np.NAN),'X',boundary='fill').interp(XC=xc,YC=yc)
    vvel = grid.interp(fldy.where(maskS,np.NAN),'Y',boundary='fill').interp(XC=xc,YC=yc)

    # get coordinate system tangent and normal to this line
    dxc = xr.DataArray(np.diff(x),dims='i')
    dyc = xr.DataArray(np.diff(y),dims='i')
    sin = dyc / np.sqrt(dxc**2 + dyc**2)
    cos = dxc / np.sqrt(dxc**2 + dyc**2)

    q = -sin*uvel + cos*vvel
    myname=fldx.name[:-1]#drop the W,S
    q.name=myname

    # add xc,yc
    q= q.to_dataset()
    q['xc'] = xc.copy()
    q['yc'] = yc.copy()
    q = q.set_coords('xc')
    q = q.set_coords('yc')

    return q[myname]

def _mov_avg(arr,n=2):
    out = np.cumsum(arr)
    out[n:] -= out[:-n]
    return out[n - 1:] / n
