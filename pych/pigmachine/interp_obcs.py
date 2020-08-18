"""
Some functions for optimal interpolation: defining the interpolation operator
"""

import numpy as np
import xarray as xr

def interp_operator_2d( dims_in, dims_out,
                        pack_index_in=None,
                        pack_index_out=None ): 
    """Make one interpolation operator to work on a "flattened"
    2D array

    Do this by tiling and repeat the separate interpolation operators

    Parameters
    ----------
    dims_in, dims_out : 2 element lists with xarray DataArrays
        contains the dimensions to interpolate from, to
        see interp_operator for this
    pack_index_in, pack_index_out : array, optional
        defines the "wet points" for the input/output space

    Returns
    -------
    F : numpy array (2D)
        with interpolation operator

    Example
    -------
    Interpolate m with dimensions (Y,X) to (Yobs,Xobs)
    e.g. from model to observation space

    >>> F = interp_operator_2d( [ds['Y'],ds['X']], [ds2['Yobs'],ds2['Xobs']])
    >>> obs = F @ m.flatten()

    or with pack_index

    >>> F = interp_operator_2d( [ds['Y'],ds['X']], [ds2['Yobs'],ds2['Xobs']],index_in)
    >>> obs = F @ m.flatten()[index]
    """

    F0 = interp_operator(dims_in[0],dims_out[0])
    F1 = interp_operator(dims_in[1],dims_out[1])

    dim_out0 = F0.shape[0]
    dim_out1 = F1.shape[0]
    dim_out = dim_out0*dim_out1

    dim_in0 = F0.shape[1]
    dim_in1 = F1.shape[1]
    dim_in  = dim_in0*dim_in1 if pack_index_in is None else len(pack_index_in)

    F = np.zeros([dim_out,dim_in])

    for i,row in enumerate(np.arange(0,dim_out,dim_out1)):
        step1 = np.tile(F1,[1, dim_in0])
        step2 = np.repeat(F0[i,:], dim_in1)
        step3 = np.tile(step2,[dim_out1,1])
        if pack_index_in is None:
            step4 = step3*step1
        else:
            step4 = step3[:,pack_index_in]*step1[:,pack_index_in]

        F[row:row+dim_out1,:] = step4

    if pack_index_out is not None:
        F = F[pack_index_out,:]

    return F


def interp_operator(dim_in,dim_out):
    """return a matrix which interpolates along a single dimension

    Parameters
    ----------
    dim_in, dim_out : xarray DataArray
        containing the coordinate information to interpolate along
        "in" and "out" refer to input and output spaces of interpolation operator

    Returns
    -------
    op : xarray DataArray
        with the input and output dimensions as dimensions

    Example
    -------
    Interpolate 1D vector v, function of dimension "X"
    from "XC" (model space) to "Xobs" (observation space)

    >>> Fx = interp_operator(ds_model['XC'],ds_obs['Xobs'])
    >>> obs = Fx @ v

    """

    op = xr.zeros_like(dim_out.reset_coords(drop=True)*
                       dim_in.reset_coords(drop=True))

    for zo in op[dim_out.name].values:

        dZarray = zo - dim_in
        Zhi = dim_in.where(dZarray==dZarray.where(dZarray<0).max())
        Zlo = dim_in.where(dZarray==dZarray.where(dZarray>0).min())

        dZhi = Zhi-zo
        dZlo = zo-Zlo
        dZ = float(Zhi.dropna(dim_in.name).values-Zlo.dropna(dim_in.name).values)
        Fhi = xr.where(np.isnan(Zhi),0., .5*(1 - dZhi + dZlo.shift({dim_in.name:1})))
        Flo = xr.where(np.isnan(Zlo),0., .5*(1 - dZlo + dZhi.shift({dim_in.name:-1})))
        op.loc[{dim_out.name:zo}] = Flo+Fhi

    return op
