"""
Some functions for optimal interpolation: defining the interpolation operator
"""

import numpy as np
import xarray as xr

from rosypig import to_xda, apply_matern_2d
from .matern import get_matern

def solve_for_map(ds, m0, obs_mean, obs_std,
                  m_packer, obs_packer, Nx, mask3D, mds):
    """Solve for m_MAP

    $m_{MAP} = m_0 + H^{-1}F^T R^{-1}(d - F m_0)$

    Parameters
    ----------
    ds : xarray Dataset
        containing the EVD of the misfit Hessian, and interpolation operator F
    m0 : xarray DataArray
        with the initial guess for the parameter field
    obs_mean, obs_std : xarray DataArray
        observational data values (mean) and uncertainty (standard deviation)
    m_packer, obs_packer : rosypig ControlField or Observable
        with pack/unpack routines and mask defining the extent of the field
        in the domain
    Nx : int
        parameterizes the prior
    mask3D : xarray DataArray
        this contains m_packer mask, see rosypig.apply_matern_2d
    mds : xarray Dataset
        containing the grid information for the domain m lives in

    Returns
    -------
    ds : xarray Dataset
        with the MAP point (m_MAP) and all misfits to characterize solution
    """

    ds = _add_map_fields(ds)

    # --- Pack some vectors
    [m0,obs_mean,obs_std] = [_unpack_field(x,y) for x,y in zip(
                                        [m0,obs_mean,obs_std],
                                        [m_packer,obs_packer,obs_packer])]
    obs_std_inv = obs_std**-1
    obs_var_inv = obs_std**-2

    # --- Compute initial misfit: F m_0 - d, and weighted versions
    initial_misfit = obs_mean - ds['F'].values @ m0
    with xr.set_options(keep_attrs=True):
        ds['initial_misfit'] = to_xda(initial_misfit,ds)
        ds['initial_misfit_normalized'] = to_xda(obs_std_inv * initial_misfit,ds)
        ds['initial_misfit_normvar'] = to_xda(obs_var_inv * initial_misfit,ds)

    # --- Interp back to model grid and apply posterior via EVD
    misfit_model = ds['F'].T.values @ ds['initial_misfit_normvar'].values
    u_misfit_model = ds['Utilde'].T.values @ misfit_model
    
    for b in ds.beta.values:
        # Currently not storing posterior because it will get big
        #post = ds['Utilde'].values @ np.diag(ds['Dinv'].sel(beta=b)) @ ds['Utilde'].T.values
        
        # --- Finish applying posterior
        du_misfit_model = ds['Dinv'].sel(beta=b).values * u_misfit_model
        udu_misfit_model = ds['Utilde'].values @ du_misfit_model
        
        # --- Compute the MAP point
        mmap = m0 + udu_misfit_model 

        # --- Compute m_map - m0, and weighted version
        dm = m_packer.unpack(mmap - m0)
        dm_normalized = (b**-1) * apply_priorhalf_inv_yz(dm,
                m_packer.unpack(ds['filternorm']),Nx,mask3D,m_packer.mask,
                mds)
        dm_normalized = to_xda(m_packer.pack(dm_normalized),ds)
        with xr.set_options(keep_attrs=True):
            ds['m_map'].loc[{'beta':b}] = to_xda(mmap,ds)
            ds['reg_norm'].loc[{'beta':b}] = .5*np.linalg.norm(dm_normalized,ord=2)

        # --- Compute misfits, and weighted version
        misfits = ds['F'].values @ mmap - obs_mean
        misfits_model_space = rp.to_xda(mmap - ds['F'].T.values @ obs_mean,ds)
        misfits_model_space = misfits_model_space.where(ds['F'].T.values @ obs_mean !=0)
        with xr.set_options(keep_attrs=True):
            ds['misfits'].loc[{'beta':b}] = to_xda(misfits,ds)
            ds['misfits_normalized'].loc[{'beta':b}] = to_xda(obs_std_inv * misfits,ds)
            ds['misfit_norm'].loc[{'beta':b}] = .5*np.linalg.norm(obs_std_inv * misfits,ord=2)
            ds['misfits_model_space'].loc[{'beta':b}] = misfits_model_space
    return ds

def apply_priorhalf_inv_yz(fld,filternorm,Nx,mask3D,mask2D,ds):
    """Apply the square root of the prior covariance operator
        $\Gamma_{prior}^{1/2}f$

    Parameters
    ----------
    fld : xarray DataArray
        To apply the operator to
    filternorm : xarray DataArray
        normalization factor for the prior, i.e. 1/sqrt(filter variance)
        must have same spatial extent as fld
    Nx, mask3D, mask2D, ds :
        see rosypig.matern.apply_matern_2d

    Returns
    -------
    xda : xarray DataArray
        
    """

    filter_std = xr.where(filternorm!=0,1/filternorm,0.)
    C,K = get_matern(Nx,mask2D)
    xda = apply_matern_2d(filter_std *fld,
                          mask3D=mask3D,
                          mask2D=mask2D,
                          ds=ds,
                          delta=C['delta'],Kux=None,Kvy=K['vy'],Kwz=K['wz'])

    return xda

def apply_ppmh(Finterp, fld, filternorm, obs_std,
               input_packer=None, output_packer=None):
    """apply the prior preconditioned misfit Hessian to fld
    (without the matern type smoothing, inverse laplacian-like operator)
    """
    HmTilde = get_ppmh(Finterp, filternorm, obs_std,
                       input_packer=input_packer, output_packer=output_packer)

    fld = fld.values.flatten if input_packer is None else input_packer.pack(fld)
    fld_out = HmTilde @ fld

    return fld_out if input_packer is None else input_packer.unpack(fld_out) 


def get_ppmh(Finterp, filternorm, obs_std,
             input_packer=None, output_packer=None):
    """Get the prior preconditioned misfit Hessian
    (without the matern type smoothing, inverse laplacian-like operator)
    """
    if output_packer is not None:
        obs_variance = xr.where((obs_std!=0)&(output_packer.mask),
                                obs_std**-2, 0.)
        obs_weight = output_packer.pack(obs_variance)
    else:
        obs_variance = xr.where(obs_std!=0,obs_std**-2,0.)
        obs_weight = obs_variance.values.flatten()

    if input_packer is not None:
        F_norm = Finterp * input_packer.pack(filternorm)
    else:
        F_norm = Finterp * filternorm.values.flatten()

    # this is the prior preconditioned misfit Hessian
    return (F_norm.T * obs_weight) @ F_norm

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

def _add_map_fields(ds):
    """Helper routine to define some container fields
    """
    ds['m_map'] = xr.zeros_like(ds['beta']*ds['filternorm'])
    ds['reg_norm'] = xr.zeros_like(ds['beta'])
    ds['misfits'] = xr.zeros_like(ds['beta']*ds['obs_ind'])
    ds['misfits_normalized'] = xr.zeros_like(ds['beta']*ds['obs_ind'])
    ds['misfit_norm'] = xr.zeros_like(ds['beta'])
    ds['misfits_model_space'] = xr.zeros_like(ds['beta']*ds['ctrl_ind'])

    ds['initial_misfit'] = xr.zeros_like(ds['obs_ind'])
    ds['initial_misfit_normalized'] = xr.zeros_like(ds['obs_ind'])
    ds['initial_misfit_normvar'] = xr.zeros_like(ds['obs_ind'])

    # --- some descriptive attributes
    ds['m_map'].attrs = {'label':r'$\mathbf{m}_{MAP}$',
            'description':r'Maximum a Posteriori solution for control parameter $\mathbf{m}$'}
    ds['reg_norm'].attrs = {'label':r'$||\mathbf{m}_{MAP} - \mathbf{m}_0||_{\Gamma_{prior}^{-1}}$',
            'label2':r'$||\Gamma_{prior}^{-1/2}(\mathbf{m}_{MAP} - \mathbf{m}_0)||_2$',
            'description':'Normed difference between initial and MAP solution, weighted by prior uncertainty'}
    ds['misfits'].attrs = {'label':r'$F\mathbf{m}_{MAP} - \mathbf{d}$',
            'description':'Difference between MAP solution and observations'}
    ds['misfits_normalized'].attrs = {'label':r'$\dfrac{F\mathbf{m}_{MAP} - \mathbf{d}}{\sigma_{obs}}$',
            'description':'Difference between MAP solution and observations, normalized by observation uncertainty'}
    ds['misfit_norm'].attrs = {'label':r'$||F\mathbf{m}_{MAP} - \mathbf{d}||_{\Gamma_{obs}^{-1}}$',
            'label2':r'$||\Gamma_{obs}^{-1/2}(F\mathbf{m}_{MAP} - \mathbf{d})||_2$',
            'description':'Normed difference between MAP solution and observations, weighted by observational uncertainty'}
    ds['misfits_model_space'].attrs = {'label':r'$\mathbf{m}_{MAP} - F^T\mathbf{d}$',
            'description':'Difference between MAP solution and observations, in model domain'}
    ds['initial_misfit'].attrs = {'label':r'$F\mathbf{m}_0 - \mathbf{d}$',
            'description':'Difference between initial guess and observations'}
    ds['initial_misfit_normalized'].attrs = {'label':r'$\dfrac{F\mathbf{m}_0 - \mathbf{d}}{\sigma_{obs}}$',
            'description':'Difference between initial guess and observations, normalized by observation uncertainty'}
    ds['initial_misfit_normvar'].attrs = {'label':r'$\Gamma_{obs}^{-1}(F\mathbf{m}_0 - \mathbf{d})$',
            'description':'Difference between initial guess and observations, normalized by observational covariance'}

    return ds


def _unpack_field(fld,packer=None):
    if packer is not None:
        if len(fld.shape)>1 or len(fld)!=packer.n_wet:
            fld = packer.pack(fld)
    else:
        if len(fld.shape)>1:
            fld = fld.flatten() if isinstance(fld,np.ndarray) else fld.values.flatten()
    return fld
