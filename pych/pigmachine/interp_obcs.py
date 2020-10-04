"""
Some functions for optimal interpolation: defining the interpolation operator
"""

import os
import warnings
import numpy as np
import xarray as xr
import subprocess
from shutil import rmtree
from time import sleep
from MITgcmutils import wrmds,rdmds

from rosypig import to_xda, apply_matern_2d, inverse_matern_2d,Simulation
from .matern import get_matern,write_matern,write_matern,get_matern_dataset
from .notation import get_nice_attrs
from .io import read_mds

def get_nx_best(ds):
    """pick a sigma, find Nx for which misfit is minimum
    """
    NxBest = xr.zeros_like(ds.sigma*ds.xi)
    for xi in ds.xi.values:
        for sigma in ds.sigma.values:
            mymisfit = ds.misfit_norm.sel(xi=xi,sigma=sigma)
            nb = mymisfit.Nx.where(mymisfit==mymisfit.min(),drop=True)
            NxBest.loc[{'xi':xi,'sigma':sigma}] = int(nb)
    ds['NxBest'] = NxBest
    return ds

def get_sigma_best(ds):
    """Given Nx best, then sigma best for Nx is the sigma at which misfit is minimized
    """
    if 'NxBest' not in ds:
        ds = get_nx_best(ds)
    sigma_best = ds.sigma.min()*xr.ones_like(ds.Nx*ds.xi)
    for xi in ds.xi.values:
        for Nx in np.unique(ds.NxBest.sel(xi=xi)):
            mymisfit = ds.misfit_norm.sel(xi=xi,Nx=Nx).where(ds.NxBest.sel(xi=xi)==Nx)
            sb = ds.sigma.where(mymisfit==mymisfit.min('sigma'),drop=True)
            sigma_best.loc[{'xi':xi,'Nx':Nx}] = float(sb)
    ds['sigma_best'] = sigma_best
    return ds

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
        Fhi = xr.where(np.isnan(Zhi),0., .5*(1 - dZhi/dZ + dZlo.shift({dim_in.name:1})/dZ))
        Flo = xr.where(np.isnan(Zlo),0., .5*(1 - dZlo/dZ + dZhi.shift({dim_in.name:-1})/dZ))
        op.loc[{dim_out.name:zo}] = Flo+Fhi

    return op



# -----------------------------------------------------------------------
# Deprecated! But I don't have the heart to delete them yet...
# -----------------------------------------------------------------------

def _unpack_field(fld,packer=None):
    if packer is not None:
        if len(fld.shape)>1 or len(fld)!=packer.n_wet:
            fld = packer.pack(fld)
    else:
        if len(fld.shape)>1:
            fld = fld.flatten() if isinstance(fld,np.ndarray) else fld.values.flatten()
    return fld

def solve_for_map_2(ds, m0, obs_mean, obs_std,
                  m_packer, obs_packer, mask3D, mds,
                  n_small=None,xdalike=None,dsim=None,dirs=None):
    """Solve for m_MAP

    $m_{MAP} = H^{-1}(F^T\Gamma_{obs}^{-1}d + \Gamma_{prior}^{-1}m_0)$

    Parameters
    ----------
    ds : xarray Dataset
        containing the EVD of the misfit Hessian, and interpolation operator F
        assumed to have Nx, xi, and beta as dimensions
    m0 : xarray DataArray
        with the initial guess for the parameter field
    obs_mean, obs_std : xarray DataArray
        observational data values (mean) and uncertainty (standard deviation)
    m_packer, obs_packer : rosypig ControlField or Observable
        with pack/unpack routines and mask defining the extent of the field
        in the domain
    mask3D : xarray DataArray
        this contains m_packer mask, see rosypig.apply_matern_2d
    mds : xarray Dataset
        containing the grid information for the domain m lives in
    n_small : int, optional
        only use the first n_small eigenmodes while applying the inverse Hessian
    xdalike : xarray DataArray, optional
        for writing out fields, essentially to reindex the vertical coordinate...
    dsim, dirs : dict, optional
        dictionaries to provide to submit prior application via MITgcm rather than
        here

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

    # --- compute first term
    obs2model = ds['F'].T.values @ (obs_var_inv * obs_mean)

    for nx in ds.Nx.values:
        for xi in ds.xi.values:

            # --- Possibly use lower dimensional subspace and get arrays
            U = ds['U'].sel(Nx=nx,xi=xi).values if n_small is None else ds['U'].sel(Nx=nx,xi=xi).values[:,:n_small]
            filternorm = m_packer.unpack(ds['filternorm'].sel(Nx=nx,xi=xi))
            filterstd = xr.where(filternorm!=0,filternorm**-1,0.)

            # --- second term, apply prior inverse
            m0_normalized = filterstd*m_packer.unpack(m0)
            m0_normalized =  apply_priorhalf_inv_yz( \
                                        fld=m0_normalized,
                                        Nx=nx,xi=xi,
                                        mask3D=mask3D,
                                        mask2D=m_packer.mask,
                                        ds=mds)
            m0_normalized = apply_priorhalf_inv_yz( \
                                        fld=m0_normalized,
                                        Nx=nx,xi=xi,
                                        mask3D=mask3D,
                                        mask2D=m_packer.mask,
                                        ds=mds)
            m0_normalized = m_packer.pack(filterstd*m0_normalized)

            # --- Add prior weighted initial guess and normalized obs in model space
            smooth2DInput = []
            for b in ds.beta.values:
                obs_and_m0 = obs2model + (b**-2)*m0_normalized
                obs_and_m0 = m_packer.unpack(obs_and_m0).reindex_like(xdalike).values
                smooth2DInput.append(obs_and_m0)

            smooth2DInput = np.stack(smooth2DInput,axis=0)

            # --- Apply prior 1/2
            smooth2DOutput = submit_priorhalf( \
                        fld=smooth2DInput,
                        mymask=m_packer.mask,
                        xdalike=xdalike,
                        Nx=nx,xi=xi,
                        write_dir=dirs['write'],
                        namelist_dir=dirs['namelist'],
                        run_dir=dirs['run'],
                        dsim=dsim)

            # --- Apply (I-UDU^T)
            smooth2DInput = []
            for s,b in enumerate(ds.beta.values):

                # --- Possibly account for lower dimensional subspace
                Dinv = ds['Dinv'].sel(Nx=nx,xi=xi,beta=b).values
                Dinv = Dinv if n_small is None else Dinv[:n_small]


                oam = m_packer.pack(filternorm*smooth2DOutput.sel(sample=s))

                udu = U @ (Dinv* (U.T @ oam))
                iudu = oam - udu
                iudu = m_packer.unpack(iudu).reindex_like(xdalike).values
                smooth2DInput.append(iudu)

            smooth2DInput = np.stack(smooth2DInput,axis=0)

            # --- Apply prior 1/2 again
            smooth2DOutput = submit_priorhalf( \
                        fld=smooth2DInput,
                        mymask=m_packer.mask,
                        xdalike=xdalike,
                        Nx=nx,xi=xi,
                        write_dir=dirs['write'],
                        namelist_dir=dirs['namelist'],
                        run_dir=dirs['run'],
                        dsim=dsim)
            for s,b in enumerate(ds.beta.values):
                mmap = (b**2)*m_packer.pack(filternorm*smooth2DOutput.sel(sample=s))
        
                # --- Compute m_map - m0, and weighted version
                dm = filterstd*m_packer.unpack(mmap - m0)
                dm_normalized = (b**-1) * apply_priorhalf_inv_yz( \
                                            fld=dm,
                                            Nx=nx,xi=xi,
                                            mask3D=mask3D,
                                            mask2D=m_packer.mask,
                                            ds=mds)
                dm_normalized = to_xda(m_packer.pack(dm_normalized),ds)
                with xr.set_options(keep_attrs=True):
                    ds['m_map'].loc[{'beta':b,'xi':xi,'Nx':nx}] = to_xda(mmap,ds)
                    ds['reg_norm'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            np.linalg.norm(dm_normalized,ord=2)

                # --- Compute misfits, and weighted version
                misfits = ds['F'].values @ mmap - obs_mean
                misfits_model_space = to_xda(mmap - ds['F'].T.values @ obs_mean,ds)
                misfits_model_space = misfits_model_space.where( \
                                        ds['F'].T.values @ obs_mean !=0)

                with xr.set_options(keep_attrs=True):
                    ds['misfits'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            to_xda(misfits,ds)
                    ds['misfits_normalized'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            to_xda(obs_std_inv * misfits,ds)
                    ds['misfit_norm'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            np.linalg.norm(obs_std_inv * misfits,ord=2)
                    ds['misfits_model_space'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            misfits_model_space
            print(f' --- Done with xi = {xi} ---')
        print(f' --- Done with Nx = {nx} ---')
    return ds
def solve_for_map(ds, m0, obs_mean, obs_std,
                  m_packer, obs_packer, mask3D, mds,
                  n_small=None,xdalike=None,dsim=None,dirs=None):
    """Solve for m_MAP

    $m_{MAP} = m_0 + H^{-1}F^T R^{-1}(d - F m_0)$

    Parameters
    ----------
    ds : xarray Dataset
        containing the EVD of the misfit Hessian, and interpolation operator F
        assumed to have Nx, xi, and beta as dimensions
    m0 : xarray DataArray
        with the initial guess for the parameter field
    obs_mean, obs_std : xarray DataArray
        observational data values (mean) and uncertainty (standard deviation)
    m_packer, obs_packer : rosypig ControlField or Observable
        with pack/unpack routines and mask defining the extent of the field
        in the domain
    mask3D : xarray DataArray
        this contains m_packer mask, see rosypig.apply_matern_2d
    mds : xarray Dataset
        containing the grid information for the domain m lives in
    n_small : int, optional
        only use the first n_small eigenmodes while applying the inverse Hessian
    xdalike : xarray DataArray, optional
        for writing out fields, essentially to reindex the vertical coordinate...
    dsim, dirs : dict, optional
        dictionaries to provide to submit prior application via MITgcm rather than
        here

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

    # --- Map back to model grid and apply posterior via EVD
    misfit_model = ds['F'].T.values @ ds['initial_misfit_normvar'].values

    for nx in ds.Nx.values:
        for xi in ds.xi.values:

            # --- Possibly use lower dimensional subspace and get arrays
            U = ds['U'].sel(Nx=nx,xi=xi).values if n_small is None else ds['U'].sel(Nx=nx,xi=xi).values[:,:n_small]
            filternorm = m_packer.unpack(ds['filternorm'].sel(Nx=nx,xi=xi))
            filterstd = xr.where(filternorm!=0,filternorm**-1,0.)

            # --- Apply prior
            prior_misfit_model = m_packer.unpack(misfit_model)
            smooth2DInput = prior_misfit_model.broadcast_like(ds.beta*prior_misfit_model).reindex_like(xdalike).values
            smooth2DOutput = submit_priorhalf( \
                                    fld=smooth2DInput,
                                    mymask=m_packer.mask,
                                    xdalike=xdalike,
                                    Nx=nx,xi=xi,
                                    write_dir=dirs['write'],
                                    namelist_dir=dirs['namelist'],
                                    run_dir=dirs['run'],
                                    dsim=dsim)
            prior_misfit_model = filternorm*smooth2DOutput.isel(sample=0)
            prior_misfit_model = m_packer.pack(prior_misfit_model)

            smooth2DInput = []
            for b in ds.beta.values:
        
                # --- Possibly account for lower dimensional subspace
                Dinv = ds['Dinv'].sel(Nx=nx,xi=xi,beta=b).values
                Dinv = Dinv if n_small is None else Dinv[:n_small]

                # --- Apply (I-UDU^T)
                udu = U @ ( Dinv * ( U.T @ (b*prior_misfit_model)))
                iudu = b*prior_misfit_model - udu

                iudu = m_packer.unpack(iudu).reindex_like(xdalike).values
                smooth2DInput.append(iudu)

            # --- Apply prior half
            smooth2DInput = np.stack(smooth2DInput,axis=0)
            smooth2DOutput = submit_priorhalf( \
                                fld=smooth2DInput,
                                mymask=m_packer.mask,
                                xdalike=xdalike,
                                Nx=nx,xi=xi,
                                write_dir=dirs['write'],
                                namelist_dir=dirs['namelist'],
                                run_dir=dirs['run'],
                                dsim=dsim)

            for s,b in enumerate(ds.beta.values):

                mmap = m0 + m_packer.pack(b*filternorm*smooth2DOutput.sel(sample=s))

                # --- Compute m_map - m0, and weighted version
                dm = filterstd*m_packer.unpack(mmap - m0)
                dm_normalized = (b**-1) * apply_priorhalf_inv_yz( \
                                            fld=dm,
                                            Nx=nx,xi=xi,
                                            mask3D=mask3D,
                                            mask2D=m_packer.mask,
                                            ds=mds)
                dm_normalized = to_xda(m_packer.pack(dm_normalized),ds)
                with xr.set_options(keep_attrs=True):
                    ds['m_map'].loc[{'beta':b,'xi':xi,'Nx':nx}] = to_xda(mmap,ds)
                    ds['reg_norm'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            np.linalg.norm(dm_normalized,ord=2)

                # --- Compute misfits, and weighted version
                misfits = ds['F'].values @ mmap - obs_mean
                misfits_model_space = to_xda(mmap - ds['F'].T.values @ obs_mean,ds)
                misfits_model_space = misfits_model_space.where( \
                                        ds['F'].T.values @ obs_mean !=0)

                with xr.set_options(keep_attrs=True):
                    ds['misfits'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            to_xda(misfits,ds)
                    ds['misfits_normalized'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            to_xda(obs_std_inv * misfits,ds)
                    ds['misfit_norm'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            np.linalg.norm(obs_std_inv * misfits,ord=2)
                    ds['misfits_model_space'].loc[{'beta':b,'xi':xi,'Nx':nx}] = \
                            misfits_model_space

            print(f' --- Done with xi = {xi} ---')
        print(f' --- Done with Nx = {nx} ---')
    return ds

def submit_priorhalf(fld,
                     mymask,
                     xdalike,
                     Nx,xi,
                     write_dir,namelist_dir,run_dir,
                     dsim,
                     smoothOpNb=1,dataprec='float64'):
    """submit to MITgcm to apply smoothing operator ... much faster
    """

    nrecs = fld.shape[0] if fld.shape!=mymask.shape else 1

    # --- Write and submit
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    fname = f'{write_dir}/smooth2DInput{smoothOpNb:03}'
    wrmds(fname,arr=fld,dataprec=dataprec,nrecords=nrecs)
    write_matern(write_dir,smoothOpNb=smoothOpNb,Nx=Nx,mymask=mymask,
                 xdalike=xdalike,xi=xi)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = Simulation(name='priorhalf',
                         namelist_dir=namelist_dir,
                         run_dir=run_dir,
                         obs_dir=write_dir,**dsim)

    sim.link_to_run_dir()
    sim.write_slurm_script()
    jid = sim.submit_slurm()

    # --- Wait until done
    doneyet = False
    check_str = 'squeue -u $USER -ho %A --sort=S'
    fnameout = f'{run_dir}/smooth2Dfld{smoothOpNb:03}'
    while not doneyet:
        pout = subprocess.run(check_str,shell=True,capture_output=True)
        jid_list = [int(x) for x in pout.stdout.decode('utf-8').replace('\n',' ').split(' ')[:-1]]
        doneyet = jid not in jid_list and os.path.isfile(fnameout+'.data') and os.path.isfile(fnameout+'.meta')
        sleep(1)

    # --- Done! Read the output
    dsout = get_matern_dataset(run_dir,smoothOpNb=smoothOpNb,
                               xdalike=xdalike,
                               sample_num=np.arange(nrecs),
                               read_filternorm=False)
    fld_out = dsout['ginv'].reindex_like(mymask).load();
    rmtree(sim.run_dir)

    return fld_out

def apply_priorhalf_inv_yz(fld,Nx,xi,mask3D,mask2D,ds):
    """Apply the inverse square root of the prior covariance operator
       just the laplace like operator no normalization
        $\Gamma_{prior}^{-1/2}f$

    Parameters
    ----------
    fld : xarray DataArray
        To apply the operator to
    Nx, xi, mask3D, mask2D, ds :
        see rosypig.matern.apply_matern_2d

    Returns
    -------
    xda : xarray DataArray
        
    """

    C,K = get_matern(Nx=Nx,mymask=mask2D,xi=xi)
    xda = apply_matern_2d(fld,
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

    fld = fld.values.flatten() if input_packer is None else input_packer.pack(fld)
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

def _add_map_fields(ds):
    """Helper routine to define some container fields
    """
    bfn = ds['beta']*ds['xi']*ds['Nx']
    ds['m_map'] = xr.zeros_like(bfn*ds['ctrl_ind'])
    ds['reg_norm'] = xr.zeros_like(bfn)
    ds['misfits'] = xr.zeros_like(bfn*ds['obs_ind'])
    ds['misfits_normalized'] = xr.zeros_like(bfn*ds['obs_ind'])
    ds['misfit_norm'] = xr.zeros_like(bfn)
    ds['misfits_model_space'] = xr.zeros_like(bfn*ds['ctrl_ind'])

    ds['initial_misfit'] = xr.zeros_like(ds['obs_ind'])
    ds['initial_misfit_normalized'] = xr.zeros_like(ds['obs_ind'])
    ds['initial_misfit_normvar'] = xr.zeros_like(ds['obs_ind'])

    # --- some descriptive attributes
    for fld in ds.keys():
        ds[fld].attrs = get_nice_attrs(fld)

    return ds
