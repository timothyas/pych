"""
Some helper functions for the matern prior covariance
Hmm... it would be nice to add this on to the ControlField
infrastructure...
Maybe once I understand things more
"""

import os
import numpy as np
import xarray as xr
from MITgcmutils import wrmds

from .io import read_mds

def get_alpha(ds):
    """Return the grid-define aspect ratio
    alpha = drF/sqrt(rA)

    """
    xda = ds['drF'] / np.sqrt(ds['rA'])
    xda.name = 'alpha'
    return xda

def getF(mymask):
    """Return Jacobian of deformation tensor as a dict"""

    ndims = len(mymask.dims)
    if ndims==2:
        if set(('XC','YC')).issubset(mymask.dims):
            return {'ux':np.sqrt(mymask['rA']),
                    'vy':np.sqrt(mymask['rA'])}
        elif set(('YC','Z')).issubset(mymask.dims):
            return {'vy':np.sqrt(mymask['rA']).broadcast_like(mymask),
                    'wz':mymask['drF'].broadcast_like(mymask)}
        elif set(('XC','Z')).issubset(mymask.dims):
            return {'ux':np.sqrt(mymask['rA']).broadcast_like(mymask),
                    'wz':mymask['drF'].broadcast_like(mymask)}
        else:
            raise TypeError('getF dims problem 2d')
    elif ndims==3:
        L = np.sqrt(mymask['rA']).broadcast_like(mymask)
        return {'ux':1,
                'vy':1,
                'wz':mymask['drF']/L}
    else:
        raise TypeError('Only 2d or 3d for this phd')

def get_delta(Nx,determinant,mymask):
    ndims = len(mymask.dims)
    nu = 1/2 if ndims==3 else 1
    numer = 8*nu
    rho_hat_squared = Nx**2 if ndims==2 else (Nx * np.sqrt(mymask['rA']).mean())**2
    denom = rho_hat_squared * determinant
    xda = (numer/denom).broadcast_like(mymask)
    xda.name='delta'
    return xda

def get_matern(Nx,mymask):

    C = {}
    K = {}
    ndims = len(mymask.dims)
    F = getF(mymask)
    C['alpha'] = get_alpha(mymask.to_dataset(name='mask')).broadcast_like(mymask)
    C['gamma'] = 1e-5
    C['nu'] = 1/2
    C['Nx'] = Nx
    if ndims == 2:
        if set(('XC','YC')).issubset(mymask.dims):
            C['determinant'] = F['vy']*F['ux']
        elif set(('YC','Z')).issubset(mymask.dims):
            C['determinant'] = F['wz']*F['vy']
        elif set(('XC','Z')).issubset(mymask.dims):
            C['determinant'] = F['wz']*F['ux']
        else:
            raise TypeError('Help my dims out')
    else:
        C['determinant'] = F['wz']*F['vy']*F['ux']

    C['randNorm'] = 1/np.sqrt(C['determinant'])
    C['delta'] = get_delta(Nx,determinant=C['determinant'],mymask=mymask)

    if 'XC' in mymask.dims:
        K['ux'] = 1 / C['determinant'] * F['ux']*F['ux']
    if 'YC' in mymask.dims:
        K['vy'] = 1 / C['determinant'] * F['vy']*F['vy']
    if 'Z' in mymask.dims:
        K['wz'] = 1 / C['determinant'] * F['wz']*F['wz']

    for dd,lbl in zip([C,K,F],['constants','K','F']):
        for key,val in dd.items():
            if key != 'alpha':
                if isinstance(val,xr.core.dataarray.DataArray):
                    try:
                        assert val.dims == mymask.dims
                    except:
                        raise TypeError(f'dim order for {lbl}[{key}] is: ',val.dims)
    return C,K

def write_matern(write_dir,smoothOpNb,Nx,mymask,xdalike):
    """Write everything to describe the SPDE operator associated
    with the Matern covariance

    Parameters
    ----------
    write_dir : str
        path with directory to write to
    smoothOpNb : int
        smooth operator number
    Nx : int
        number of neighboring grid cells to smooth by...
    mymask : xarray DataArray
        defining the ControlField
    xdalike : xarray DataArray
        to write the fields like, since mymask may have a different
        ordering than what the MITgcm wants
    """

    ndims = len(mymask.dims)

    # Make the tensor and put into big array
    C,K = get_matern(Nx,mymask)

    # Write out the fields
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    dimstr = f'{ndims}D'

    for el in ['ux','vy','wz']:
        if el in K.keys():
            K[el] = K[el].reindex_like(xdalike)
            wrmds(f'{write_dir}/smooth{dimstr}K{el}{smoothOpNb:03}',
                    arr=K[el].values,
                    dataprec='float64')

    for f,fstr in zip(['delta','randNorm'],
                      ['Delta','RandNorm']):
        C[f] = C[f].reindex_like(xdalike)
        wrmds(f'{write_dir}/smooth{dimstr}{fstr}{smoothOpNb:03}',
              arr=C[f].values,
              dataprec='float64')

def get_matern_dataset(run_dir,smoothOpNb,xdalike,sample_num=None):
    
    ndims = len(xdalike.dims)
    smooth_mean = read_mds(f'{run_dir}/smooth{ndims}Dmean{smoothOpNb:03}',
                           xdalike=xdalike)
    smooth_norm = read_mds(f'{run_dir}/smooth{ndims}Dnorm{smoothOpNb:03}',
                          xdalike=xdalike)
    if sample_num is None:
        fld_fname = f'{run_dir}/smooth{ndims}Dfld{smoothOpNb:03}'
        smooth_fld = read_mds(fld_fname,xdalike=xdalike)
    else:
        if isinstance(sample_num,int):
            fld_fname = f'{run_dir}/smooth{ndims}Dfld{smoothOpNb:03}.{sample_num:04}'
            smooth_fld = read_mds(fld_fname,xdalike=xdalike)
        else:
            # add a dimension, sample number
            sample = xr.DataArray(sample_num,
                                  coords={'sample':sample_num},
                                  dims=('sample',),name='sample')
            smooth_fld = xr.zeros_like(sample*smooth_norm)
            for sn in sample_num:
                fld_fname = f'{run_dir}/smooth{ndims}Dfld{smoothOpNb:03}.{sn:04}'
                smooth_fld.loc[{'sample':sn}] = read_mds(fld_fname,xdalike=xdalike)

    names = ['ginv','filternorm','ginv_norm','ginv_nomean_norm']
    fldlist = [smooth_fld,smooth_norm,smooth_fld*smooth_norm,
               (smooth_fld-smooth_mean)*smooth_norm]
    labels = [r'$\mathcal{A}^{-1}g$',
              r'$\Lambda$',
              r'$\Lambda\mathcal{A}^{-1}g$',
              r'$\Lambda\mathcal{A}^{-1}(g-\bar{g}$']
    ds = xr.Dataset(dict(zip(names,fldlist)))
    for key,lbl in zip(ds.data_vars,labels):
        ds[key].attrs['label'] = lbl
    return ds
