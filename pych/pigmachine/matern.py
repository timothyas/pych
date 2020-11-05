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
from scipy.special import gamma

from .io import read_mds

def calc_variance(Nx,ndims=2):
    nu = 1/2 if ndims==3 else 1
    delta_hat = 8*nu / (Nx**2)
    denom = gamma(nu+ndims/2)*((4*np.pi)**(ndims/2))*(delta_hat**(nu))
    return gamma(nu)/denom

def _getL(ds):
    """get horizontal length scale"""
    if isinstance(ds,xr.core.dataarray.DataArray):
        ds = ds.to_dataset(name='tmp')

    if 'rA' in ds:
        L = np.sqrt(ds['rA'])
    elif 'dyG' in ds:
        L = ds['dyG']
    elif 'dxG' in ds:
        L = ds['dxG']
    else:
        raise NotImplementedError('Other length scales not recognized')
    return L

def get_alpha(ds):
    """Return the grid-define aspect ratio
    alpha = drF/sqrt(rA)

    """
    L = _getL(ds)
    xda = ds['drF'] / L
    xda.name = 'alpha'
    return xda

def getPhi(mymask,xi=1):
    """Return Jacobian of deformation tensor as a dict

              ux   0   0
        Phi = 0   vy   0
              0    0  wz

    or a 2D section of that...

    xi is an additional factor on ux and/or vy
    to accentuate the horizontal scales over the vertical

    xi must be same size as mymask, or just a scalar

    """

    ndims = len(mymask.dims)

    L = _getL(mymask).broadcast_like(mymask)
    H = mymask['drF'].broadcast_like(mymask) if 'drF' in mymask.coords else xr.ones_like(mymask)

    xi = xi*xr.ones_like(mymask)

    ux = xi*L
    vy = xi*L
    wz = H

    if ndims==2:
        if set(('XC','YC')).issubset(mymask.dims):
            return {'ux':ux/xi,'vy':vy/xi}
        elif set(('YC','Z')).issubset(mymask.dims):
            return {'vy':vy,'wz':wz}
        elif set(('XC','Z')).issubset(mymask.dims):
            return {'ux':ux,'wz':wz}
        else:
            raise TypeError('getPhi dims problem 2d')
    elif ndims==3:
        return {'ux':ux,'vy':vy,'wz':wz}
    else:
        raise TypeError('Only 2d or 3d for this phd')

def get_delta(Nx,determinant,mymask):
    ndims = len(mymask.dims)
    nu = 1/2 if ndims==3 else 1
    numer = 8*nu
    Nx_hat_squared = Nx**2
    denom = Nx_hat_squared * determinant
    xda = (numer/denom).broadcast_like(mymask)
    xda.name='delta'
    return xda

def get_cell_volume(mymask):
    """return the cell volume as part of the normalization factor
    for the white noise process"""

    ndims = len(mymask.dims)
    L = _getL(mymask)
    if ndims==2:
        if set(('XC','YC')).issubset(mymask.dims):
            return L**2
        elif set(('YC','Z')).issubset(mymask.dims):
            return mymask['drF']*L
        elif set(('XC','Z')).issubset(mymask.dims):
            return mymask['drF']*L
    else:
        return mymask['drF']*L**2

def get_matern(Nx,mymask,xi=1):

    C = {}
    K = {}
    ndims = len(mymask.dims)
    Phi = getPhi(mymask,xi=xi)
    C['alpha'] = get_alpha(mymask.to_dataset(name='mask')).broadcast_like(mymask)
    C['Nx'] = Nx
    if ndims == 2:
        if set(('XC','YC')).issubset(mymask.dims):
            C['determinant'] = Phi['vy']*Phi['ux']
        elif set(('YC','Z')).issubset(mymask.dims):
            C['determinant'] = Phi['wz']*Phi['vy']
        elif set(('XC','Z')).issubset(mymask.dims):
            C['determinant'] = Phi['wz']*Phi['ux']
        else:
            raise TypeError('Help my dims out')
    else:
        C['determinant'] = Phi['wz']*Phi['vy']*Phi['ux']

    C['randNorm'] = 1/np.sqrt(C['determinant'])/np.sqrt(get_cell_volume(mymask))
    C['delta'] = get_delta(Nx,determinant=C['determinant'],mymask=mymask)

    if 'XC' in mymask.dims:
        K['ux'] = 1 / C['determinant'] * Phi['ux']*Phi['ux']
    if 'YC' in mymask.dims:
        K['vy'] = 1 / C['determinant'] * Phi['vy']*Phi['vy']
    if 'Z' in mymask.dims:
        K['wz'] = 1 / C['determinant'] * Phi['wz']*Phi['wz']

    for dd,lbl in zip([C,K,Phi],['constants','K','Phi']):
        for key,val in dd.items():
            if key != 'alpha':
                if isinstance(val,xr.core.dataarray.DataArray):
                    try:
                        assert val.dims == mymask.dims
                    except:
                        raise TypeError(f'dim order for {lbl}[{key}] is: ',val.dims)
    return C,K

def write_matern(write_dir,smoothOpNb,Nx,mymask,xdalike,xi=1):
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
    C,K = get_matern(Nx,mymask,xi=xi)

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

def get_matern_dataset(run_dir,smoothOpNb,xdalike,sample_num=None,
                       read_filternorm=True):

    ndims = len(xdalike.dims)
    if read_filternorm:
        smooth_mean = read_mds(f'{run_dir}/smooth{ndims}Dmean{smoothOpNb:03}',
                               xdalike=xdalike)
        smooth_norm = read_mds(f'{run_dir}/smooth{ndims}Dnorm{smoothOpNb:03}',
                              xdalike=xdalike)
    fld_fname = f'{run_dir}/smooth{ndims}Dfld{smoothOpNb:03}'
    if sample_num is None:
        smooth_fld = read_mds(fld_fname,xdalike=xdalike)
    else:

        if isinstance(sample_num,list):
            sample_num = np.array(sample_num)
        elif isinstance(sample_num,int):
            sample_num = np.array([sample_num])

        sample = xr.DataArray(sample_num,
                              coords={'sample':sample_num},
                              dims=('sample',),name='sample')
        smooth_fld = read_mds(fld_fname,xdalike=(sample*xdalike).squeeze(),
                              rec=sample_num)

    if read_filternorm:
        names = ['ginv','filternorm','ginv_norm','ginv_nomean_norm']
        fldlist = [smooth_fld,smooth_norm,smooth_fld*smooth_norm,
                   (smooth_fld-smooth_mean)*smooth_norm]
        labels = [r'$\mathcal{A}^{-1}g$',
                  r'$X$',
                  r'$X\mathcal{A}^{-1}g$',
                  r'$X\mathcal{A}^{-1}(g-\bar{g}$']
    else:
        names = ['ginv']
        fldlist = [smooth_fld]
        labels = [r'$\mathcal{A}^{-1}g$']

    ds = xr.Dataset(dict(zip(names,fldlist)))
    for key,lbl in zip(ds.data_vars,labels):
        ds[key].attrs['label'] = lbl
    return ds
