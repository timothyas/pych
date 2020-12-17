"""
Some routines necessary for ice shelf domains
"""

import numpy as np
import xarray as xr
from MITgcmutils.jmd95 import densjmd95
from MITgcmutils.mdjwf import densmdjwf


def calc_phiHyd(ds, tRef, sRef, rhoConst=1030, eos='jmd95z'):
    """Compute Hydrostatic pressure component from density anomaly

    phiHydAnom(z) = \int_z^{\eta-h} g (rho(z) - rho_0)dz

    where h=iceshelf draft, \eta is free surface, rho(z) is the density
    profile from provided reference temperature and salinity, and the

    This is the integral term in Eqn. 8.27 in the shelfice documentation:
    https://mitgcm.readthedocs.io/en/latest/phys_pkgs/shelfice.html

    Parameters
    ----------
    ds : xarray Dataset
        containing grid information:
            rLowC, rSurfC
        reference temperature, salinity profiles:
            tRef, sRef
    rhoConst : float, optional
        reference density of seawater
    eos : str, optional
        Equation of state, currently only jmd95z and mdjwf implemented

    Returns
    -------
    phiHydAnomC, phiHydAnomF : xarray DataArray
        vertical profile of hydrostatic pressure from density anomaly
        at tracer location (C) and vertical cell interfaces (F)
    """

    phiHydAnomF = xr.DataArray(data=0.*ds['Zp1'].values,
                                coords=ds['Zp1'].coords,dims=ds['Zp1'].dims,
                                name='phiHydAnomF')
    phiHydAnomC = xr.DataArray(data=0.*ds['Z'].values,
                                coords=ds['Z'].coords,dims=ds['Z'].dims,
                                name='phiHydAnomC')

    gravity = 9.81
    p = 1e-4 * rhoConst * gravity * np.abs(ds['Z'])

    # half vertical cell distances
    dzlhalf = np.abs(ds['Zl'].values - ds['Z'].values)
    dzuhalf = np.abs(ds['Z'].values - ds['Zu'].values)

    # solver stuff
    dp = p
    tol = 1e-13

    if not (eos == 'jmd95z' or eos == 'mdjwf'):
        raise NotImplementedError('Only JMD95Z and MDJWF implemented...')
    density_function = densjmd95 if 'eos'=='jmd95z' else densmdjwf

    while np.sqrt(np.mean(dp**2)) > tol:
        p0 = p.copy()
        phiHydAnomF.loc[{'Zp1':ds.Zp1.isel(Zp1=0)}] = 0.
        for k in np.arange(len(ds['Z'])):
            drho = density_function(sRef.isel(Z=k),tRef.isel(Z=k),p.isel(Z=k)) - rhoConst
            phiHydAnomC.loc[{'Z':ds['Z'].isel(Z=k)}] = \
                phiHydAnomF.isel(Zp1=k).values + dzlhalf[k]*gravity*drho

            phiHydAnomF.loc[{'Zp1':ds['Zp1'].isel(Zp1=k+1)}] = \
                phiHydAnomC.isel(Z=k).values + dzuhalf[k]*gravity*drho

        if 'eos'=='mdjwf':
            p = (gravity*rhoConst*abs(ds['Z']) + phiHydAnomC)/gravity/rhoConst
        dp = p-p0

    return phiHydAnomC, phiHydAnomF

def calc_phi0surf(phiHydAnomF,icetopo,ds,grid):
    """Compute 2D field in x-y plane phi0surf from phiHydAnom below the iceshelf
    """

    maskC = ds['maskC'].any('Z')
    phi0surf = xr.DataArray(data=0.*maskC.values,
                            coords=maskC.coords,
                            dims=maskC.dims,
                            name='phi0surf')



    maskZ = get_3d_mask('Zl',icetopo,ds,grid).swap_dims({'Zl':'Z'})
    maskZp1 = get_3d_mask('Zp1',icetopo,ds,grid)

    # fields to interpolate pressure based on partial cell
    dPhi = grid.diff(phiHydAnomF,'Z',to='center')
    drLoc = (1 - ds.hFacC)*maskZ

    phi0surf = (phiHydAnomF*maskZp1).sum('Zp1') + \
            (drLoc*dPhi).sum('Z').where(icetopo!=0, 0.)

    return phi0surf

def get_3d_mask(zfld,fld2d,ds,grid):
    """make a 3D mask for first vertical wet point
    """

    Z3D, fld3D = xr.broadcast(ds[zfld],fld2d)
    zdiff = Z3D - fld3D
    if zfld == 'Zl':
        dZ = np.abs(grid.diff(ds.Z,'Z',to='left',boundary='fill'))
    elif zfld == 'Zu':
        dZ = np.abs(grid.diff(ds.Z,'Z',to='right',boundary='fill'))
    elif zfld == 'Z':
        dZ = ds.drF
    else:
        dZ = ds.drC

    return ((zdiff >=0) & (zdiff<=dZ))
