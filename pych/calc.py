#!/workspace/anaconda3/bin/python

"""
Some quick computation functions, some in progress...

    haversine - distance btwn two points using Haversine formula
    calc_vertical_avg - compute average in z dimension 
    calc_baro_stf - compute barotropic streamfunction 
    calc_vel_at_mxl - compute 2D velocity field at mixed layer depth

"""

import numpy as np
import xarray as xr
import xgcm 

def haversine(lon1, lat1, lon2, lat2):
    """Use Haversine formula to compute great circle distance
    In kilometers

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        longitude and latitude in degrees of first and second points

    Returns
    -------
    distance : float
        between the two points along the great arc between them
        in km
    """

    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    r = 6371 #km
    h = (np.sin(dlat / 2)**2) + np.cos(lat1)*np.cos(lat2)* (np.sin(dlon/2)**2)
    return 2*r*np.arcsin(np.sqrt(h))
    

def calc_vertical_avg(fld,msk):
    """Compute vertical average, ignoring continental or iceshelf points """
    
    # Make mask of nans, assume input msk is 3D of same size as fld 3 spatial dims
    nanmsk = np.where(msk==1,1,np.NAN)
    v_avg = fld.copy()
    v_avg.values = v_avg.values*msk.values
    
    if 'Z' in fld.dims:
        vdim = 'Z'
        
    elif 'Zl' in fld.dims:
        vdim = 'Zl'
        
    else:
        raise TypeError('Could not find recognizable vertical field in input dataset')
        
    # Once vertical coordinate is found, compute avg along dimension
    v_avg = v_avg.sum(dim=vdim,skipna=True)
    
    return v_avg

def calc_baro_stf(ds,grid):
    """
    Compute barotropic streamfunction as

        u_bt = - d\psi / dy
        \psi = \int_{-90}^{90} u_bt dy

    Parameters
    ----------
    ds          :   xarray Dataset from MITgcm output, via 
                    e.g. xmitgcm.open_mdsdataset
                    must contain 'U' or 'UVELMASS' fields
    grid        :   xgcm grid object defined via xgcm.Grid(ds)

    Output
    ------
    baro_stf    :   xarray DataArray containing 2D field with 
                    barotropic streamfunction in Sv, \psi above
    """


    # Grab the right velocity field from dataset
    if 'U' in ds.keys():
        ustr = 'U'
    elif 'UVELMASS' in ds.keys():
        ustr = 'UVELMASS'
    else:
        raise TypeError('Could not find recognizable velocity field in input dataset')

    # Define barotropic velocity as vertically integrated velocity
    if ustr == 'UVELMASS':
        u_bt = (ds[ustr] * ds['drF']).sum(dim='Z')
    else:
        u_bt = (ds[ustr] * ds['hFacW'] * ds['drF']).sum(dim='Z')

    u_bt = u_bt * ds['dyG']

    # Integrate in Y
    baro_stf = grid.cumsum(-u_bt,'Y',boundary='fill',fill_value=0.)

    # Convert m/s to Sv
    baro_stf = baro_stf * 10**-6

    return baro_stf

def calc_overturning_stf(ds,grid,doFlip=True):
    """
    Only for simple domains, compute meridional overturning streamfunction

    Parameters
    ----------
    ds          :   xarray Dataset from MITgcm output, via 
                    e.g. xmitgcm.open_mdsdataset
                    must contain 'V' or 'VVELMASS' fields
    grid        :   xgcm grid object defined via xgcm.Grid(ds)

    doFlip      :   if true, compute by accumulating from bottom to top

    Output
    ------
    ov_stf      :   xarray DataArray containing 2D field with 
                    overturning streamfunction in Sv above
    """

    # Grab the right velocity field from dataset
    if 'V' in ds.keys():
        vstr = 'V'
    elif 'VVELMASS' in ds.keys():
        vstr = 'VVELMASS'
    else:
        raise TypeError('Could not find recognizable velocity field in input dataset')

    # Compute volumetric transport
    v_trsp = ds[vstr] * ds['dxG'] * ds['drF']
    if vstr != 'VVELMASS':
        print(f' *** Multiplying {vstr} by hFacS***')
        v_trsp = v_trsp * ds['hFacS']

    v_trsp = v_trsp.sum(dim=['XC'])

    # flip dim, accumulate in vertical, flip back
    if doFlip:
        v_trsp = v_trsp.isel(Z=slice(None,None,-1))

    ov_stf = grid.cumsum(v_trsp,'Z',boundary='fill')

    if doFlip:
        ov_stf = -ov_stf.isel(Zl=slice(None,None,-1))

    # Convert m/s to Sv
    ov_stf = ov_stf * 10**-6

    return ov_stf
    


def calc_vel_at_mxl(ds):
    """
    Compute velocity components at mixed layer depth

    Parameters
    ----------
    ds      ::  xarray Dataset which must contain 
                UVELMASS, VVELMASS, MXLDEPTH

    Output
    ------
    new_ds  ::  same dataset with new fields w/ velocity at mxl depth
                u_m, v_m  
    """

    necessary_fields = ['UVELMASS','VVELMASS','MXLDEPTH','hFacC']
    for name in necessary_fields:
        if name not in ds.keys():
            raise AttributeError('Could not find field %s in input dataset' % name)

    
    # Make 3D fields with depth and cell height  
    #z_depth = ds['hFacC'].copy().values
    #drf     = ds['hFacC'].copy().values

    # Make potentially 4D field for MXLDEPTH copied over depth
    nTime = len(ds['MXLDEPTH'].time)
    #mxld    = np.tile(ds['hFacC'].copy().values, [nTime, 1, 1, 1])
    mxl_mask= 0*np.tile(ds['hFacC'].copy().values, [nTime, 1, 1, 1])
    
    for k in np.arange(len(ds['Z'])):
        #z_depth[k,:,:]  = ds['Z'][k]
        #drf[k,:,:]      = ds['drF'][k]

        for n in np.arange(nTime):
            mxl_mask[n,k,:,:] = np.where(
                    (-ds['MXLDEPTH'][n,:,:] >= ds['Zu'][k]) & 
                    (-ds['MXLDEPTH'][n,:,:] <  ds['Zl'][k]),1,0)


              
    return mxl_mask
