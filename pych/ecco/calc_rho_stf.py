"""
Module for computing meridional transport of quantities in density space
This is still in development...
"""

import numpy as np
import xarray as xr
# xarray compatibility
try:
    from xarray.core.pycompat import OrderedDict
except ImportError:
    from collections import OrderedDict

from ecco_v4_py.ecco_utils import get_basin_mask, get_llc_grid
from ecco_v4_py.vector_calc import get_latitude_masks
from ecco_v4_py.calc_section_trsp import get_section_line_masks, _parse_section_trsp_inputs

# Define constants
METERS_CUBED_TO_SVERDRUPS = 10**-6
WATTS_TO_PETAWATTS = 10**-15
RHO_CONST = 1000
HEAT_CAPACITY = 4000

# -------------------------------------------------------------
# Meridional stf 
# --------------------------------------------------------------

def calc_rho_moc(ds,lat_vals,doFlip=True,basin_name=None,grid=None):
    """Compute the meridional overturning streamfunction in Sverdrups 
    at specified latitude(s) in density space

    see ecco_v4_py.calc_stf.calc_meridional_stf for inputs, 
    all others are the same except

    Parameters
    ----------
    ds : xarray DataSet
        must contain UVELMASS,VVELMASS, RHOAnoma, drF, dyG, dxG

    Returns
    -------
    psi_ds : xarray Dataset
        Contains the DataArray variable 'psi_moc' which contains the
        streamfunction at each denoted latitude band at
        the center of each density bin level with dimensions 
        'time' (if in given dataset), 'k_rho' (density index with 
        values in 'rho_c'), and 'lat' 

        Additionally, the density bin edges are given as 'rho_f' with index 'k_rho_f'
    """

    # Compute volume transport
    trsp_x = ds['UVELMASS'] * ds['drF'] * ds['dyG']
    trsp_y = ds['VVELMASS'] * ds['drF'] * ds['dxG']

    # Get density
    rho = ds['RHOAnoma'] + 1029.

    # Creates an empty streamfunction
    psi_ds = meridional_trsp_at_rho(trsp_x, trsp_y, rho,
                                    lat_vals=lat_vals, 
                                    cds=ds.coords.to_dataset(), 
                                    basin_name=basin_name, 
                                    grid=grid)
    psi_ds = psi_ds.rename({'trsp':'psi_moc'})

    # Flip depth dimension, take cumulative sum, flip back
    if doFlip:
        psi_ds['psi_moc'] = psi_ds['psi_moc'].isel(k_rho=slice(None,None,-1))

    # Should this be done with a grid object??? 
    psi_ds['psi_moc'] = psi_ds['psi_moc'].cumsum(dim='k_rho')
    
    if doFlip:
        psi_ds['psi_moc'] = -1 * psi_ds['psi_moc'].isel(k_rho=slice(None,None,-1))

    # Convert to Sverdrups
    psi_ds['psi_moc'] = psi_ds['psi_moc'] * METERS_CUBED_TO_SVERDRUPS
    psi_ds['psi_moc'].attrs['units'] = 'Sv'

    return psi_ds


def meridional_trsp_at_rho(ufld, vfld, rho, lat_vals, cds, 
                           basin_name=None, grid=None):
    """
    Compute transport of vector quantity in density space 
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    ufld, vfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edges
    rho : xarray DataArray
        3D spatial (+ time, optional) field at cell center with density
    lat_vals : int or array of ints
        latitude value(s) rounded to the nearest degree
        specifying where to compute transport
    cds : xarray Dataset
        with all LLC90 coordinates, including: maskW/S, YC
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see utils.get_available_basin_names for options
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see utils.get_llc_grid

    Returns
    -------
    trsp_ds : xarray Dataset
        Contains the DataArray variable 'trsp' which contains the
        transport of vector quantity across denoted latitude band at
        each the center of each density bin level with dimensions 
        'time' (if in given dataset), 'k_rho' (density index with 
        values in 'rho_c'), and 'lat' 

        Additionally, the density bin edges are given as 'rho_f' with index 'k_rho_f'

    """

    if grid is None:
        grid = get_llc_grid(cds)

    if np.isscalar(lat_vals):
        lat_vals = [lat_vals]

    # Initialize empty DataArray with coordinates and dims
    trsp_ds = _initialize_rho_trsp_dataset(cds, rho, lat_vals)

    # Compute density on west and south grid cell faces
    rho_w = grid.interp(rho, 'X', boundary='fill')
    rho_s = grid.interp(rho, 'Y', boundary='fill')

    # Get basin mask
    if basin_name is not None:
        basin_maskW = get_basin_mask(basin_name,cds['maskW'].isel(k=0))
        basin_maskS = get_basin_mask(basin_name,cds['maskS'].isel(k=0))
    else:
        basin_maskW = cds['maskW'].isel(k=0)
        basin_maskS = cds['maskS'].isel(k=0)

    for lat in lat_vals:

        # Compute mask for particular latitude band
        lat_maskW, lat_maskS = get_latitude_masks(lat, cds['YC'], grid)

        trsp_ds = _calc_trsp_at_density_levels(trsp_ds, rho_w, rho_s, 
                                               ufld * lat_maskW * basin_maskW, 
                                               vfld * lat_maskS * basin_maskS,lat=lat) 

    return trsp_ds

# -------------------------------------------------------------
# Section stf (e.g. for OSNAP)
# --------------------------------------------------------------

def calc_rho_section_stf(ds, 
                         pt1=None, pt2=None, 
                         section_name=None,
                         maskW=None, maskS=None,
                         doFlip=True,grid=None):
    """Compute the overturning streamfunction in plane normal to section 
    defined by pt1 and pt2 in density space

    See ecco_v4_py.calc_section_trsp.calc_section_vol_trsp for the various 
    ways to call this function 
    All inputs are the same except:

    Parameters
    ----------
    ds : xarray DataSet
        must contain UVELMASS,VVELMASS, RHOAnoma, drF, dyG, dxG

    Returns
    -------
    psi_ds : xarray Dataset
        Contains the DataArray variable 'psi_ov' which contains the
        streamfunction at each denoted latitude band at
        the center of each density bin level with dimensions 
        'time' (if in given dataset), 'k_rho' (density index with 
        values in 'rho_c'), and 'lat' 

        Additionally, the density bin edges are given as 'rho_f' with index 'k_rho_f'
    """

    # Compute volume transport
    trsp_x = ds['UVELMASS'] * ds['drF'] * ds['dyG']
    trsp_y = ds['VVELMASS'] * ds['drF'] * ds['dxG']

    # Get density
    rho = ds['RHOAnoma'] + 1029.

    maskW, maskS = _parse_section_trsp_inputs(ds,pt1,pt2,maskW,maskS,section_name)

    # Creates an empty streamfunction
    psi_ds = section_trsp_at_rho(trsp_x, trsp_y, rho,
                                  maskW, maskS, 
                                  cds=ds.coords.to_dataset(), 
                                  grid=grid)

    psi_ds = psi_ds.rename({'trsp':'psi_ov'})

    # Flip depth dimension, take cumulative sum, flip back
    if doFlip:
        psi_ds['psi_ov'] = psi_ds['psi_ov'].isel(k_rho=slice(None,None,-1))

    # Should this be done with a grid object??? 
    psi_ds['psi_ov'] = psi_ds['psi_ov'].cumsum(dim='k_rho')
    
    if doFlip:
        psi_ds['psi_ov'] = -1 * psi_ds['psi_ov'].isel(k_rho=slice(None,None,-1))

    # Convert to Sverdrups
    psi_ds['psi_ov'] = psi_ds['psi_ov'] * METERS_CUBED_TO_SVERDRUPS
    psi_ds['psi_ov'].attrs['units'] = 'Sv'

    return psi_ds

def section_trsp_at_rho(ufld, vfld, rho, maskW, maskS, cds, 
                        grid=None):
    """
    Compute transport of vector quantity in density space 
    across section defined by maskW, maskS

    Parameters
    ----------
    ufld, vfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edges
    rho : xarray DataArray
        3D spatial (+ time, optional) field at cell center with density
    maskW, maskS : xarray DataArray
        defines the section to define transport across
    cds : xarray Dataset
        with all LLC90 coordinates, including: maskW/S, YC
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see utils.get_llc_grid

    Returns
    -------
    trsp_ds : xarray Dataset
        Contains the DataArray variable 'trsp' which contains the
        transport of vector quantity across denoted latitude band at
        each the center of each density bin level with dimensions 
        'time' (if in given dataset) and 'rho_c' (with index k_rho)

        Additionally, the density bin edges are given as 'rho_f' with index 'k_rho_f'

    """

    if grid is None:
        grid = get_llc_grid(cds)

    # Initialize empty DataArray with coordinates and dims
    trsp_ds = _initialize_rho_trsp_dataset(cds, rho)

    # Compute density on west and south grid cell faces
    rho_w = grid.interp(rho, 'X', boundary='fill')
    rho_s = grid.interp(rho, 'Y', boundary='fill')

    trsp_ds = _calc_trsp_at_density_levels(trsp_ds, rho_w, rho_s, 
                                           ufld * maskW, 
                                           vfld * maskS) 

    return trsp_ds

# -------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------

def _calc_trsp_at_density_levels(trsp_ds, rho_w, rho_s, ufld, vfld, lat=None):
    """Helper function to compute transport at binned density levels 
    given the maskW and maskS. Extracted here to do the same operations for 
    latitude or section masks

    Parameters
    ----------
    trsp_ds : xarray Dataset
        initialized with _initialize_trsp_rho_dataset
    rho_w, rho_s : xarray DataArray
        3D (+ time, optional) field with density at 
        west and south grid cell edges
    ufld, vfld : xarray DataArray
        3D (+ time, optional) field with transport across 
        west and south grid cell edges
    lat : int, optional
        if computing across latitude band, provide value for accessing
        the dataset

    Returns
    -------
    trsp_ds : xarray Dataset
        return the input transport dataset with 'trsp' DataArray
        filled with total transport across that section or latitude band
        at each density level
    """

    # Determine density level and sum
    for kr in trsp_ds['k_rho'].values:

        if kr == 0:
            # First bin
            bin_condition_w = rho_w < trsp_ds['rho_f'][kr+1].values
            bin_condition_s = rho_s < trsp_ds['rho_f'][kr+1].values

        elif kr == trsp_ds['k_rho'].values[-1]:
            # Last bin
            bin_condition_w = rho_w >= trsp_ds['rho_f'][kr].values
            bin_condition_s = rho_s >= trsp_ds['rho_f'][kr].values

        else:
            # all others
            bin_condition_w = (rho_w <  trsp_ds['rho_f'][kr+1].values) & \
                              (rho_w >= trsp_ds['rho_f'][kr].values)
            bin_condition_s = (rho_s <  trsp_ds['rho_f'][kr+1].values) & \
                              (rho_s >= trsp_ds['rho_f'][kr].values)
            
        # Compute transport within this density bin
        trsp_x = ufld.where( 
                        bin_condition_w,0).sum(dim=['i_g','j','tile','k'])
        trsp_y = vfld.where( 
                        bin_condition_s,0).sum(dim=['i','j_g','tile','k'])

        if lat is not None:
            trsp_ds['trsp'].loc[{'lat':lat,'k_rho':kr}] = trsp_x + trsp_y
        else:
            trsp_ds['trsp'].loc[{'k_rho':kr}] = trsp_x + trsp_y

    return trsp_ds


def _initialize_rho_trsp_dataset(cds, rho, lat_vals=None):
    """Create an xarray Dataset with time, depth, and latitude dims

    Parameters
    ----------
    ds : xarray Dataset
        Must contain the coordinates 'k' and (optionally) 'time'
    rho : xarray DataArray
        Containing the density field to be binned and made into our new vertical coordinate
    lat_vals : int or array of ints, optional
        latitude value(s) rounded to the nearest degree
        specifying where to compute transport

    Returns
    -------
    ds : xarray Dataset
        zero-valued Dataset with time, depth, and latitude dimensions
    """

    # Create density bins
    rho_bin_edges, rho_bin_centers = get_rho_bins(rho.min().values, rho.max().values,
                                                  len(cds['k']))
    Nrho = len(rho_bin_centers)
    k_rho = np.arange(Nrho)
    k_rho_f = np.arange(len(rho_bin_edges))

    coords = OrderedDict()
    dims = ()

    if 'time' in cds:
        coords.update( {'time': cds['time'].values} )
        dims += ('time',)
        if lat_vals is not None:
            zeros = np.zeros((len(cds['time'].values),
                              Nrho,
                              len(lat_vals)))
        else:
            zeros = np.zeros((len(cds['time'].values),
                              Nrho))
    else:
        if lat_vals is not None:
            zeros = np.zeros((Nrho,
                              len(lat_vals)))
        else:
            zeros = np.zeros((Nrho))

    coords.update( {'k_rho': k_rho} )
    dims += ('k_rho',)
    if lat_vals is not None:
        coords.update( {'lat': lat_vals} )
        dims += ('lat',)

    da = xr.DataArray(data=zeros, coords=coords, dims=dims)

    # This could be much cleaner, and should mirror the 
    # xgcm notation. 
    ds = da.to_dataset(name='trsp')
    ds['rho_c'] = rho_bin_centers
    ds['rho_f'] = rho_bin_edges
    ds['k_rho_f'] = k_rho_f

    return ds

def get_rho_bins(rho_min,rho_max,nbins=50):
    """Compute linearly spaced density bins between rho_min and rho_max 

    Parameters
    ----------

    rho_min, rho_max : float
        indicate the minimum and maximum density values to consider for binning
    nbins : int, optional
        number of density bin centers to include

    Returns
    -------
    bin_edges : np.ndarray
        1d array with density bin edges
    bin_centers : np.ndarray
        1d array with density bin centers
    """

    bin_edges = np.linspace(rho_min,rho_max,nbins+1)
    bin_centers = np.diff(bin_edges)/2 + bin_edges[0:len(bin_edges)-1]

    return bin_edges, bin_centers
