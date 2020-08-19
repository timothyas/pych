"""
Functions related to Equatorial Under Current in ECCO
"""

import numpy as np
import ecco_v4_py as ecco

def get_euc_masks(ds,pt1,pt2,grid):
    """Get masks defining EUC region

    Parameters
    ----------
    see calc_euc

    Returns
    -------
    maskW, maskS : xarray DataArray
        defining W/S components
    """
    return ecco.calc_section_trsp._parse_section_trsp_inputs(ds,pt1,pt2,None,None,'EUC')

def calc_euc(ds,pt1,pt2,grid):
    """Compute the Equatorial Under Current at 
    a specific "transect" defined by pt1,pt2.
    
    For positive eastward, pt1 latitude >0, pt2 latitude<0
    
    Compute as fields in output dataset:
    	- 'trsp': 
    		EUC transport, compute from (assumed) time mean velocity fields
    		= integrated velocity to ~400m depth (21 depth levels in ECCO)
    		across section defined by pt1->pt2
    	- 'maskW/S': 
    		masks defining the transect
    	- 'Z_core': 
    		depth of maximum transport, not limited to 400m depth
    	- 'u_eq'/'v_eq':
    		zonal velocity interpolated (first order) to equator
    		one field should be empty, and the other has zonal velocity
    		depending on the LLC face
    	- 'Zl_upper'/'Zl_lower':
    		upper and lower bounds of the EUC core, computed as the depth level
    		where U~0, via first order interpolation
    
    Parameters
    ----------
    ds : xarray dataset
    	defining the grid
    pt1, pt2 : list or tuple with 2 elements
    	defining [lon, lat] for section transport
    grid : xgcm Grid
    	defining the grid operations
    
    Returns
    -------
    trsp_ds : xarray dataset
    	with all the fields listed above
    """
    
    # --- initialize output container
    trsp_ds = ecco.calc_section_trsp._initialize_section_trsp_data_array(ds)
    
    # --- Compute transport
    # get transport masks 
    maskW,maskS = get_euc_masks(ds,pt1,pt2,grid)
    
    # Get surface area for transport
    area_x = ds['drF']*ds['dyG']*maskW
    area_y = ds['drF']*ds['dxG']*maskS
    
    # Get volumetric transport at grid cell
    xvol = ds['UVELMASS']*area_x
    yvol = ds['VVELMASS']*area_y
    
    # Integrate, only consider U>0
    # Note: I know this condition is OK because after multiplying by the masks above,
    # both xvol and yvol should be >0 for U>0 on the tiles in consideration here
    trsp_x = xvol.where(ds.Z>-400).where(xvol>0).sum(dim=['i_g','j','tile','k'])
    trsp_y = yvol.where(ds.Z>-400).where(yvol>0).sum(dim=['i','j_g','tile','k'])
    trsp_ds['trsp'] = (trsp_x + trsp_y)* (10**-6)
    trsp_ds['trsp'].attrs['units'] = 'Sv'
    trsp_ds['maskW'] = maskW
    trsp_ds['maskS'] = maskS
    
    # --- Get depth where velocity is ~0
    # get latitude interpolated to velocity grid point
    yW = grid.interp(ds.YC,'X',boundary='fill')
    yS = grid.interp(ds.YC,'Y',boundary='fill')
    
    # get velocity -vs- depth at this longitude
    u_eq = ds.UVELMASS.where(( np.abs(yW)<0.5) & maskW,drop=True).mean('j').squeeze()
    v_eq = ds.VVELMASS.where(( np.abs(yS)<0.5) & maskS,drop=True).mean('i').squeeze()
    
    # --- Get depth of maximum zonal velocity
    zu_core = ds.Z.where(u_eq==u_eq.max('k'))
    zv_core = ds.Z.where(v_eq==v_eq.max('k'))
    
    if not zv_core.any():
        if not zu_core.any():
            z_core=np.nan
        else:
            z_core = zu_core
    else:
        z_core = zv_core
    
    trsp_ds['Z_core'] = z_core.dropna('k')[0]
    
    trsp_ds['u_eq'] = u_eq
    trsp_ds['v_eq'] = v_eq
    
    # get locations where sign of velocity changes
    sgn_u = u_eq*u_eq.shift(k=1,fill_value=np.nan)
    sgn_v = v_eq*v_eq.shift(k=1,fill_value=np.nan)
    
    # and interpolate depth from this
    # get where lower velocity is greater than zero for top of the core
    u_frac = -u_eq/(u_eq.shift(k=1,fill_value=np.nan)-u_eq)
    v_frac = -v_eq/(v_eq.shift(k=1,fill_value=np.nan)-v_eq)
    z_u_upper = (ds.Z+(u_frac*(ds.Z.shift(k=1,fill_value=np.nan)-ds.Z))).where((sgn_u<=0) & (u_eq>=0)).dropna('k')
    z_v_upper = (ds.Z+(v_frac*(ds.Z.shift(k=1,fill_value=np.nan)-ds.Z))).where((sgn_v<=0) & (v_eq>=0)).dropna('k')
           
    # get where lower velocity is less than zero for bottom of the core
    u_frac = u_eq.shift(k=1)/(u_eq.shift(k=1,fill_value=np.nan)-u_eq)
    v_frac = v_eq.shift(k=1)/(v_eq.shift(k=1,fill_value=np.nan)-v_eq)
    z_u_lower = (ds.Z.shift(k=1,fill_value=np.nan)-(u_frac*(ds.Z.shift(k=1,fill_value=np.nan)-ds.Z))).where((sgn_u<=0) & (u_eq<=0)).dropna('k')
    z_v_lower = (ds.Z.shift(k=1,fill_value=np.nan)-(v_frac*(ds.Z.shift(k=1,fill_value=np.nan)-ds.Z))).where((sgn_v<=0) & (v_eq<=0)).dropna('k')
    
    Zl_upper = z_u_upper if not z_v_upper.any() else z_v_upper #z_u[0] if not z_v[0] else z_v[0]
    Zl_lower = z_u_lower if not z_v_lower.any() else z_v_lower #z_u[1] if not z_v[1] else z_v[1]
    
    # get lower and upper as Z values that bound Z_core
    trsp_ds['Zl_lower'] = Zl_lower.where(Zl_lower<float(trsp_ds['Z_core']),drop=True)[0]
    trsp_ds['Zl_upper'] = Zl_upper.where(Zl_upper>float(trsp_ds['Z_core']),drop=True)[-1]
    
    return trsp_ds
