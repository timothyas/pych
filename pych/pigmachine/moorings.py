"""
Routines to create a synthetic mooring based observation network
"""
import xarray as xr
from rosypig import Observable

def make_mooring_obs(ds,
                     name='theta',avg_period='month',
                     barfile='m_theta',
                     uncertainty_filename='sigma_theta',
                     mult_filename='mult_theta'):
    """
    Make a mooring array observable object as a test case
    See make_mooring_array_mask for the observation locations

    Parameters
    ----------
    ds : xarray Dataset
        containing the grid information

    Returns
    -------
    mooring_obs : rosypig Observable
        Observable object with mooring locations for temperature
    """

    mooring_obs = Observable(name,grid=ds,
                             avg_period=avg_period,barfile=barfile,
                             uncertainty_filename=uncertainty_filename,
                             mult_filename=mult_filename)

    mooring_array_mask = make_mooring_array_mask(ds)
    lon,lat,depth = get_loc_from_mask(ds,mooring_array_mask)
    for x,y,z in zip(lon,lat,depth):
        mooring_obs.add_loc_to_masks(x,y,z)
    return mooring_obs
            

def make_mooring_array_mask(ds):
    """make a 3D mask which defines mooring locations
    This took some iterations to get something that "looks good"
    just to test some linear algebra routines

    Parameters
    ----------
    ds : xarray Dataset
        containing the grid information

    Returns
    -------
    maskC : xarray DataArray
        3D mask denoting mooring locations
    """

    # make a 2D array of surface
    moorings = False*ds.maskC.isel(Z=0)

    # get a "spread out" array, this took some iterations
    mv = moorings.values.flatten()
    mv[slice(0,None,108)]=True

    moorings = xr.DataArray(np.reshape(mv,moorings.shape),coords=ds.Depth.coords,dims=ds.Depth.dims)
    moorings = moorings.where(ds.maskC.isel(Z=0),False).where(ds.maskInC,False)

    # a guess at depths
    depths = np.arange(-250,-850,-100)

    maskZ = ds.Z != ds.Z

    for d in depths:
            maskZ = maskZ + (ds.Z == ds.Z.sel(Z=d,method='nearest'))

    return (maskZ*moorings).where(ds.maskC,False)

def get_loc_from_mask(ds,mask,
        xdim='XC',ydim='YC',zdim='Z'):
    """given a mask, return the lon,lat,depth coordinates
    where it is active

    Parameters
    ----------
    ds : xarray Dataset
        containing the grid information
    mask : xarray DataArray
        True/False where active/inactive
    xdim, ydim, zdim : str, optional
        denoting grid cell location, must correspond to mask dims
        e.g. XC,YC,Z for tracer, 'C'
             XG,YC,Z for west (e.g. zonal velocity), 'W'
             etc...

    Returns
    -------
    lon, lat, depth : list of floats
        denoting where mask is active (i.e. True)
    """

    def _return_list_from_mask(mask_in,coord_as_grid):
        """helper function to reduce typing"""
        coord_list = coord_as_grid.where(mask_in,drop=True).values.flatten()
        return coord_list[~np.isnan(coord_list)]

    # broadcast to get coordinates as a grid
    (z,y,x) = xr.broadcast(ds[zdim],ds[ydim],ds[xdim])

    lon = _return_list_from_mask(mask,x)
    lat = _return_list_from_mask(mask,y)
    depth = _return_list_from_mask(mask,z)

    return lon, lat, depth
