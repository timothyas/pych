
def get_xy_coords(xda):
    """Return the dimension name for x and y coordinates
        e.g. XC or XG
    
    Parameters
    ----------
    xda : xarray DataArray
        with all grid information

    Returns
    -------
    x,y : str
        with e.g. 'XC' or 'YC'
    """

    x = 'XC' if 'XC' in xda.coords else 'XG'
    y = 'YC' if 'YC' in xda.coords else 'YG'
    return x,y
