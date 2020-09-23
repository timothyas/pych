import matplotlib.pyplot as plt
from matplotlib import cm, colors
import cartopy.crs as ccrs
import cartopy.feature as cf

from .utils import get_xy_coords

def plot_xy(xda,
            fig=None, ax=None,
            subplot_kw={
                'projection':
                    ccrs.Orthographic(central_longitude=-45,central_latitude=50)
                    },
            **kwargs):
    """Plot a 2D field in the X-Y plane (i.e. lat/lon)

    Parameters
    ----------
    xda : xarray DataArray
        with the 2D field and underlying x (lon) and y (lat) coordinates
    fig, ax : matplotlib figure and axis objects, optional
        if None then they are created
    subplot_kw : dict, optional
        options passed to matplotlib.pyplot.subplots, e.g. for creating nice
        projections
     
    """

    if fig is None and ax is None:
        fig,ax = plt.subplots(subplot_kw=subplot_kw)
    norm = colors.Normalize(vmin=xda.min().values,vmax=xda.max().values)

    x,y = get_xy_coords(xda)

    for ff in xda.face:
        C=ax.pcolormesh(xda[x][ff],xda[y][ff],
                        xda.sel(face=ff).where(xda.sel(face=ff)!=0.),
                        norm=norm,
                        transform=ccrs.PlateCarree(),**kwargs)

    ax.coastlines()
    ax.add_feature(cf.LAND,facecolor='0.75')
    g1=ax.gridlines(draw_labels=False)
    mylabel = ''
    if 'long_name' in xda.attrs:
        mylabel = mylabel+xda.long_name+' '
    if 'units' in xda.attrs:
        mylabel = mylabel+f'[{xda.units}]'
    plt.colorbar(C,norm=norm,label=mylabel)
    return fig, ax
