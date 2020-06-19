"""
Some PIG specific plotting routines
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs


def stereo_plot(xda,xr_cbar=False,
                cbar_kwargs={},background='black',figsize=(12,10),
                **kwargs):
    """Make a South Polar Stereographic projection plot of Pine Island

    Parameters
    ----------
    xda : xarray DataArray
        with the field to be plotted and spatial coordinates lon/lat
    
    """
    x = xda['XC'] if 'XC' in xda.coords else xda['XG']
    lon0 = -float(np.abs(x.mean()))

    fig,ax = plt.subplots(1,1,figsize=figsize,
                          subplot_kw={'projection':
                              ccrs.SouthPolarStereo(central_longitude=lon0)})

    ax.set_extent([-102.75,-99,-75.44,-74.45])
    p = xda.plot(ax=ax,transform=ccrs.PlateCarree(),
                 add_colorbar=xr_cbar,**kwargs)
    # Clean it up
    ax.axis('off')
    ax.set(ylabel='',xlabel='',title='')

    # Gridlines
    color = 'white' if background == 'black' or background == 'gray' else 'gray'
    ax.gridlines(draw_labels=True,color=color,alpha=.3,y_inline=False,
                 xlocs=-99.5-np.arange(4),ylocs=-75.25+.25*np.arange(4))

    # --- Colorbar
    if not xr_cbar:
        cbar_defaults = _get_cbar_defaults('horizontal') \
                if 'orientation' not in cbar_kwargs else \
                _get_cbar_defaults(cbar_kwargs['orientation'])

        for key,val in cbar_defaults.items():
            if key not in cbar_kwargs:
                cbar_kwargs[key]=val

        fig.colorbar(p,ax=ax,**cbar_kwargs)

    return fig,ax

def _get_cbar_defaults(orientation):
    """get some cbar stuff based on 
    orientation = 'horizontal' or 'vertical

    See commented out code for "inside" colorbar which was abandoned...
    """

    cbar_kw = {'orientation':orientation,'shrink':0.6}

    if orientation == 'horizontal':
        cbar_kw['pad'] = 0.06

    # inside
    #cax = inset_axes(ax,height='25%',width='2.5%',loc='upper left')
    #fig.colorbar(p,ax=ax,cax=cax,orientation='vertical',
    #             extend='both',label='Meltrate (m/yr)',ticks=np.arange(0,71,10))
    return cbar_kw
