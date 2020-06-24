"""
Some PIG specific plotting routines
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs


def stereo_plot(xda,nrows=1, ncols=1, xr_cbar=False,
                cbar_kwargs={},background='black',figsize=(12,10),
                **kwargs):
    """Make a South Polar Stereographic projection plot of Pine Island

    Parameters
    ----------
    xda : xarray DataArray or list of DataArrays for multiple subplots
        with the field to be plotted and spatial coordinates lon/lat
    nrows, ncols : int, optional
        number of rows and columns for subplots, nrows*ncols must equal
        number of DataArrays to be plotted
    xr_cbar : bool, optional
        use the xarray colorbar or use a nicer custom one
    cbar_kwargs : dict, optional
        if custom colorbar, add a label, ticks, extend? ... etc
    background : str, optional
        color for plot background, where 0 or NaN
    figsize : tuple, optional

    kwargs
        get passed to xarray's plot function

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
        from plt.subplots() call
    """

    if isinstance(xda,xr.core.dataarray.DataArray):
        xda = [xda]

    if len(xda) != nrows*ncols:
        raise TypeError('nrows*ncols != number of provided fields')

    x = xda[0]['XC'] if 'XC' in xda[0].coords else xda[0]['XG']
    lon0 = -float(np.abs(x.mean()))

    fig,axs = plt.subplots(nrows,ncols,figsize=figsize,
                          subplot_kw={'projection':
                              ccrs.SouthPolarStereo(central_longitude=lon0)})

    axs = axs if isinstance(axs,np.ndarray) else np.array(axs)

    for i,(fld,ax) in enumerate(zip(xda,axs.flatten())):
        irow = int(i/nrows)
        icol = i%ncols
        gridline_kw={}
        gridline_kw['right_labels'] = (not xr_cbar) and (icol == ncols-1)
        gridline_kw['left_labels'] = icol == 0 if ncols>1 else True
        gridline_kw['top_labels'] = irow == 0 if nrows>1 else True
        gridline_kw['bottom_labels'] = True#irow == nrows-1
        _make_stereo_plot(fig,ax,fld,xr_cbar,cbar_kwargs,background,
                          gridline_kw,**kwargs)

    axs = ax if nrows*ncols==1 else axs

    return fig,axs

def _make_stereo_plot(fig,ax,fld,xr_cbar,cbar_kwargs,
                      background,gridline_kw,**kwargs):

    ax.set_extent([-102.75,-99,-75.44,-74.45])
    p = fld.plot(ax=ax,transform=ccrs.PlateCarree(),
                 add_colorbar=xr_cbar,**kwargs)
    # Clean it up
    ax.axis('off')
    ax.set(ylabel='',xlabel='',title='')

    # Gridlines
    color = 'white' if background == 'black' or background == 'gray' else 'gray'
    gl = ax.gridlines(draw_labels=True,color=color,alpha=.2,y_inline=False,
                 xlocs=-99.5-np.arange(4),ylocs=-75.25+.25*np.arange(4))
    for key,val in gridline_kw.items():
        setattr(gl,key,val)

    # --- Colorbar
    add_cbar = not xr_cbar if 'add_cbar' not in cbar_kwargs else (not xr_cbar) and cbar_kwargs['add_cbar']
    if add_cbar:
        cbar_defaults = _get_cbar_defaults('horizontal') \
                if 'orientation' not in cbar_kwargs else \
                _get_cbar_defaults(cbar_kwargs['orientation'])

        for key,val in cbar_defaults.items():
            if key not in cbar_kwargs:
                cbar_kwargs[key]=val

        fig.colorbar(p,ax=ax,**cbar_kwargs)

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
