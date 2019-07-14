"""
Module for generic standard analysis plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cart
import xarray as xr
import ecco_v4_py as ecco


def global_and_stereo_map(lat, lon, fld,
                          plot_type='pcolormesh',
                          cmap='YlOrRd',
                          title=None,
                          cmin=None,
                          cmax=None,
                          dpi=100,
                          show_colorbar=True):

    """Generate the Robinson and Arctic/Antarctic plot.

    Parameters
    ----------
    lat         :   xarray.DataArray 

    lon         :   xarray.DataArray

    fld         :   xarray.DataArray

    plot_type   :   string, optional
                    plot type to use, 'pcolormesh', or 'contourf'

    cmap        :   string or colormap object (TBD)

    cmin        :   double, optional
                    minimum value for colorbar

    cmax        :   double, optional
                    maximum value for colorbar

    dpi         :   int, optiopnal
                    plot resolution in dots (pixels) per inch

    title,show_colorbar
        
    figsize?

    Output
    ------

    """

    # to do
    #   -figsize option?
    #   -cmin/cmax defaults handling with plot_proj ... 
    #   -colorbar defaults with diverging/sequential
    #   -number of colors in plot
    #   -suppress dask warnings
    #   -get the subplot size "just right" no matter the figsize
    #   -arrows for when colorbar is exceeded

    # handle colorbar limits
    cmin, cmax, extend_cbar = set_colorbar_limits(fld,cmin,cmax)

    # default figsize which seems to work for a laptop screen
    plt.figure(figsize=(12,6),dpi=dpi)

    # the big top global plot
    fig, ax1, p1, cb1 = ecco.plot_proj_to_latlon_grid(
            lat,lon,fld,
            cmap=cmap,
            plot_type=plot_type,
            subplot_grid=[2,1,1],
            projection_type='robin',
            show_colorbar=False,
            cmin=cmin,
            cmax=cmax,
            user_lon_0=0
    )

    # Arctic: bottom left
    fig, ax2, p2, cb2 = ecco.plot_proj_to_latlon_grid(
            lat,lon,fld,
            cmap=cmap,
            plot_type=plot_type,
            subplot_grid=[2,2,3],
            projection_type='stereo',
            show_colorbar=False,
            cmin=cmin,
            cmax=cmax,
            lat_lim=50,
            user_lon_0=0
    )


    # ACC: bottom right
    fig, ax3, p3, cb3 = ecco.plot_proj_to_latlon_grid(
            lat,lon,fld,
            cmap=cmap,
            plot_type=plot_type,
            subplot_grid=[2,2,4],
            projection_type='stereo',
            show_colorbar=False,
            cmin=cmin,
            cmax=cmax,
            lat_lim=-40,
            user_lon_0=180
    )

    # Set land color to gray
    ax1.add_feature(cart.feature.LAND,facecolor='0.7',zorder=2)
    ax2.add_feature(cart.feature.LAND,facecolor='0.7',zorder=2)
    ax3.add_feature(cart.feature.LAND,facecolor='0.7',zorder=2)

    # Make a single title
    if title is not None:
        fig.suptitle(title,verticalalignment='top',fontsize=24)

    # Make an overyling colorbar
    if show_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.025, 0.8])
        fig.colorbar(p3, cax=cbar_ax, extend=extend_cbar)



    return fig, (ax1,ax2,ax3)

def plot_depth_slice(x, depth, fld, 
                     stretch_depth=-500,
                     plot_type='pcolormesh',
                     cmap='YlOrRd',
                     title=None,
                     cmin=None,
                     cmax=None,
                     dpi=100,
                     show_colorbar=True):
    """2D plot of depth vs some other variable, stretching first 500m of depth.

    Parameters
    ----------
    depth : xarray DataArray or numpy array
        depth variable
    x : xarray DataArray or numpy array
        variable for x-axis. Likely to be time, latitude, or longitude
    fld : xarray DataArray or numpy array
        2D field with depth + 1 dim
    stretch_depth : scalar (int or float), optional
        stretch top depth to this limit
    """

    # Ensure negative values 
    #if (depth>0).any():
    #    depth = -depth

    #if stretch_depth > 0:
    #    stretch_depth = -stretch_depth

    # Handle shape
    if len(x) == fld.shape[0]:
        fld = fld.transpose()

    # handle colorbar limits
    cmin, cmax, extend_cbar = set_colorbar_limits(fld,cmin,cmax)

    # default figsize which seems to work for a laptop screen
    fig = plt.figure(figsize=(12,6),dpi=dpi)

    # Could also use plt.subplots here ...

    # First top 500m
    ax1 = plt.subplot(2,1,1)
    if plot_type == 'pcolormesh':
        p1 = ax1.pcolormesh(x,depth,fld,vmin=cmin,vmax=cmax,cmap=cmap)

    elif plot_type == 'contourf':
        p1 = ax1.contourf(x,depth,fld,vmin=cmin,vmax=cmax,cmap=cmap)

    # Handle y-axis
    plt.ylim([stretch_depth, 0])
    ax1.yaxis.axes.set_yticks(np.arange(stretch_depth,1,100))
    plt.ylabel('Depth [%s]' % depth.attrs['units'])


    # Remove top plot xtick label
    ax1.xaxis.axes.set_xticklabels([])

    # Now the rest ...
    ax2 = plt.subplot(2,1,2)
    if plot_type == 'pcolormesh':
        p2 = ax2.pcolormesh(x,depth,fld,vmin=cmin,vmax=cmax,cmap=cmap)

    elif plot_type == 'contourf':
        p2 = ax2.contourf(x,depth,fld,vmin=cmin,vmax=cmax,cmap=cmap)

    # Handle y-axis
    plt.ylim([depth.min(), stretch_depth])
    yticks = np.flip(np.arange(2*stretch_depth,depth.min(),-1000))
    ax2.yaxis.axes.set_yticks(yticks)
    plt.ylabel('Depth [%s]' % depth.attrs['units'])

    # Reduce space between subplots
    fig.subplots_adjust(hspace=0.05)

    # Make a single title
    if title is not None:
        fig.suptitle(title,verticalalignment='top',fontsize=24)

    # Make an overyling colorbar
    if show_colorbar:
        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.025, 0.8])
        fig.colorbar(p2, cax=cbar_ax, extend=extend_cbar)

    plt.show()

    return fig,ax1,ax2


def set_colorbar_limits(fld,cmin,cmax):
    """If unset, compute colorbar limits based on field max/min values, sequential/divergent
       Determine if colorbar needs to be extended

    Parameters
    ----------
    fld         :   xarray.DataArray
                    2D field to be plotted

    Output
    ------
    cmin        :   double 
                    colorbar min value
    cmax        :   double 
                    colorbar max value
    extend_cbar :   string 
                    flag to colorbar extension

    """

    # handle input
    if (cmin is None) and (cmax is not None):
        raise RuntimeError('Only cmax given, must provide both cmin and cmax')
    elif (cmin is not None) and (cmax is None):
        raise RuntimeError('Only cmin given, must provide both cmin and cmax')
    else:
        # handle colorbar limits accidentally passed as with xarray functions
        if type(cmin) is xr.DataArray:
            cmin = cmin.values()
        elif cmin is not None:
            raise TypeError('Unsure of cmin type: ',type(cmin))
        if type(cmax) is xr.DataArray:
            cmax = cmax.values()
        elif cmax is not None:
            raise TypeError('Unsure of cmax type: ',type(cmax))

    # compute fld limits
    fld_min = fld.min(skipna=True).values
    fld_max = fld.max(skipna=True).values

    # if cmin/cmax not set, compute
    if (cmin is None) and (cmax is None):

        cmin = fld_min
        cmax = fld_max

        # determine if divergent colorbar 
        # Note: Not making divergent colorbar for temperature
        #       in degC because still sequential even though +/-
        if (fld_max*fld_min < 0) and (fld.name is not 'THETA'):
            cmax = np.nanmax(np.abs(fld.values))
            cmin = -cmax

    # determine if colorbar needs to be extended
    if (cmin > fld_min) and (cmax < fld_max):
        extend_cbar = "both"
    elif cmin > fld_min:
        extend_cbar = "min"
    elif cmax < fld_max:
        extend_cbar = "max"
    else:
        extend_cbar = "neither"

    return cmin, cmax, extend_cbar
