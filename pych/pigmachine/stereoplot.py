"""Class to improve the stereo plot function"""

from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class StereoPlot():
    """Class for convenient South Polar Stereographic PIG plots

    Examples:

        1. Make a single plot

            sp = StereoPlot()
            sp.plot(ds['Depth'])

        2. Make a grid of plots

            fldlist = [ ... list of xarray DataArrays ... ]
            sp.StereoPlot(nrows=2,ncols=2)
            for fld,ax in zip(fldlist,sp.axs):
                sp.plot(fld,ax=ax)
    """

    lon0 = -100.875
    extent = [-102.75,-99,-75.44,-74.45]

    gridline_kw = {'draw_labels':True,
                   'alpha':0.2,
                   'y_inline':False,
                   'xlocs':-99.5-np.arange(4),
                   'ylocs':-75.25+0.25*np.arange(4)}

    def __init__(self, nrows=1, ncols=1,
                 figsize=(12,10), background='black',
                 xr_cbar=False,
                 **kwargs):
        """Create StereoPlot object

        Parameters
        ----------
        nrows, ncols : int, optional
            passed to matplotlib.pyplot.subplots
        figsize : tuple, optional
            passed to matplotlib.pyplot.subplots
        background : str or matplotlib colormap object
            sets the "bad_color" or background color
        xr_cbar : bool, optional
            use xarray's default colorbar, or not
        kwargs
            passed to matplotlib.pyplot.subplots
        """


        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,
                               figsize=figsize,
                               subplot_kw={'projection':
                                           ccrs.SouthPolarStereo(central_longitude=self.lon0)},
                               **kwargs)

        if ncols*nrows>1:
            self.isSingle = False
            axs = axs if isinstance(axs, np.ndarray) else np.array(axs)
        else:
            self.isSingle = True

        self.fig = fig
        self.axs = axs
        self.nrows=nrows
        self.ncols=ncols
        self.xr_cbar = xr_cbar
        self.background = background

    def plot(self, xda, ax=None,
             cbar_kwargs={},
             plot=None,
             **kwargs):
        """Main StereoPlot routine, plot a single 2D xarray DataArray
        on the South Polar Stereographic projection

        Parameters
        ----------
        xda : xarray.DataArray
            the field to plot
        ax : matplotlib axis, optional
            if not passed, use object's axs attribute, which only works if
            the object is a single plot, not multiple subplots
        cbar_kwargs : dict, optional
            options for the colorbar, e.g. "label" or "orientation"
            pass as None to turn off colorbar
            the only default: "orientation":"horizontal"

        Returns
        -------
        ax : matplotlib.axis
        """

        ax = ax if ax is not None else self.axs
        ax.set_extent(self.extent)

        # take care of colormap if in kwargs
        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            cmap = plt.get_cmap(kwargs['cmap']) if isinstance(cmap,str) else cmap
            cmap = copy(cmap)
            cmap.set_bad(self.background)
            kwargs['cmap'] = cmap

        if self.xr_cbar:
            kwargs['cbar_kwargs'] = cbar_kwargs

        if plot is None:
            mappable = xda.plot(ax=ax,
                                add_colorbar=self.xr_cbar,
                                transform=ccrs.PlateCarree(),
                                **kwargs)
        else:
            if 'xarray' in str(plot):
                mappable = plot(ax=ax,transform=ccrs.PlateCarree(),
                                add_colorbar=self.xr_cbar,
                                **kwargs)
            else:
                plot = getattr(ax,plot)
                mappable = plot(xda,transform=ccrs.PlateCarree(),**kwargs)
        # Clean it up
        ax.axis('off')
        ax.set(ylabel='',xlabel='',title='')

        self.add_gridlines(ax)

        if not (cbar_kwargs is None or self.xr_cbar):
            self.add_colorbar(mappable, ax, cbar_kwargs)

        return self.fig, ax

    def add_gridlines(self,ax):
        """helper method for StereoPlot.plot
        add gridlines to plot with axis ax

        Parameters
        ----------
        ax : matplotlib.axis
        """

        if self.isSingle:
            index=0

        else:

            # make a list, only non-NaN value is flattened index location
            index = [i if self.axs.flatten()[i]==ax else np.nan \
                        for i in range(self.nrows*self.ncols)]
            index = np.nansum(index)

        # Now configure the gridlines, depending on subplot arrangement
        irow = index // self.nrows
        icol = index % self.ncols
        gridline_kw_loc={}
        gridline_kw_loc['right_labels'] = (not self.xr_cbar) and (icol == self.ncols-1)
        gridline_kw_loc['left_labels'] = icol == 0 if self.ncols>1 else True
        gridline_kw_loc['top_labels'] = irow == 0 if self.nrows>1 else True
        gridline_kw_loc['bottom_labels'] = True#irow == self.nrows-1

        # set the color and create
        color = 'white' if self.background == 'black' or self.background == 'gray' else 'gray'
        gl = ax.gridlines(color=color, **self.gridline_kw)

        # for some reason, have to add these after
        for key, val in gridline_kw_loc.items():
            setattr(gl,key,val)

    def add_colorbar(self, p, ax, cbar_kwargs):
        """helper method for StereoPlot.plot
        add a non-xarray colorbar

        Parameters
        ------
        p : colorbar mappable
            e.g. p = plt.colormesh( ... )
        ax : matplotlib.axes
        cbar_kwargs : dict
            if 'orientation' not in dictionary, default is 'horizontal'
            see _get_cbar_defaults for some other defaults
            'label' labels the colorbar...
        """

        if 'orientation' not in cbar_kwargs:
            cbar_kwargs['orientation'] = 'horizontal'

        # get defaults and stick it together
        cbar_defaults = _get_cbar_defaults(cbar_kwargs['orientation'])

        cbkw = deepcopy(cbar_kwargs)

        for key,val in cbar_defaults.items():
            if key not in cbar_kwargs:
                cbkw[key]=val

        # make the colorbar
        self.fig.colorbar(p, ax=ax, **cbkw)


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
