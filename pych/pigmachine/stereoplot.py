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
    extent = [-102.75,-99,-75.44,-74.4675]

    gridline_kw = {'draw_labels':True,
                   'alpha':0.2,
                   'y_inline':False,
                   'xlocs':-99.5-np.arange(4),
                   'ylocs':-75.25+0.25*np.arange(4)}

    def __init__(self, nrows=1, ncols=1,
                 figsize=(12,10), background='black',
                 xr_cbar=False,
                 subplot_kw=None,
                 just_init_fig=False,
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

        skw=subplot_kw if subplot_kw is not None else \
            {'projection':ccrs.SouthPolarStereo(central_longitude=self.lon0)}

        if not just_init_fig:
            fig,axs = plt.subplots(nrows=nrows,ncols=ncols,
                                   figsize=figsize,
                                   subplot_kw=skw,
                                   **kwargs)
            if isinstance(axs,list):
                axs = np.ndarray(axs)
            self.axs = axs
        else:
            fig = plt.figure(figsize=figsize)

        if ncols*nrows>1:
            self.isSingle = False

        else:
            self.isSingle = True

        self.fig = fig
        self.nrows=nrows
        self.ncols=ncols
        self.xr_cbar = xr_cbar
        self.background = background
        self.dLon = self.extent[ 1] - self.extent[ 0]
        self.dLat = self.extent[-1] - self.extent[-2]

    def plot(self, xda, ax=None,
             cbar_kwargs={},
             gridline_kwargs={},
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
        gridline_kwargs : dict, optional
            arguments passed to self.add_gridlines, which is used to set
            attributes in the gridline object. E.g. {"right_labels":True}

        Returns
        -------
        fig, ax : if cbar_kwargs is not None
        fig, ax, mappable : if cbar_kwargs is None
        """

        if ax is None and self.isSingle:
            ax = self.axs
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

        # Contourf ignores the "cmap.set_bad()" call b/c it never plots these
        # points at all
        # facecolor also doesn't work with ax.axis('off')
        # removing the gridlines and ticks doesn't work either b/c the facecolor
        # does not get "projected"
        # so, best way to handle this is by adding a patch with the background color
        # this doesn't change the look of the "set_bad" for pcolormesh either, so
        # consistent for all plot types
        ax.add_patch(plt.Rectangle((self.extent[0],self.extent[2]),
                                   self.dLon, self.dLat,
                                   facecolor=self.background,
                                   transform=ccrs.PlateCarree(),
                                   zorder=0))

        self.add_gridlines(ax,**gridline_kwargs)

        if not (cbar_kwargs is None or self.xr_cbar):
            self.add_colorbar(mappable, ax, cbar_kwargs)
            return self.fig, ax
        elif cbar_kwargs is None:
            return self.fig, ax, mappable

    def add_gridlines(self,ax,**kwargs):
        """helper method for StereoPlot.plot
        add gridlines to plot with axis ax

        Parameters
        ----------
        ax : matplotlib.axis
        kwargs : optional
            override gridline_kw_loc, e.g. "right_labels" or "left_labels"
        """

        colspan = list(ax.get_subplotspec().colspan)
        rowspan = list(ax.get_subplotspec().rowspan)
        gridline_kw_loc={}
        gridline_kw_loc['right_labels'] = (not self.xr_cbar) and (colspan[-1]==self.ncols-1)
        gridline_kw_loc['left_labels'] = colspan[0] == 0
        gridline_kw_loc['top_labels'] = rowspan[0] == 0
        gridline_kw_loc['bottom_labels'] = rowspan[-1] == self.nrows-1

        if kwargs is not None:
            for key,val in kwargs.items():
                gridline_kw_loc[key] = val

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
