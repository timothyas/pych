"""
Collection of quick and simple plotting functions

  horizontal_map - driver to make two nice horizontal maps next to each other
  depth_slice - same, but for contour plot of depth vs some coordinate
  _nice_plot - underlying script for a single nice figure
"""

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import xarray as xr
from warnings import warn
try:
    import ecco_v4_py as ecco
except ImportError:
    print('You need to reorganize pych at some point')

from matplotlib.ticker import MultipleLocator

from .utils import get_cmap_rgb

def plot_logbin(xda,x=None, y=None,
                nbins=3,bin_edges=None,
                ax=None,
                cmap='RdBu_r',
                cbar_label=None,
                **kwargs):
    """Make a plot, binning field by log10 of values

    Parameters
    ----------
    xda : xarray.DataArray
        field to be plotted, must be 2D
    x, y : array_like, optional
        x and y coordinates for the plot
    nbins : int, optional
        number of colored bin (centers) positive and negative values
        i.e. we get 2*nbins+1, bins. one is neutral (middle)
    bin_edges : array-like, optional
        exclusive with nbins, specify bin edges (positive only)
    ax : matplotlib.axes, optional
        to make plot at
    cmap : str, optional
        specifies colormap
    cbar_label : str, optional
        label for colorbar, default grabs units from DataArray
    kwargs
        passed to matpotlib.pyplot.contourf

    Returns
    -------
    ax : matplotlib.axes
        if one is not provided
    """
    return_ax = False
    if ax is None:
        _,ax = plt.subplots()
        return_ax=True

    if nbins is not None and bin_edges is not None:
        raise TypeError('one or the other')

    log = np.log10(np.abs(xda))
    log = log.where((~np.isnan(log)) & (~np.isinf(log)),0.)
    if nbins is not None:
        _,bin_edges = np.histogram(log,bins=nbins)
    else:
        nbins = len(bin_edges)-1
    logbins=np.round(bin_edges)

    # determine if colorbar will be extended
    maxExtend = (xda>10**logbins[-1]).any().values
    minExtend = (xda<-10**logbins[-1]).any().values
    extend='neither'
    if minExtend and maxExtend:
        extend='both'
    elif maxExtend:
        extend='max'
    elif minExtend:
        extend='min'

    # determine number of colors, adding one for each extension
    # and always one extra, the middle color bin
    ncolors=2*nbins+1
    ncolors = ncolors+1 if maxExtend else ncolors
    ncolors = ncolors+1 if minExtend else ncolors

    # if only one end is extended,
    # chop off the extreme value from the other end to fit
    # in the middle (neutral) colorbin
    if extend in ['min' ,'max']:
        cmap = get_cmap_rgb(cmap,ncolors+1)
        bot =  np.arange(1,nbins+1) if extend=='max' else np.arange(0,nbins+1)
        top = np.arange(ncolors-nbins,ncolors) if extend=='min' else np.arange(ncolors-nbins,ncolors+1)
        index = list(bot)+[nbins+1]+list(top)
        cmap = cmap[index,:]
    else:
        cmap=get_cmap_rgb(cmap,ncolors)

    # levels and plot
    levels=10**logbins
    levels = np.concatenate([-levels[::-1],levels],axis=0)
    if x is None or y is None:
        im=ax.contourf(xda,levels=levels,colors=cmap,extend=extend,**kwargs)
    else:
        im=ax.contourf(x,y,xda,levels=levels,colors=cmap,extend=extend,**kwargs)

    # label dem ticks
    if cbar_label==None and 'units' in xda.attrs:
            cbar_label=f'[{xda.attrs["units"]}]'
    p=plt.colorbar(im,ax=ax,label=cbar_label)
    ticklabels = [f'-10^{b:.0f}' for b in logbins[::-1]]
    ticklabels += [f'10^{b:.0f}' for b in logbins]
    p.set_ticklabels(ticklabels)

    if return_ax:
        return ax
    else:
        return im

def nice_inward_ticks(ax,
                      xminor_skip=None,yminor_skip=None):
    """Make nice inward pointing ticks

    Parameters
    ----------
    ax : matplotlib axis object

    xminor_skip, yminor_skip : int, optional
        interval of "minor" ticks, if None, then no minor ticks
    """

    ax.tick_params(direction='in',which='major',length=8,
                   top=True,right=True,pad=6)

    if xminor_skip is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(xminor_skip))
    if yminor_skip is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(yminor_skip))

    if xminor_skip is not None or yminor_skip is not None:
        top = xminor_skip is not None
        right = yminor_skip is not None
        ax.tick_params(direction='in',which='minor',length=5,
                       top=top,right=right,pad=6)

def fill_between_std(x,ymean,ystd,
        ax=None,fill_alpha=0.4,**kwargs):
    """A simple version of fill between to reduce typing"""

    fill_kwargs = copy(kwargs)
    if 'alpha' in kwargs:
        warn(f'Resetting fill_alpha with provided alpha={kwargs["alpha"]}')
        fill_kwargs['alpha'] = kwargs['alpha']
    else:
        fill_kwargs['alpha'] = fill_alpha

    ax.plot(x,ymean,**kwargs) if ax is not None else plt.plot(x,ymean,**kwargs)
    ax.fill_between(x,ymean-ystd,ymean+ystd,**fill_kwargs) if ax is not None else \
         plt.fill_between(x,ymean-ystd,ymean+ystd,**fill_kwargs)

def plot_section(fld, left, right,
                 datasets, grids,
                 labels=None,
                 collapse_dim='x',
                 plot_diff0=False,
                 plot_sections_at_bottom=False,
                 single_plot=False,
                 nrows=None,
                 ncols=5,
                 fig=None,
                 xr_kwargs={}):
    """Plot a field in each dataset provided along a section in the domain

    Parameters
    ----------
    fld : str
        string denoting field to grab in each dataset
    left, right : pair of floats
        denoting in longitude/latitude the coordinates of the left and rightmost
        points to get a section of
    datasets : list of xarray Datasets
        containing all the data
    grids : list of or a single xgcm Grid object(s)
        this allows one to get a section of the data
        use a single grid if all datasets have same grid information
    labels : list of strings, optional
        corresponding to the different datasets to label in figure
    collapse_dim : str, optional
        dimension along which to collapse
    plot_diff0 : bool, optional
        plot difference between first dataset and all others
    plot_sections_at_bottom : bool, optional
        if True, add a row at the bottom showing the section line
        for each field
    single_plot : bool, optional
        if True, plot all fields on one plot, better be 1D
    ncols : int, optional
        changes the relative width of the quantity being plotted
        and the rightmost plot showing the section
    fig : matplotlib figure object, optional
        for a different figure size
    xr_kwargs : dict, optional
        arguments to pass to xarray's plotting wrapper

    Returns
    -------
    fig : matplotlib figure object
    axs : matplotlib axis object(s)
    """

    # setup the plot

    if not single_plot:
        nrows = len(datasets) if not plot_sections_at_bottom else len(datasets)+1
    else:
        nrows = 1 if not plot_sections_at_bottom else 2

    ncols = ncols if not plot_sections_at_bottom else len(datasets)
    fig = plt.figure(figsize=(18,6*nrows)) if fig is None else fig
    axs = []
    gs = fig.add_gridspec(nrows,ncols)

    # handle list or single
    datasets = [datasets] if not isinstance(datasets,list) else datasets
    grids = [grids] if not isinstance(grids,list) else grids
    labels = [labels] if not isinstance(labels,list) else labels

    # assumption: same grid for all datasets if length 1
    if len(grids)==1:
        grids = grids*nrows

    if len(labels)==1:
        labels = labels*nrows

    # set colormap for depth plot with section
    cmap_deep = copy(plt.get_cmap('cmo.deep'))
    cmap_deep.set_bad('gray')

    if single_plot:
        ax = fig.add_subplot(gs[0,:-1]) if not plot_sections_at_bottom else \
            fig.add_subplot(gs[0,:])
        axs.append(ax)
    for i,(ds,g,lbl) in enumerate(zip(datasets,grids,labels)):

        # what to plot
        plotme = ds[fld] - datasets[0][fld] if plot_diff0 and i!=0 else ds[fld]

        # get the section as a mask
        m={}
        m['C'],m['W'],m['S'] = ecco.get_section_line_masks(left,right,ds,g)

        # get coordinates for field
        x,y,mask,sec_mask = _get_coords_and_mask(ds[fld].coords,m)

        # replace collapse dim with actual name
        rm_dim = x if collapse_dim == 'x' else y

        # get mask and field
        mask = mask.where(sec_mask,drop=True).mean(rm_dim)
        plotme = plotme.where(sec_mask,drop=True).mean(rm_dim).where(mask)

        # Plot the field
        if len(plotme.dims)>1:
            if single_plot:
                raise TypeError('Can''t put multiple fields on single plot')
            ax = fig.add_subplot(gs[i,:-1]) if not plot_sections_at_bottom else \
                fig.add_subplot(gs[i,:])
            axs.append(ax)

            plotme.plot.contourf(y='Z',ax=ax,**xr_kwargs)
        else:
            if not single_plot:
                ax = fig.add_subplot(gs[i,:-1]) if not plot_sections_at_bottom else \
                    fig.add_subplot(gs[i,:])
                axs.append(ax)
            plot_dim = x if rm_dim==y else y
            plotme.plot.line(x=plot_dim,ax=ax,label=lbl,**xr_kwargs)
            ax.grid()

        if lbl is not None:
            if not single_plot:
                if plot_diff0 and i!=0:
                    ax.set_title(f'{fld}({lbl}) - {fld}({labels[0]})')
                else:
                    ax.set_title(f'{fld}({lbl})')
            else:
                ax.legend()

        # Plot the section
        axb = fig.add_subplot(gs[i,-1]) if not plot_sections_at_bottom else \
            fig.add_subplot(gs[-1,i])

        datasets[i].Depth.where(datasets[i].maskC.any('Z')).plot(
                ax=axb,cmap=cmap_deep,add_colorbar=False)
        m['C'].cumsum(dim=rm_dim[0]+'C').where(m['C']).plot(ax=axb,cmap='Greys',add_colorbar=False)
        axb.set(title=f'',ylabel='',xlabel='')
        axs.append(axb)

    return fig,axs

def plot_zlev_with_max(xda,use_mask=True,ax=None,xr_kwargs={}):
    """Make a 2D plot at the vertical level where data array
    has it's largest value in amplitude

    Parameters
    ----------
    xda : xarray DataArray
        with the field to be plotted, function of (Z,Y,X)
    use_mask : bool, optional
        mask the field
    ax : matplotlib axis object, optional
        current plotting axis
    xr_kwargs : dict, optional
        additional arguments for xarray plotting method

    Returns
    -------
    z : float
        height of zlevel at maximum
    """
    def _make_float(xarr):
        """useful for putting x,y,z of max val in plot title"""
        if len(xarr)>1:
            warn(f'{xarr.name} has more than one max location, picking first...')
            xarr=xarr[0]
        return float(xarr.values)

    xda_max = np.abs(xda).max()
    x,y,mask = _get_coords_and_mask(xda.coords)

    # get X, Y, Z of max value
    xda_maxloc = xda.where(xda==xda_max,drop=True)
    if len(xda_maxloc)==0:
        xda_maxloc = xda.where(xda==-xda_max,drop=True)
    xsel = _make_float(xda_maxloc[x])
    ysel = _make_float(xda_maxloc[y])
    zsel = _make_float(xda_maxloc['Z'])

    # grab the zlev
    xda = xda.sel(Z=zsel)

    # mask?
    if use_mask:
        xda = xda.where(mask.sel(Z=zsel))

    if ax is not None:
        xda.plot(ax=ax,**xr_kwargs)
        ax.set_title(f'max loc (x,y,z) = ({xsel:.2f},{ysel:.2f},{zsel:.2f})')
    else:
        xda.plot(**xr_kwargs)
        plt.title(f'max loc (x,y,z) = ({xsel:.2f},{ysel:.2f},{zsel:.2f})')

    return zsel

def horizontal_map(x,y,fld1,fld2=None,
        title1=None,title2=None,
        depth=None,log_data=False,
        mask1=None,mask2=None,
        ncolors=None,
        c_lim=None,c_lim1=None,c_lim2=None,
        cmap=None,cmap1=None,cmap2=None):
    """
    Make a figure with plots of fld1 and fld2 over x,y next to e/o

    Parameters
    ----------
        x,y:        Grid information, giving lat/lon coordinates
        fld1/2:     2D field as numpy array or xarray DataArray
                    fld2 optional, otherwise generate single figure

    Optional Parameters
    -------------------
        title1/2:   string for title above figure
        depth:      depth field as an xarray DataArray to be used as
                    plt.contour(depth.XC,depth.YC,depth.Depth)
        log_data:   plot log_10(fld)
        mask1/2:    mask field to with given mask array
        ncolors:    Number of colors for colormap
        c_lim:      two element array with colorbar limits
        c_lim1/2:   different colorbar limits for each plot
                    c_lim is used for both, c_lim1/2 are for left or right plot
        cmap:       string or colormap object
                    default for sequential data is 'YlGnBu_r'
                    default for diverging data is 'BuBG_r'
        cmap1/2:    similar logic for c_lim, c_lim1/2.
                    cmap is global, cmap1/2 are for individual plots
    Returns
    -------
    fig : matplotlib.figure.Figure object

    """

    # Test for c_lim or c_lim1/2
    if c_lim is not None and (c_lim1 is not None or c_lim2 is not None):
        raise ValueError('Can only provide c_lim or c_lim1/2, not all three')
    if cmap is not None and (cmap1 is not None or cmap2 is not None):
        raise ValueError('Can only provide cmap or cmap1/2, not all three')

    if c_lim is not None:
        c_lim1 = c_lim
        c_lim2 = c_lim

    if cmap is not None:
        cmap1 = cmap
        cmap2 = cmap


    fig = plt.figure(figsize=(15,6))


    plt.subplot(1,2,1)
    _single_horizontal_map(x,y,fld1,title1,depth,log_data,mask1,ncolors,c_lim1,cmap1)

    if fld2 is not None:
        plt.subplot(1,2,2)
        _single_horizontal_map(x,y,fld2,title2,depth,log_data,mask2,ncolors,c_lim2,cmap2)

    plt.show()

    return fig

def depth_slice(x,z,fld1,fld2=None,
        title1=None,title2=None,
        depth=None,log_data=False,
        mask1=None,mask2=None,
        ncolors=None,
        c_lim=None,c_lim1=None,c_lim2=None,
        cmap=None,cmap1=None,cmap2=None):
    """
    Make a slice through depth with plots of fld1 and fld2 and depth on y axis next to e/o

    Parameters
    ----------
        x,z:        Grid information, x is some generic coordinate, z is depth
        fld1/2:     2D field as numpy array or xarray DataArray
                    fld2 optional, otherwise generate single figure

    Optional Parameters
    -------------------
        title1/2:   string for title above figure
        depth:      depth field as an xarray DataArray to be used as
                    plt.contour(depth.XC,depth.YC,depth.Depth)
        log_data:   plot log_10(fld)
        mask1/2:    mask field to with given mask array
        ncolors:    Number of colors for colormap
        c_lim:      two element array with colorbar limits
        c_lim1/2:   different colorbar limits for each plot
                    c_lim is used for both, c_lim1/2 are for left or right plot
        cmap:       string or colormap object
                    default for sequential data is 'YlGnBu_r'
                    default for diverging data is 'BuBG_r'
        cmap1/2:    similar logic for c_lim, c_lim1/2.
                    cmap is global, cmap1/2 are for individual plots
    Returns
    -------
    fig : matplotlib.figure.Figure object
    """

    # Test for c_lim or c_lim1/2
    if c_lim is not None and (c_lim1 is not None or c_lim2 is not None):
        raise ValueError('Can only provide c_lim or c_lim1/2, not all three')

    if cmap is not None and (cmap1 is not None or cmap2 is not None):
        raise ValueError('Can only provide cmap or cmap1/2, not all three')

    if c_lim is not None:
        c_lim1 = c_lim
        c_lim2 = c_lim

    if cmap is not None:
        cmap1 = cmap
        cmap2 = cmap

    fig = plt.figure(figsize=(15,6))


    plt.subplot(1,2,1)
    _single_depth_slice(x,z,fld1,title1,depth,log_data,mask1,ncolors,c_lim1,cmap1)

    if fld2 is not None:
        plt.subplot(1,2,2)
        _single_depth_slice(x,z,fld2,title2,depth,log_data,mask2,ncolors,c_lim2,cmap2)

    plt.show()

    return fig


def _single_horizontal_map(x,y,fld,titleStr,depth,log_data,mask,ncolors,c_lim,cmap):
    """
    Non-user facing function to distill horizontal data to numpy array for plotting
    """

    if isinstance(fld, np.ndarray):
        if len(np.shape(fld))==2:
            fld_values = fld
            fld_name = ''
        elif len(np.shape(fld))==3:
            print('Warning: input fld is 3D, taking fld[0,:,:]')
            fld_values = fld[0,:,:]
            fld_name = ''
        else:
            raise TypeError('Input field is >3D and I don''t want to guess the 2 dims to grab')
    else:
        # Assume xarray DataArray
        if 'time' in fld.dims:
            print('Warning: Time dimension present, grabbing first record')
            fld=fld.isel(time=0)

        if 'Z' in fld.dims:
            print('Warning: Z dimension present, grabbing top layer')
            fld=fld.isel(Z=0)

        fld_values = fld.values
        fld_name = fld.name

    # If desired, mask the field
    # Note: do this before getting cbar limits
    if mask is not None:
        if not isinstance(fld,np.ndarray):
            #Assume xarray DataArray
            mask = mask.values

        mask = np.where(mask==0,np.NAN,1)
        fld_values = fld_values * mask

    _nice_plot(x,y,fld_values,titleStr,depth,log_data,mask,ncolors,c_lim,cmap)

def _single_depth_slice(x,z,fld,titleStr,depth,log_data,mask,ncolors,c_lim,cmap):
    """
    Non-user facing function to distill depth slice data to numpy array for plotting
    """

    if isinstance(fld, np.ndarray):
        if len(np.shape(fld))==2:
            fld_values = fld
            fld_name = ''
        elif len(np.shape(fld))==3:
            print('Warning: input fld is 3D, taking fld[0,:,:]')
            fld_values = fld[0,:,:]
            fld_name = ''
        else:
            raise TypeError('Input field is >3D and I don''t want to guess the 2 dims to grab')
    else:
        # Assume xarray DataArray
        if 'time' in fld.dims:
            print('Warning: Time dimension present, grabbing first record')
            fld=fld.isel(time=0)

        # Can't do this for other dimensions because who knows what they will be

        fld_values = fld.values
        fld_name = fld.name

    # If desired, mask the field
    # Note: do this before getting cbar limits
    if mask is not None:
        if not isinstance(fld,np.ndarray):
            #Assume xarray DataArray
            mask = mask.values

        mask = np.where(mask==0,np.NAN,1)
        fld_values = fld_values * mask

    _nice_plot(x,z,fld_values,titleStr,depth,log_data,mask,ncolors,c_lim,cmap)


def _nice_plot(x,y,fld_values,titleStr,depth,log_data,mask,ncolors,c_lim,cmap):
    """
    Generic plotting routine for pcolormesh
    """

    # If desired, take log_10 of data
    if log_data:
        fld_values = np.where(fld_values==0,np.NAN,fld_values)
        fld_values = np.log10(fld_values)


    # Set colorbar limits
    fld_max = np.nanmax(fld_values)
    fld_min = np.nanmin(fld_values)
    if c_lim is not None:
        cmin = c_lim[0]
        cmax = c_lim[1]
    else:
        cmax = fld_max
        cmin = fld_min
        if (cmin*cmax < 0):
            cmax = np.nanmax(np.abs(fld_values))
            cmin=-cmax

    # Determine whether or not to extend colorbar
    if (cmin > fld_min) and (cmax < fld_max):
        extend_cbar = "both"
    elif cmin > fld_min:
        extend_cbar = "min"
    elif cmax < fld_max:
        extend_cbar = "max"
    else:
        extend_cbar = "neither"

    # Set colormap
    if cmap is None:
        if (cmin*cmax < 0):
            cmap=plt.cm.get_cmap(name='BrBG_r',lut=ncolors)
        else:
            cmap = plt.cm.get_cmap(name='YlGnBu_r',lut=ncolors)
    elif type(cmap) is str:
        cmap = plt.cm.get_cmap(name=cmap,lut=ncolors)

    # At last, make the plot
    plt.pcolormesh(x,y,fld_values,
                   vmin=cmin, vmax=cmax,
                   cmap=cmap)
                   #shading='gouraud')
    plt.colorbar(extend=extend_cbar)

    # If depth given, overlay bathymetry contours
    if depth is not None:
        plt.contour(depth.XC,depth.YC,depth.Depth,colors='0.5')

    if titleStr is not None:
        plt.title(titleStr)

def _get_coords_and_mask(coords,xda_dict=None):
    """get coordinate information and mask C/W/S"""

    if set(('XC','YC')).issubset(coords):

        x = 'XC'
        y = 'YC'
        xda = xda_dict['C'] if xda_dict is not None else None
        mask = coords['maskC'] if 'maskC' in coords else None

    elif set(('XG','YC')).issubset(coords):


        x = 'XG'
        y = 'YC'
        xda = xda_dict['W'] if xda_dict is not None else None
        mask = coords['maskW'] if 'maskW' in coords else None

    elif set(('XC','YG')).issubset(coords):


        x = 'XC'
        y = 'YG'
        xda = xda_dict['S'] if xda_dict is not None else None
        mask = coords['maskS'] if 'maskS' in coords else None

    else:

        x = 'XG'
        y = 'YG'
        mask = None
        xda = None
        warn('Unknown mask for field at vorticity points')

    if mask is None:
        warn('No mask in coordinates, returning ones')
        mask = xr.ones_like(coords[y]*coords[x])

    if xda_dict is not None:
        return x, y, mask, xda
    else:
        return x, y, mask
