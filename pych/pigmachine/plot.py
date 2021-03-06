"""
Some PIG specific plotting routines
"""

from copy import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs

from .matern import calc_variance
from .utils import convert_units

def plot_meltrate(ds,cmap='inferno',
                  add_text=True,
                  bad_color='black',
                  vmin=0, vmax=16, dv=4,
                  units='Mt/yr',**kwargs):
    """make nice meltrate plot with desired units

    Parameters
    ----------
    ds : xarray.Dataset
        with SHIfwFlx in it
    cmap : str or matplotlib.colormap, optional
        the colormap
    add_text : bool, optional
        with mean, max, and total meltrate
    vmin, vmax, dv : float, optional
        defines the colorbar min, max, and increment
    units : str, optional
        desired unit to convert to, see pych.pigmachine.utils.convert_units
    kwargs
        additional arguments sent to xarray's plotting wrapper via stereo_plot

    Returns
    -------
    fig, ax : matplotlib.figure / axis

    """

    fld = 'SHIfwFlx'
    if 'units' not in ds[fld].attrs:
        warnings.warn(f'No units in DataArray ds.{fld}, assuming all good...')
    else:
        convertto = 'Mt/m^2/yr' if units=='Mt/yr' else units
        meltrate = -convert_units(ds[fld],units_out=convertto)
        meltrate = meltrate*ds['rA'] if units=='Mt/yr' else meltrate

    cmap = copy(plt.get_cmap(cmap))
    cmap.set_bad(bad_color)

    fig,ax = stereo_plot(meltrate,
                         cbar_kwargs={'label':'Meltrate (Mt/yr)','ticks':np.arange(vmin,vmax+1,dv),'extend':'both'},
                         cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
    if add_text:
        tdict = {}
        tdict['Mean'] = [float(meltrate.mean().values),units]
        tdict['Max'] = [float(meltrate.max().values),units]
        tdict['Total'] = [float(meltrate.sum(['XC','YC']).values),units]
        for key, val in tdict.items():
            if val[0]>1000:
                val[1] = val[1].replace('M','G')
                val[0] = val[0]/1000 if val[1][0]=='G' else val[0]

        color='white' if bad_color=='black' else None
        txt=''
        for key, val in tdict.items():
            txt+=f'{key}: {val[0]:.2f} {val[1]}\n'
        ax.text(-102.7,-74.8,txt,color=color,
                transform=ccrs.PlateCarree())

    return fig,ax

def plot_lcurve_discrep(ds,d3,dim1='sigma',dim2='Nx',dim3='Fxy',
                        fig=None,axs=None,
                        doLabel=True,lbl_skip=2,
                        legend_labels=None,
                        labeltype='dim',
                        **kwargs):
    """plot L-Curve Criterion and "discrepancy principle plots next to each other"""

    if axs is None:
        fig,axs = plt.subplots(1,2,figsize=(18,6))
    if legend_labels is None:
        if dim1=='Nx':
            legend_labels = ['%d'%d1 for d1 in ds[dim1].values]
        else:
            legend_labels = ['%1.1e'%d1 for d1 in ds[dim1].values]

    for llbl,d1 in zip(legend_labels,ds[dim1].values):

        if isinstance(d3,xr.core.dataarray.DataArray):
            raise NotImplementedError('Need to finish this')
        else:
            myd3 = d3
            plotme = ds.sel({dim1:d1,dim3:myd3},method='nearest')
        var = calc_variance(plotme.Nx)
        reg = var / plotme.sigma**2
        misfits = plotme.misfit_norm
        soln_norm = 1/reg * plotme.reg_norm

        # Make labels of either dimension values (sigma, Nx)
        # or as the actual regularization
        if labeltype=='dim':
            if dim2=='Nx':
                lbl_list = [f'{dd}' for dd in plotme[dim2].values]
            else:
                lbl_list = [f'{dd:1.1e}' for dd in plotme[dim2].values]
        else:
            lbl_list = [f'{rr:2.2e}' for rr in list(reg.values)]
        prefix = ds[dim2].label if labeltype=='dim' else r'$\upsilon\beta^2$'
        lbl_list = [prefix+'= '+lbl for lbl in lbl_list]

        # --- L-Curve and Discrepancy
        axs[0].loglog(misfits,soln_norm,marker='o',label=llbl,**kwargs)
        axs[1].loglog(reg,misfits,marker='o',label=llbl,**kwargs)

        # --- labeling
        if doLabel:
            for x_i,y_i,l_i in zip(misfits[::lbl_skip],
                                   soln_norm[::lbl_skip],lbl_list[::lbl_skip]):
                axs[0].text(x_i,y_i,l_i)
            for x_i,y_i,l_i in zip(reg[::lbl_skip],
                                   misfits[::lbl_skip],lbl_list[::lbl_skip]):
                axs[1].text(x_i,y_i,l_i)

    if isinstance(d3,xr.core.dataarray.DataArray):
        [ax.legend(title=ds[dim1].label+f', ({ds[dim3].label})') for ax in axs];
    else:
        [ax.legend(title=ds[dim1].label+f', ({ds[dim3].label}={d3})') for ax in axs];
    axs[0].set(xlabel=ds.misfit_norm.label,ylabel=r'$\dfrac{1}{\upsilon\beta^2}$'+ds.reg_norm.label)
    axs[1].set(xlabel=r'$\upsilon\beta^2$',ylabel=ds.misfit_norm.label);
    [ax.grid(True) for ax in axs];
    return fig,axs

def plot_map_and_misfits(ds_in,Nx,xi,sigma,obs_packer,ctrl_packer,
                         misfit_fld='misfits_normalized',**kwargs):

    if isinstance(sigma,xr.core.dataarray.DataArray):
        sigma = sigma.sel(Nx=Nx,xi=xi)
    ds = ds_in.sel(Nx=Nx,xi=xi,sigma=sigma,method='nearest')

    fig,axs = plt.subplots(1,2,figsize=(18,6))

    # --- MAP Point
    mmap = ctrl_packer.unpack(ds['m_map'],np.NAN)
    mmap.plot(ax=axs[0],**kwargs)

    mylabel =   ds.m_map.label+'\n'+ \
                ds.Nx.label+f' = {int(ds.Nx.values)}\n'+\
                ds.xi.label+f' = {float(ds.xi.values)}\n'+\
                ds.sigma.label+f' = {float(ds.sigma.values):1.1e}'
    axs[0].text(-75.3,-1000,mylabel,horizontalalignment='center')

    # --- Normalized Misfits
    misfits = obs_packer.unpack(ds[misfit_fld],np.NAN)
    misfits.plot(ax=axs[1])
    axs[1].text(-75.05,-1000, ds[misfit_fld].label)

    [ax.set(title='',xlabel='',ylabel='') for ax in axs];

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
