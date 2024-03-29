"""
Some PIG specific plotting routines
"""

from copy import copy
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxesSubplot

from .stereoplot import StereoPlot
from .matern import calc_variance
from .utils import convert_units
from ..calc import calc_baro_stf

def fix_obcsw_plot(ax,
                   xlim=[-75.2,None],
                   ylim=[-1150,0],
                   background='gray'):
    ax.set(xlabel='',ylabel='',title='',
           xlim=xlim,ylim=ylim)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dY = ylim[1]-ylim[0]
    dX = xlim[1]-xlim[0]
    ax.grid(False)
    ax.add_patch(plt.Rectangle((xlim[0],ylim[0]),dX,dY,
                               facecolor=background,
                               zorder=0))

    nrows = ax.get_gridspec().nrows
    if ax.get_subplotspec().colspan[0]==0:
        ax.set_ylabel('Height\n(m)',labelpad=30,rotation=0)
    if ax.get_subplotspec().rowspan[0]==nrows-1:
        ax.set_xlabel('Latitude')
    return ax

def plot_meltrate(ds,sp=None,ax=None,
                  cmap='inferno',
                  add_text=True,
                  vmin=0, vmax=16, dv=4,
                  plot_type='contourf',
                  units='Mt/yr',
                  cbar_kwargs={},
                  textcolor=None,
                  texttitle=None,
                  **kwargs):
    """make nice meltrate plot with desired units

    Parameters
    ----------
    ds : xarray.Dataset
        with SHIfwFlx in it
    sp : pych.pigmachine.StereoPlot, optional
        defining the plot, if None, default is created
    ax : matplotlib.axes, optional
        particular axis to pass to sp, optional
    cmap : str or matplotlib.colormap, optional
        the colormap
    add_text : bool, optional
        with mean, max, and total meltrate
    vmin, vmax, dv : float, optional
        defines the colorbar min, max, and increment
    units : str, optional
        desired unit to convert to, see pych.pigmachine.utils.convert_units
    plot_type : str, optional
        default to contourf for now, other options is pcolormesh
    cbar_kwargs : dict, optional
        see StereoPlot.plot
    textcolor : str, optional
        textcolor if add_text is True
    kwargs
        additional arguments sent to xarray's plotting wrapper via stereo_plot

    Returns
    -------
    fig, ax : if cbar_kwargs is not None
    fig, ax, mappable : if cbar_kwargs is None

    """

    # get the units
    fld = 'SHIfwFlx'
    if 'units' not in ds[fld].attrs:
        raise TypeError("No units in ds.{fld} ... this is causing issues")
        warnings.warn(f'No units in DataArray ds.{fld}, assuming all good...')
    else:
        convertto = 'Mt/m^2/yr' if units=='Mt/yr' else units
        meltrate = -convert_units(ds[fld],units_out=convertto)
        meltrate = meltrate*ds['rA'] if units=='Mt/yr' else meltrate

        # always get Mt/yr for total
        meltrateMT = -ds['rA']*convert_units(ds[fld],units_out='Mt/m^2/yr')

    if ax is None and sp is None:
        sp = StereoPlot()

    # set inputs for StereoPlot object
    if cbar_kwargs == {}:
        cbar_kwargs={'label':f'Meltrate ({units})',
                     'ticks':np.arange(vmin,vmax+1,dv)}
    pkw = {'cmap':cmap, 'vmin':vmin, 'vmax':vmax,
           'cbar_kwargs':cbar_kwargs,
           **kwargs}

    # Make the actual plot
    plot = getattr(meltrate.plot, plot_type)
    if ax is None:
        returns = sp.plot(meltrate, plot=plot, **pkw)
    else:
        returns = sp.plot(meltrate,ax=ax, plot=plot, **pkw)

    fig = returns[0]
    ax = returns[1]
    if len(returns)>2:
        mappable = returns[2]

    # add some nice text
    if add_text:
        tdict = {}
        # omittimg mean for now
        #mean = (meltrate*ds.rA).sum() / ds.rA.sum()
        #tdict['Mean'] = [float(mean),units]
        #tdict['Max'] = [float(meltrate.max().values),units]
        tdict['Total'] = [float(meltrateMT.sum(['XC','YC']).values),'Mt/yr']
        for key, val in tdict.items():
            if val[0]>1000:
                val[1] = val[1].replace('M','G')
                val[0] = val[0]/1000 if val[1][0]=='G' else val[0]

        if textcolor is None:
            textcolor='white' if sp.background=='black' else None

        tY = -74.85 #if texttitle is None else -74.685

        txt='' if texttitle is None else texttitle+'\n'
        for key, val in tdict.items():
            txt+=f'{key}: {val[0]:.2f} {val[1]}\n'
        ax.text(-102.075,tY,txt,
                color=textcolor,
                transform=ccrs.PlateCarree(),
                horizontalalignment='center')

    return returns

def plot_barostf(ds,grid,sp=None,ax=None,
                 Z=None,
                 addBathyContours=True,
                 addStreamlines=True,
                 addQuiver=False,
                 addText=True,
                 vmax=0.2,
                 cmap='cmo.balance',
                 cbar_kwargs={},
                 **kwargs):
    """Plot barotropic streamfuncion

    Parameters
    ----------
    ds : xarray Datset
        with 'UVELMASS' for sreamfunction only, 'VVELMASS' if doing quiver or streamlines
    grid : xgcm.Grid
    sp : pych.pigmachine.StereoPlot, optional
    ax : matplotlib.axis, optional
    Z : float, array_like, or xarray.DataArray, optional
        specifying depth levels to grab
    addBathyContours : bool, optional
        if True, add gray bathymetry contours in the background
    addStreamlines, addQuiver : bool, optional
        Show the flow as streamlines or "quiver".
        Both cannot be True, but both can be False.
    addText : bool, optional
        Show the maximum and minimum streamfunction text
    vmax : float, optional
        maximum amplitude for colormap
    cmap : str or matplotlib colormap object, optional
        colormap for the streamfunction

    Returns
    -------
    fig,ax : matplotlib.figure / axis
    """

    if addStreamlines and addQuiver:
        assert (not (addStreamlines and addQuiver)), 'cannot do both quiver and streamlines'


    # do some calc's
    xds = ds if Z is None else ds.sel(Z=Z)
    barostf=calc_baro_stf(xds,grid)
    maskG = xds.maskC if Z is None else xds.maskC.sel(Z=Z)

    # If selecting Z, then mask out Z levels with "all"
    # otherwise, want full streamfunction "anywhere" in Z
    maskG=maskG.any('Z').values if Z is None else maskG.all('Z').values
    barostf=barostf.where(maskG)

    # setup args
    label = 'Barotropic Streamfunction (Sv)'
    if Z is not None:
        dup = int(np.abs(xds.Z.max().values))
        dlo = int(np.abs(xds.Z.min().values))
        label += f'\n Depth: {dup}-{dlo}m'

    if cbar_kwargs == {}:
        cbar_kwargs={'extend':'max','label':label,
                     'ticks':1e-1*np.arange(-2,3,1)}

    # plot
    sp = StereoPlot() if sp is None else sp
    if ax is None:
        results = sp.plot(barostf,vmin=-vmax,vmax=vmax,cmap=cmap,
                         cbar_kwargs=cbar_kwargs, **kwargs)
    else:
        results = sp.plot(barostf,ax=ax,vmin=-vmax,vmax=vmax,cmap=cmap,
                         cbar_kwargs=cbar_kwargs, **kwargs)

    if cbar_kwargs is not None:
        fig,ax = results
    else:
        fig,ax,mappable = results


    if addBathyContours:
        ds.Depth.plot.contour(cmap='gray_r',levels=15,alpha=.3,ax=ax,
                              transform=ccrs.PlateCarree())
    if addStreamlines:
        streamplot(xds,grid,ax=ax,maskW=xds.maskW,maskS=xds.maskS,
                   scaleByKE=4,ke_threshold=.025,density=2,color='k');
    if addQuiver:
        quiver(xds,grid,ax=ax,skip=4,scale=2,width=.004,alpha=.8,ke_threshold=0.05);
    ax.set_title('')

    if addText:
        ax.text(-100,-74.91,
                f'Max: {float(barostf.max()):.2f} Sv\nMin: {float(barostf.min()):.2f} Sv',
                fontsize=16,color='white',transform=ccrs.PlateCarree());

    if cbar_kwargs is not None:
        return fig,ax
    else:
        return fig,ax,mappable

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

def quiver(ds,grid,ax=None,maskW=None,maskS=None,skip=1, ke_threshold=.1, **kwargs):
    """
    Make a quiver plot from velocities in a dataset

    Parameters
    ----------
    ds : xarray Dataset
        with UVELMASS, VVELMASS fields
    grid : xgcm Grid object
        to interp with
    ax : matplotlib axis object, optional
        defining the axis to plot on
    maskW, maskS : xarray DataArrays
        selecting the field to be plotted
    skip : int, optional
        how many velocity points to skip by when plotting the arrows
    ke_threshold : float, optional
        fractional of max(np.sqrt(u**2 + v**2)) to plot, removes tiny dots where velocity
        is near zero
    **kwargs : dict, optional
        passed to matplotlib.pyplot.quiver, some common ones are:

        scale : int, optional
            to scale the arrows by
        width : float, optional
            width of the arrowhead
        alpha : float, optional
            opacity of the arrows
    """
    if ax is None:
        _,ax = plt.subplots()
    sl = slice(None,None,skip)
    x = ds.XC[sl]
    y = ds.YC[sl]

    u = ds.UVELMASS if maskW is None else ds.UVELMASS.where(maskW)
    v = ds.VVELMASS if maskS is None else ds.VVELMASS.where(maskS)

    u,v = grid.interp_2d_vector({'X':u.mean('Z'),'Y':v.mean('Z')},boundary='fill').values()
    u,v = [ff.where(ds['maskC'].any('Z'))[sl,sl] for ff in [u,v]]

    # hide the little dots where velocity ~0
    ke = np.sqrt(u**2+v**2)
    u = u.where(ke>ke_threshold*ke.max(),np.nan)
    v = v.where(ke>ke_threshold*ke.max(),np.nan)

    # quiver wants numpy arrays
    x,y,u,v = [ff.values for ff in [x,y,u,v]]
    if isinstance(ax,GeoAxesSubplot):
        ax.quiver(x,y,u,v,pivot='tip',transform=ccrs.PlateCarree(),**kwargs)
    else:
        ax.quiver(x,y,u,v,pivot='tip',**kwargs)
    return ax

def streamplot(ds,grid,ax=None,maskW=None,maskS=None, ke_threshold=.1,
               scaleByKE=0,**kwargs):
    """
    Make a quiver plot from velocities in a dataset

    Parameters
    ----------
    ds : xarray Dataset
        with UVELMASS, VVELMASS fields
    grid : xgcm Grid object
        to interp with
    ax : matplotlib axis object, optional
        defining the axis to plot on
    maskW, maskS : xarray DataArrays
        selecting the field to be plotted
    skip : int, optional
        how many velocity points to skip by when plotting the arrows
    ke_threshold : float, optional
        fractional of max(np.sqrt(u**2 + v**2)) to plot, removes tiny dots where velocity
        is near zero
    scaleByKE : float, optional
        if >0, then scale linewidth by this * KE/max(KE)
    **kwargs : dict, optional
        passed to matplotlib.pyplot.quiver, some common ones are:

        density : float, optional
            density of the contours
        linewidth : float, optional
            width of the streamlines, if array same size as u/v, scales with values
        alpha : float, optional
            opacity of the arrows
    """
    if ax is None:
        _,ax = plt.subplots()
    x = ds.XC
    y = ds.YC.interp(YC=np.linspace(ds.YC.min(),ds.YC.max(),len(ds.YC)))

    u = ds.UVELMASS if maskW is None else ds.UVELMASS.where(maskW)
    v = ds.VVELMASS if maskS is None else ds.VVELMASS.where(maskS)

    u,v = grid.interp_2d_vector({'X':u.mean('Z'),'Y':v.mean('Z')},boundary='fill').values()
    u,v = [ff.where(ds['maskC'].any('Z')) for ff in [u,v]]

    # hide the little dots where velocity ~0
    ke = np.sqrt(u**2+v**2)
    u = u.where(ke>ke_threshold*ke.max(),np.nan)
    v = v.where(ke>ke_threshold*ke.max(),np.nan)

    # quiver wants numpy arrays
    if scaleByKE is not None:
        kwargs = {} if kwargs is None else kwargs
        #if kwargs is not None:
        #    #assert 'linewidth' not in kwargs.keys(), \
        #    #        'do not pass linewidth if scaling by KE'
        #    pass
        #else:
        #    kwargs={}

        if isinstance(scaleByKE,(list,tuple)):
            key = scaleByKE[0]
            ratio_factor = scaleByKE[1]
        else:
            key = 'linewidth'
            ratio_factor = scaleByKE

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scaling = xr.where(ke>0,ratio_factor*ke/ke.max(),0.).values

        kwargs[key] = scaling

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x,y,u,v,ke = [ff.values for ff in [x,y,u,v,ke]]
    if isinstance(ax,GeoAxesSubplot):
        ax.streamplot(x,y,u,v,transform=ccrs.PlateCarree(),**kwargs)
    else:
        ax.streamplot(x,y,u,v,**kwargs)
    return ax
