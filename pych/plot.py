"""
Collection of quick and simple plotting functions

  horizontal_map - driver to make two nice horizontal maps next to each other
  depth_slice - same, but for contour plot of depth vs some coordinate
  _nice_plot - underlying script for a single nice figure
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


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
