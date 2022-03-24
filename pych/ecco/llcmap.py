"""
Map class for creating lon/lat projections of LLC grid
This was originally copied form the xgcm documentation:

    https://xgcm.readthedocs.io/en/latest/example_eccov4.html#A-Pretty-Map

Because wow, it's well written
"""
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import pyresample as pr

class atlantic_map:

    # Note that most of this was copied and pasted from the xgcm documentation

    # Because I couldn't write anything fancier

    def __init__(self, ds, dx=0.25, dy=0.25):
        # Extract LLC 2D coordinates
        lons_1d = ds.XC.values.ravel()
        lats_1d = ds.YC.values.ravel()

        # Define original grid
        self.orig_grid = pr.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)

        # Longitudes latitudes to which we will we interpolate
        lon_tmp = np.arange(-180, 180, dx) + dx/2
        lat_tmp = np.arange(-90, 90, dy) + dy/2

        # Define the lat lon points of the two parts.
        self.new_grid_lon, self.new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)
        self.new_grid  = pr.geometry.GridDefinition(lons=self.new_grid_lon,
                                                    lats=self.new_grid_lat)

    def __call__(self, da, ax=None, projection=ccrs.Robinson(), lon_0=-60,
                 lon_bds=[-100,40],
                 lat_bds=[-65,65],
                 show_cbar=True,
                 cbar_label='',
                 **plt_kwargs):

        tiledim = 'tile' if 'face' not in da.dims else 'face'
        assert set(da.dims) == set([tiledim, 'j', 'i']), f"da must have dimensions [{tiledim}, 'j', 'i']"

        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection':projection})


        field = self.regrid(da)

        vmax = plt_kwargs.pop('vmax', np.nanmax(field))
        vmin = plt_kwargs.pop('vmin', np.nanmin(field))
        if vmax*vmin < 0:
            vmax = np.nanmax(np.abs([vmax,vmin]))
            vmin = -vmax

        # Handle colorbar and NaN color
        default_cmap = 'RdBu_r' if vmax*vmin < 0 else 'viridis'
        cmap = plt_kwargs.pop('cmap', default_cmap)
        if type(cmap)==str:
            cmap = copy(plt.cm.get_cmap(cmap))
        cmap.set_bad(color='gray',alpha=.6)

        if lon_bds is not None and lat_bds is not None:
            ax.set_extent([lon_bds[0],lon_bds[1],lat_bds[0],lat_bds[1]])
        x,y = self.new_grid_lon, self.new_grid_lat

        # Find index where data is splitted for mapping
        split_lon_idx = round(x.shape[1]/(360/(lon_0 if lon_0>0 else lon_0+360)))


        # Plot each separately
        p = ax.pcolormesh(x[:,:split_lon_idx], y[:,:split_lon_idx], field[:,:split_lon_idx],
                           vmax=vmax, vmin=vmin, cmap=cmap,
                           transform=ccrs.PlateCarree(), zorder=1, **plt_kwargs)
        p = ax.pcolormesh(x[:,split_lon_idx:], y[:,split_lon_idx:], field[:,split_lon_idx:],
                           vmax=vmax, vmin=vmin, cmap=cmap,
                           transform=ccrs.PlateCarree(), zorder=2, **plt_kwargs)

        # Add land and coastlines
        ax.add_feature(cf.LAND.with_scale('50m'), zorder=3, color=[.8, .8, .8])
        ax.add_feature(cf.COASTLINE.with_scale('50m'), zorder=3)

        # Add gridlines
        ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels= True if projection==ccrs.Mercator() else False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='-')

        # Colorbar...
        if show_cbar:
            cb=plt.colorbar(p,ax=ax,shrink=.8,label=cbar_label,
                            orientation='horizontal',pad=0.05)
            cb.ax.tick_params()

        return ax

    def regrid(self,xda):
        """regrid xda based on llcmap grid"""
        return pr.kd_tree.resample_nearest(self.orig_grid, xda.values,
                                           self.new_grid,
                                           radius_of_influence=100000,
                                           fill_value=None)

