import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import cartopy.crs as ccrs
import cartopy.feature as cf
import pyresample as pr


from .utils import get_xy_coords


def plot_xy(
    xda,
    fig=None,
    ax=None,
    subplot_kw={
        "projection": ccrs.Orthographic(central_longitude=-45, central_latitude=50)
    },
    **kwargs,
):
    """Plot a 2D field in the X-Y plane (i.e. lat/lon)

    Parameters
    ----------
    xda : xarray DataArray
        with the 2D field and underlying x (lon) and y (lat) coordinates
    fig, ax : matplotlib figure and axis objects, optional
        if None then they are created
    subplot_kw : dict, optional
        options passed to matplotlib.pyplot.subplots, e.g. for creating nice
        projections

    """

    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
    norm = colors.Normalize(vmin=xda.min().values, vmax=xda.max().values)

    x, y = get_xy_coords(xda)

    for ff in xda.face:
        C = ax.pcolormesh(
            xda[x][ff],
            xda[y][ff],
            xda.sel(face=ff).where(xda.sel(face=ff) != 0.0),
            norm=norm,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )

    ax.coastlines()
    ax.add_feature(cf.LAND, facecolor="0.75")
    g1 = ax.gridlines(draw_labels=False)
    mylabel = ""
    if "long_name" in xda.attrs:
        mylabel = mylabel + xda.long_name + " "
    if "units" in xda.attrs:
        mylabel = mylabel + f"[{xda.units}]"
    plt.colorbar(C, norm=norm, label=mylabel)
    return fig, ax


class aste_map:
    # Note that most of this was copied and pasted from the xgcm documentation
    # Because I couldn't write anything fancier

    def __init__(self, ds, dx=0.25, dy=0.25):
        # Extract LLC 2D coordinates
        lons_1d = ds.XC.values.ravel()
        lats_1d = ds.YC.values.ravel()

        # Define original grid
        self.orig_grid = pr.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)

        # Longitudes latitudes to which we will we interpolate
        lon_tmp = np.arange(-180, 180, dx) + dx / 2
        lat_tmp = np.arange(-35, 90, dy) + dy / 2

        # Define the lat lon points of the two parts.
        self.new_grid_lon, self.new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)
        self.new_grid = pr.geometry.GridDefinition(
            lons=self.new_grid_lon, lats=self.new_grid_lat
        )

    def __call__(
        self,
        da,
        ax=None,
        projection=ccrs.Orthographic,
        lon_0=-45,
        lat_0=50,
        figsize=(6, 6),
        show_cbar=True,
        cbar_label=None,
        **plt_kwargs,
    ):

        tiledim = "tile" if "face" not in da.dims else "face"
        assert set(da.dims) == set(
            [tiledim, "j", "i"]
        ), f"da must have dimensions [{tiledim}, 'j', 'i']"

        if ax is None:
            proj = projection(central_longitude=lon_0, central_latitude=lat_0)
            _, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})

        field = self.regrid(da)

        vmax = np.nanmax(field)
        vmin = np.nanmin(field)
        if vmax * vmin < 0:
            vmax = np.nanmax([np.abs(vmax), np.abs(vmin)])
            vmin = -vmax
        vmax = plt_kwargs.pop("vmax", vmax)
        vmin = plt_kwargs.pop("vmin", vmin)

        # Handle colorbar and NaN color
        cmap = "RdBu_r" if vmax * vmin < 0 else "viridis"
        cmap = plt_kwargs.pop("cmap", cmap)
        if isinstance(cmap,str):
            cmap = plt.cm.get_cmap(cmap)

        x, y = self.new_grid_lon, self.new_grid_lat

        # Find index where data is splitted for mapping
        split_lon_idx = round(
            x.shape[1] / (360 / (lon_0 if lon_0 > 0 else lon_0 + 360))
        )

        # Plot each separately
        p = ax.pcolormesh(
            x[:, :split_lon_idx],
            y[:, :split_lon_idx],
            field[:, :split_lon_idx],
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            zorder=1,
            **plt_kwargs,
        )
        p = ax.pcolormesh(
            x[:, split_lon_idx:],
            y[:, split_lon_idx:],
            field[:, split_lon_idx:],
            vmax=vmax,
            vmin=vmin,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            zorder=2,
            **plt_kwargs,
        )

        # Add land and coastlines
        ax.add_feature(cf.LAND.with_scale("50m"), facecolor='0.75',zorder=3)
        ax.add_feature(cf.COASTLINE.with_scale("50m"), zorder=3)

        # Add gridlines
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True if projection == ccrs.Mercator() else False,
            linewidth=2,
            color="gray",
            alpha=0.5,
            linestyle="-",
        )

        # Add label from attributes
        if cbar_label is None:
            cbar_label = ""
            if "long_name" in da.attrs:
                cbar_label = cbar_label+da.long_name+" "
            if "units" in da.attrs:
                cbar_label = cbar_label+f"[{da.units}]"

        # Colorbar...
        if show_cbar:
            cb = plt.colorbar(
                p,
                ax=ax,
                shrink=0.8,
                orientation="horizontal",
                pad=0.05,
                label=cbar_label
            )
            cb.ax.tick_params()


        return ax

    def regrid(self, xda):
        """regrid xda based on llcmap grid"""
        return pr.kd_tree.resample_nearest(
            self.orig_grid,
            xda.values,
            self.new_grid,
            radius_of_influence=100000,
            fill_value=None,
        )
