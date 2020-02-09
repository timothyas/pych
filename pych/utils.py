"""
A collection of miscellanious utility functions
"""

from matplotlib import cm

def get_cmap_rgb(cmap, n_colors=256):
    """Enter a matplotlib colormap name, return rgb array

    Parameters
    ----------
    cmap : str or colormap object
        if matplotlib color, should be string to make sure
        n_colors is selected correctly. If cmocean, pass the object

        e.g.
            matplotlib :  get_cmap_rgb('viridis',10)
            cmocean : get_cmap_rgb(cmocean.cm.thermal,10)
    n_colors : int, optional
        number of color levels in color map
    """

    return cm.get_cmap(cmap,n_colors)(range(n_colors))
