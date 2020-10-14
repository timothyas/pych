"""
A collection of miscellanious utility functions
"""

from fileinput import FileInput
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

def search_and_replace(fname,search,replace):
    """find and replace strings within text file

    Parameters
    ----------
    fname : str
        path to text file to search and replace text
    search, replace : str
        strings to look for and replace with inside the text file, line by line

    Returns
    -------
    found : bool
        True if the search string was found anywhere in the file, and replaced
        or if replace is already there
    """

    found = False
    with FileInput(fname, inplace=True, backup='.bak') as file:
        for line in file:
            if search in line or replace in line:
                found=True
                print(line.replace(search, replace), end='')
            else:
                print(line,end='')
    return found
