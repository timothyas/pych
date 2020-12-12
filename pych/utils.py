"""
A collection of miscellanious utility functions
"""

from fileinput import FileInput
from matplotlib import cm
from shutil import which
import os

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

def write_m1qn3_makefile(write_dir,mitgcm_src,mitgcm_build,Nctrl,
                   compiler='ifort',template='Makefile.template'):

    max_independ = f'    -DMAX_INDEPEND={Nctrl}\t\t\\\n'

    build_include = '\t\t-I'+mitgcm_build+'\n'

    # --- Get FC (compiler) and FFLAGS from mitgcm_build/Makefile
    fflags=None
    fc=None
    md=None
    with open(os.path.join(mitgcm_build,'Makefile'),'r') as f:
        for line in f.readlines():
            if "F77_SRC_FILES" in line:
                break
            elif "FFLAGS =" in line or "FFLAGS=" in line:
                fflags=line
            elif "FC =" in line or "FC=" in line:
                fc=line
            elif "MAKEDEPEND =" in line or "MAKEDEPEND=" in line:
                md=line

    for fld,name in zip([fflags,fc,md],['FFLAGS','FC','MAKEDEPEND']):
        if fld is None:
            raise TypeError(f'Could not find {name} from Makefile in {mitgcm_build}')

    file_content=''
    with open(template,'r') as f:
        for line in f.readlines():
            if 'MAKEDEPEND=' in line:
                file_content+=md
            elif 'FC=' in line:
                file_content+=fc
            elif 'FFLAGS=' in line:
                file_content+=fflags
            elif 'MAX_INDEPEND=' in line:
                file_content+=max_independ
            elif '-I' in line and 'MITgcm/verification' in line:
                file_content+=build_include
            else:
                file_content+=line

    fname=os.path.join(write_dir,'Makefile')
    with open(fname,'w') as f:
        f.write(file_content)
