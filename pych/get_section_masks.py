"""
Module for computing mask defining great circle arc between two endpoints
Specifically for stereographic projection!
"""

import warnings
import numpy as np
import xarray as xr
from ecco_v4_py import scalar_calc

# -------------------------------------------------------------------------------
# Main function to compute section masks 
# -------------------------------------------------------------------------------

def get_section_line_masks(pt1, pt2, cds, grid):
    """Compute 2D mask with 1's along great circle line 
    from lat/lon1 -> lat/lon2

    Parameters
    ----------
    pt1, pt2 : tuple or list with 2 floats
        [longitude, latitude] or (longitude, latitude) of endpoints
    cds : xarray Dataset
        containing grid coordinate information, at least XC, YC
    grid : xgcm grid object

    Returns
    -------
    section_mask : xarray DataArray
        2D mask along section
    """

    # Get cartesian coordinates of end points 
    x1, y1, z1 = _convert_stereo_to_cartesian(pt1[0],pt1[1])
    x2, y2, z2 = _convert_stereo_to_cartesian(pt2[0],pt2[1])

    # Compute rotation matrices
    # 1. Rotate around x-axis to put first point at z = 0
    theta_1 = np.arctan2(-z1, y1)
    rot_1 = np.vstack(( [1, 0, 0],
                        [0, np.cos(theta_1),-np.sin(theta_1)],
                        [0, np.sin(theta_1), np.cos(theta_1)]))

    x1, y1, z1 = _apply_rotation_matrix(rot_1, (x1,y1,z1))
    x2, y2, z2 = _apply_rotation_matrix(rot_1, (x2,y2,z2))

    # 2. Rotate around z-axis to put first point at y = 0
    theta_2 = np.arctan2(x1,y1)
    rot_2 = np.vstack(( [np.cos(theta_2),-np.sin(theta_2), 0],
                        [np.sin(theta_2), np.cos(theta_2), 0],
                        [0, 0, 1]))

    x1, y1, z1 = _apply_rotation_matrix(rot_2, (x1,y1,z1))
    x2, y2, z2 = _apply_rotation_matrix(rot_2, (x2,y2,z2))

    # 3. Rotate around y-axis to put second point at z = 0
    theta_3 = np.arctan2(-z2, -x2)
    rot_3 = np.vstack(( [ np.cos(theta_3), 0, np.sin(theta_3)],
                        [ 0, 1, 0],
                        [-np.sin(theta_3), 0, np.cos(theta_3)]))

    x1, y1, z1 = _apply_rotation_matrix(rot_3, (x1,y1,z1))
    x2, y2, z2 = _apply_rotation_matrix(rot_3, (x2,y2,z2))

    # Now apply rotations to the grid 
    # and get cartesian coordinates at cell centers 
    xc, yc, zc = _rotate_the_grid(cds.XC, cds.YC, rot_1, rot_2, rot_3)

    # Interpolate for x,y to west and south edges
    xw = grid.interp(xc, 'X', boundary='fill')
    yw = grid.interp(yc, 'X', boundary='fill')
    xs = grid.interp(xc, 'Y', boundary='fill')
    ys = grid.interp(yc, 'Y', boundary='fill')

    # Compute the great circle mask, covering the entire globe
    maskC = scalar_calc.get_edge_mask(zc>0,grid) 
    maskW = grid.diff( 1*(zc>0), 'X', boundary='fill')
    maskS = grid.diff( 1*(zc>0), 'Y', boundary='fill')

    # Get section of mask pt1 -> pt2 only
    maskC = _calc_section_along_full_arc_mask(maskC, x1, y1, x2, y2, xc, yc)
    maskW = _calc_section_along_full_arc_mask(maskW, x1, y1, x2, y2, xw, yw)
    maskS = _calc_section_along_full_arc_mask(maskS, x1, y1, x2, y2, xs, ys)

    return maskC, maskW, maskS


# -------------------------------------------------------------------------------
#
# All functions below are non-user facing
#
# -------------------------------------------------------------------------------
# Helper functions for computing section masks 
# -------------------------------------------------------------------------------

def _calc_section_along_full_arc_mask( mask, x1, y1, x2, y2, xg, yg ):
    """Given a mask which has a great circle passing through 
    pt1 = (x1, y1) and pt2 = (x2,y2), grab the section just connecting pt1 and pt2

    Parameters
    ----------
    mask : xarray DataArray
        2D LLC mask with 1's along great circle across globe, crossing pt1 and pt2
    x1,y1,x2,y2 : scalars
        cartesian coordinates of rotated pt1 and pt2. Note that z1 = z2 = 0
    xg, yg : xarray DataArray
        cartesian coordinates of the rotated horizontal grid

    Returns
    -------
    mask : xarray DataArray
        mask with great arc passing from pt1 -> pt2
    """

    theta_1 = np.arctan2(y1,x1)
    theta_2 = np.arctan2(y2,x2)
    theta_g = np.arctan2(yg,xg)

    if theta_2 < 0:
        theta_g = theta_g.where( theta_g > theta_2, theta_g + 2*np.pi )
        theta_2 = theta_2 + 2 * np.pi

    if (theta_2 - theta_1) <= np.pi:
        mask = mask.where( (theta_g <= theta_2) & (theta_g >= theta_1), 0)
    else:
        mask = mask.where( (theta_g > theta_2) | (theta_g < theta_1), 0)

    return mask

def _rotate_the_grid(sx, sy, rot_1, rot_2, rot_3):
    """Rotate the horizontal grid at lon, lat, via rotation matrices rot_1/2/3

    Parameters
    ----------
    sx, sy : xarray DataArray
        giving longitude, latitude in degrees of LLC horizontal grid
    rot_1, rot_2, rot_3 : np.ndarray
        rotation matrices

    Returns
    -------
    xg, yg, zg : xarray DataArray
        cartesian coordinates of the horizontal grid
    """

    # Get cartesian of 1D view of lat/lon
    sx_v = sx.values.ravel()
    sy_v = sy.values.ravel()
    if len(sx_v) != len(sy_v):
        sx_v,sy_v = np.meshgrid(sx_v,sy_v)
        sx_v = sx_v.ravel()
        sy_v = sy_v.ravel()
    xg, yg, zg = _convert_stereo_to_cartesian(sx_v,sy_v)

    # These rotations result in:
    #   xg = 0 at pt1
    #   yg = 1 at pt1
    #   zg = 0 at pt1 and pt2 (and the great circle that crosses pt1 & pt2)
    xg, yg, zg = _apply_rotation_matrix(rot_1, (xg,yg,zg))
    xg, yg, zg = _apply_rotation_matrix(rot_2, (xg,yg,zg))
    xg, yg, zg = _apply_rotation_matrix(rot_3, (xg,yg,zg))

    # Remake into LLC xarray DataArray
    tmp = sy*sx # template
    def make_xda(fld,template):
        return xr.DataArray(np.reshape(fld,template.shape),
                            coords=tmp.coords,dims=tmp.dims)
    xg = make_xda(xg,tmp)
    yg = make_xda(yg,tmp)
    zg = make_xda(zg,tmp)

    return xg, yg, zg

def _apply_rotation_matrix(rot_mat,xyz):
    """Apply a rotation matrix to a tuple x,y,z (each x,y,z possibly being arrays)

    Parameters
    ----------
    rot_mat : numpy matrix
        2D matrix defining rotation in 3D cartesian coordinates
    xyz : tuple of arrays
        with cartesian coordinates

    Returns
    -------
    xyz_rot : tuple of arrays
        rotated a la rot_mat
    """

    # Put tuple into matrix form
    xyz_mat = np.vstack( (xyz[0],xyz[1],xyz[2]) )

    # Perform rotation
    xyz_rot_mat = np.matmul( rot_mat, xyz_mat )

    # Either return as scalar or array
    if np.isscalar(xyz[0]):
        return xyz_rot_mat[0,0], xyz_rot_mat[1,0], xyz_rot_mat[2,0]
    else:
        return xyz_rot_mat[0,:], xyz_rot_mat[1,:], xyz_rot_mat[2,:]


def _convert_stereo_to_cartesian(sx, sy):
    """Convert ...

    Parameters
    ----------
    sx,sy : numpy or dask array
        stereographic projection plane x,y coordinates

    Returns
    -------
    x : numpy or dask array
        x- component of cartesian coordinate
    y : numpy or dask array
    z : numpy or dask array
    """

    # Get cartesian
    x = 2*sx / (1 + sx**2 + sy**2)
    y = 2*sy / (1 + sx**2 + sy**2)
    z = (-1 + sx**2 + sy**2)/(1 + sx**2 + sy**2)

    return x, y, z
