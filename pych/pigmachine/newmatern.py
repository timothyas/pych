import os
import numpy as np
import xarray as xr
import xmitgcm

from MITgcmutils import wrmds
import ecco_v4_py
import scipy.special

# Some issues
# - with LLC, ordering by k is not the same as ordering by Z ... so this would not work in the OIDriver ...
# - Here... not worrying about any of this

class MaternField():

    def __init__(self, xdalike, n_range, horizontal_factor,
                 isotropic=False):

        raise DeprecationWarning("Using implementation in separate repo")
#        self.xdalike = xdalike
#        self.n_range = n_range
#        self.horizontal_factor = horizontal_factor
#        self.isotropic = isotropic
#
#        # Set some properties that don't change
#        self.dims = xdalike.dims
#        self.llc = 'face' in xdalike.dims or 'tile' in xdalike.dims
#        self.n_dims = xdalike.ndim - 1 if self.llc else xdalike.ndim
#        self.xyz = _get_dims(self.dims)
#
#        try:
#            assert self.n_dims == 2 or self.n_dims == 3
#        except:
#            raise TypeError(f"Only 2D or 3D data, got n_dims={self.n_dims} with data {self.xdalike}")
#
#        if self.llc and self.n_dims == 3 and self.xyz['z'] != 'k':
#            raise NotImplementedError(f"vertical dimension needs to be 'k' due to line 1709 in xmitgcm.utils...")
#
#        if self.llc and self.n_dims == 2 and self.xyz['z'] is not None:
#            raise NotImplementedError(f"Due to writing routines (and probably other problems), can't do X-Z or Y-Z slice in LLC grid ... it's a global grid anyway!")
#
#        self.mean_differentiability = 1/2 if self.n_dims == 3 else 1
#        self.delta_hat = 8*self.mean_differentiability / (self.n_range**2)
#
#        denom = scipy.special.gamma(self.mean_differentiability + self.n_dims/2) * \
#                ((4*np.pi)**(self.n_dims/2)) * \
#                self.delta_hat**self.mean_differentiability
#        self.ideal_variance = scipy.special.gamma(self.mean_differentiability) / denom
#
#        self.horizontal_length = _horizontal_length_scale(xdalike)
#        self.vertical_length = _vertical_length_scale(xdalike)
#        self.cell_volume = self.get_cell_volume()
#
#
#        self.Phi, self.detPhi = self.get_deformation_jacobian()
#        self.rhs_factor = 1 / np.sqrt(self.detPhi) / np.sqrt(self.cell_volume)
#        self.delta = (self.delta_hat / self.detPhi ).broadcast_like(xdalike)
#        self.delta.name = 'delta'
#
#        K = {}
#        for key, val in self.Phi.items():
#            K[key] = 1 / self.detPhi * self.Phi[key]**2
#        self.K = K

        # This isn't used... could delete...
        #self.aspect_ratio = _aspect_ratio(self.horizontal_length, self.vertical_length)


    def ideal_correlation(self, distance):
        """Compute correlation in ideal space"""

        factor = 2 ** (1-self.mean_differentiability) / scipy.special.gamma(self.mean_differentiability)
        arg = np.sqrt(8*self.mean_differentiability)/self.n_range * distance

        left = arg ** self.mean_differentiability
        right = scipy.special.kv(self.mean_differentiability, arg)
        return factor * left * right


    def get_deformation_jacobian(self):
        """Get Jacobian of deformation tensor, and its determinant"""

        ux = (self.horizontal_factor * self.horizontal_length).broadcast_like(self.xdalike)
        vy = (self.horizontal_factor * self.horizontal_length).broadcast_like(self.xdalike)
        wz = self.vertical_length.copy().broadcast_like(self.xdalike)

        if self.n_dims == 2:

            if self.xyz['z'] is None:
                # X-Y plane
                Phi = {'ux':ux/self.horizontal_factor,'vy':vy/self.horizontal_factor}
                det = Phi['vy'] * Phi['ux']

            elif self.xyz['y'] is None:
                # X-Z plane
                Phi = {'ux':ux,'wz':wz}
                det = wz*ux

            elif self.xyz['x'] is None:
                # Y-Z plane
                Phi = {'vy':vy,'wz':wz}
                det = wz*vy

            else:
                raise TypeError('getPhi dims problem 2d')

        else:
            Phi = {'ux':ux,'vy':vy,'wz':wz}
            det = wz*vy*ux

        # --- For isotropic case, rewrite Jacobian with ones
        if self.isotropic:
            isoPhi = {}
            for key,val in Phi.items():
                isoPhi[key] = 1

            Phi = isoPhi
            det = 1
        return Phi, det


    def get_cell_volume(self):

        if self.n_dims == 2:
            if self.xyz['z'] is None:
                V = self.horizontal_length**2

            else:
                V = self.vertical_length * self.horizontal_length

        else:
            V = self.vertical_length * self.horizontal_length**2

        return V


    def write_binaries(self, write_dir, smoothOpNb, dtype=None):

        if not os.path.isdir(write_dir):
            os.makedirs(write_dir)

        dataprec = str(self.delta.dtype) if dtype is None else dtype

        # Write tensor
        for key in self.K.keys():
            writeme = self.prepare_to_write(self.K[key])

            fname = os.path.join(write_dir, f"smooth{self.n_dims}DK{key}{smoothOpNb:03}")
            wrmds(fname, writeme, dataprec=dataprec)

        # Write constants
        for name, fld in zip(["Delta", "RandNorm"],
                             [self.delta, self.rhs_factor]):
            writeme = self.prepare_to_write(fld)
            fname = os.path.join(write_dir, f"smooth{self.n_dims}D{name}{smoothOpNb:03}")
            wrmds(fname, writeme, dataprec=dataprec)


    def prepare_to_write(self, xda):

        # If vertical dimension is present, we want to sort it from surface to depth
        # this is "descending" sort for Z (or Zl, Zu, Zp1...) coordinate, and
        # and "ascending" for k (or k_l, etc..)
        writeme = xda.where(~np.isnan(xda), 0.)
        if self.xyz['z'] is not None:
            ascending = False if 'Z' in self.xyz['z'] else True
            writeme = writeme.sortby(self.xyz['z'], ascending=ascending)

        # Deal with LLC
        if self.llc:

            writeme = ecco_v4_py.llc_tiles_to_compact(writeme.values)
            # xmitgcm way
            #hor_dim = self.xyz['x'] if self.xyz['x'] is not None else self.xyz['y']
            #nx = len(self.xdalike[hor_dim])
            #extra_metadata = xmitgcm.utils.get_extra_metadata(domain="llc", nx=nx)

            #facets = xmitgcm.utils.rebuild_llc_facets(writeme, extra_metadata)

            #if self.n_dims == 2:
            #    writeme = xmitgcm.utils.llc_facets_2d_to_compact(facets, extra_metadata)
            #else:
            #    writeme = xmitgcm.utils.llc_facets_3d_spatial_to_compact(facets, self.xyz['z'], extra_metadata)
        else:
            writeme = writeme.values
        return writeme





def _get_dims(dims):
    xyz = {'x': None, 'y': None, 'z': None}
    xlist = ['XC', 'XG', 'i', 'i_g']
    ylist = ['YC', 'YG', 'j', 'j_g']
    zlist = ['Z', 'Zl', 'Zu', 'Zp1', 'k', 'k_l', 'k_u', 'kp1']

    for key, this_list in zip(xyz.keys(), [xlist, ylist, zlist]):
        which_one = [a in dims for a in this_list]
        if any(which_one):
            assert np.sum(which_one) == 1, "Yo we got more than one of the same dim..."
            xyz[key] = this_list[which_one.index(True)]

    return xyz


def _aspect_ratio(L, H):
    aspect = H / L
    aspect.name = 'aspect'
    return aspect


def _horizontal_length_scale(xds):
    """Get horizontal length scale from data array or dataset"""

    xds = xds.to_dataset(name='tmp') if isinstance(xds, xr.DataArray) else xds

    if 'rA' in xds:
        L = np.sqrt(xds['rA'])
    elif 'dyG' in xds:
        L = xds['dyG']
    elif 'dxG' in xds:
        L = xds['dxG']
    else:
        raise TypeError(f"Couldn't find recognizable horizontal length scale in dataset: {xds}")

    L.name = 'L'
    return L


def _vertical_length_scale(xds):
    xds = xds.to_dataset(name='tmp') if isinstance(xds, xr.DataArray) else xds

    choices = ['drF','drC']
    options = [x for x in xds.reset_coords().keys() if x in choices]

    if len(options) == 0:
        raise TypeError(f"Couldn't find recognizable vertical length scale in dataset: {xds}")

    H = xds[options[0]].copy()
    H.name = 'H'
    return H
