"""
OIDriver class, which defines an object to aid in herding these experiments
Caveats:
    - Assumes control parameter is cell centered, (see e.g. pickup where sorted)
    - obs_mask/obs_std and mymodel have to have dimension order like 'Z','Y','X'
    - get_matern_dataset calls are loaded into memory so I can graduate faster!
    - need to note that mymodel has its dimensions ordered the way the MITgcm
        likes it (i.e. Z starts at 0), but the ctrl = ControlField mask has
        dimensions ordered ... in order! i.e. Z goes from bottom to top.

"""

import os
import sys
import json
import subprocess
import warnings
import numpy as np
from shutil import rmtree
from time import sleep
from scipy import linalg
import xarray as xr
from xmitgcm import open_mdsdataset
from MITgcmutils import rdmds, wrmds
import rosypig as rp

from . import matern
from .. import pigmachine as pm
from .notation import get_nice_attrs

class OIDriver:
    """Defines the driver for getting an Eigenvalue decomposition
    associated with the optimal interpolation problem

    Example
    -------
    To start an experiment

        >>> myoi = OIDriver('myoi')
        >>> myoi.start(dirsdict,simdict,mymodel,obs_std)

    see the start method for descriptions on input parameters.
    By default, this starts the whole show, but a different starting point
    can be specified.

    To pick up at a specific stage, e.g. specified by the string mystage

        >>> myoi = OIDriver('myoi',stage=mystage)

    """
    slurm = {'be_nice':True,
             'max_job_submissions':9,
             'dependency':'afterany'}
    n_small = 950
    n_over = 50
    n_rand = 1000
    n_norm = 1000
    dataprec = 'float64'
    NxList  = [5, 10, 15, 20, 30, 40]
    xiList  = [0.5,   1,   2]#,   5]
    sorDict = {0.5:1.8, 1:1.6, 2:1.3, 5:1.2}
    n_sigma = 9
    sigma = 10**np.linspace(-5,-3,n_sigma)
    smoothOpNb = 1
    smooth2DDims = 'yzw'
    jacobi_max_iters = 20000
    conda_env = 'py38_tim'
    doRegularizeDebug = False
    doRayleigh = False
    def __init__(self, experiment,stage=None):
        """Should this initialize? Or do we want another method to do it?

        Parameters
        ----------
        experiment : str
            an identifying name for the experiment
            this should identify:
            1. the observable quantity
            2. the observation set used (e.g. just obs from one year, or multiple?)
            3. the observational uncertainty profile used
        stage : str
            the stage to start or pickup at when this is called

            'range_approx_one' : first stage of range approximation
            'range_approx_two'
            'basis_projection_one'
            'basis_projection_two'
            'do_the_evd'
            'prior_to_misfit'
            'solve_for_map'
            'save_the_map'
        """
        self.experiment = experiment
        self._send_to_stage(stage)


    def start(self,dirs,dsim,
              mymodel,m0,ctrl_ds,obs_mask,obs_mean,obs_std,
              startat='range_approx_one',**kwargs):
        """Start the experiment by writing everything we need to file,
        to be read in later by "pickup"

        Parameters
        ----------
        dirs : dict
            containing the key directories for the experiment
            'main_run' : the directory where MITgcm files are written
                and simulations are started
            'netcdf' : where to save netcdf files when finished
                Note: a separate tmp/ directory is created inside for
                intermittent saves
        dsim : dict
            containing the base parameters for a rosypig.Simulation
            'machine', 'n_procs', 'exe_file', 'binary_dir', 'pickup_dir', 'time'
            see rosypig.Simulation for details
        mymodel : xarray DataArray
            a mask field with True denoting domain wet points, e.g. maskC
            Note: the dimension order should be appropriate for MITgcm read/write,
            with vertical going from top to bottom
            Note: dimensions are assumed to be ordered like
            (vertical, meridional, zonal)
        m0 : xarray DataArray
            with the initial guess
            Note: must have attribute "standard_name" with identifying name, e.g.
            m0.attrs['standard_name'] = 'zeros'
        ctrl_ds : xarray DataArray
            dataset with the FULL coordinate dataset, for evaluating the laplacian
        obs_mask : xarray DataArray
            mask field denoting where observations are taken, on an "observation grid"
            Note: dimensions are assumed to be ordered like
            (vertical, meridional, zonal)
        obs_mean, obs_std : xarray DataArray
            containing observation mean and uncertainties
            (of course, at the mask points!)
        startat : str, optional
            stage to start at
        kwargs
            Are passed to override the default class attributes
        """

        if 'standard_name' not in m0.attrs.keys():
            raise TypeError('No identifying name for initial guess, add as standard_name in DataArray attributes')
        if m0.attrs['standard_name']=='':
            raise TypeError('Blank name for initial guess, add something useful as standard_name in DataArray attributes')

        try:
            assert self.n_rand <= self.n_norm
        except AssertionError as err:
            err.args += ('n_rand (# samples for REVD) must be <= n_norm '+\
                    '(# samples for covariance filter normalization')
            raise err
        try:
            assert self.n_small+self.n_over == self.n_rand
        except AssertionError as err:
            err.args += ('n_rand must equal small dimension + # for oversampling')
            raise err

        # --- Add tmp netcdf directory
        dirs['nctmp'] = dirs['netcdf']+'/tmp'
        for mydir in ['json',dirs['netcdf'],dirs['nctmp'],dirs['main_run']]:
            _dir(mydir);

        # --- Write the directories
        def write_json(mydict,mysuff):
            json_file = f'json/{self.experiment}'+mysuff
            with open(json_file,'w') as f:
                json.dump(mydict, f)

        write_json(dirs,'_dirs.json')
        write_json(dsim,'_sim.json')
        if kwargs !={}:
            write_json(kwargs,'_kwargs.json')

        # --- Write ctrl and obs datasets to netcdf
        m0 = m0.reindex_like(mymodel)
        myctrl = xr.Dataset({'mymodel':mymodel,'m0':m0})
        myctrl.to_netcdf(dirs['nctmp']+f'/{self.experiment}_ctrl.nc')
        ctrl_ds.to_netcdf(dirs['nctmp']+f'/{self.experiment}_cds.nc')
        myobs = xr.Dataset({'obs_mask':obs_mask,'obs_std':obs_std,'obs_mean':obs_mean})
        myobs.to_netcdf(dirs['nctmp']+f'/{self.experiment}_obs.nc')

        # --- "pickup" experiment at startat
        self.pickup()
        startsim = rp.Simulation('startmeup',**dsim)
        self.submit_next_stage(next_stage=startat,mysim=startsim)


    def pickup(self):
        """Read in the files saved in start, prepare self for next stage
        """

        # --- Read
        def read_json(mysuff):
            json_file = f'json/{self.experiment}' + mysuff
            if not os.path.isfile(json_file):
                return None

            with open(json_file,'r') as f:
                mydict = json.load(f)
            return mydict

        dirs = read_json('_dirs.json')
        dsim = read_json('_sim.json')
        kwargs = read_json('_kwargs.json')

        myctrl = xr.open_dataset(dirs['nctmp']+f'/{self.experiment}_ctrl.nc')
        cds = xr.open_dataset(dirs['nctmp']+f'/{self.experiment}_cds.nc')
        myobs = xr.open_dataset(dirs['nctmp']+f'/{self.experiment}_obs.nc')

        modeldims=list(myctrl['mymodel'].dims)
        obsdims = list(myobs['obs_mask'].dims)

        # --- Name for final stages, where initial guess matters
        self.expWithInitGuess = self.experiment+'_'+myctrl.m0.attrs['standard_name']

        # --- Carry these things around
        self.dirs = dirs
        self.dsim = dsim
        self.mymodel = myctrl['mymodel']
        self.cds = cds

        self.ctrl = rp.ControlField('ctrl',self.mymodel.sortby(modeldims))
        self.obs = rp.ControlField('obs',
                                   myobs['obs_mask'].sortby(obsdims).astype('bool'))

        self.m0 = self.ctrl.pack(myctrl['m0'].sortby(modeldims))
        self.obs_mean = self.obs.pack(myobs['obs_mean'].sortby(obsdims))
        self.obs_std = self.obs.pack(myobs['obs_std'].broadcast_like(self.obs.mask).sortby(obsdims))
        self.obs_std_inv = self.obs_std**-1
        self.obs_var_inv = self.obs_std**-2

        # --- Get the interpolation operator
        mdimssort = [self.mymodel[dim].sortby(dim) for dim in modeldims]
        odimssort = [myobs[dim].sortby(dim) for dim in obsdims]
        self.F = pm.interp_operator_2d(mdimssort,odimssort,
                                       pack_index_in=self.ctrl.wet_ind,
                                       pack_index_out=self.obs.wet_ind)

        # --- If kwargs exist, use to rewrite default attributes
        if kwargs is not None:
            for key,val in kwargs.items():
                if key == 'sigma' and isinstance(val,list):
                    val = np.array(val)
                self.__dict__[key] = val

    def _send_to_stage(self,stage):
        possible_stages = ['range_approx_one','range_approx_two',
                           'basis_projection_one','basis_projection_two',
                           'do_the_evd','prior_to_misfit',
                           'solve_for_map','save_the_map',
                           'calc_rayleigh']

        if stage in possible_stages:
            self.pickup()
            eval(f'self.{stage}()')
        else:
            if stage is not None:
                raise NameError(f'Incorrectly specified stage: {stage}.\n'+\
                    'Available possibilities are: '+str(possible_stages))


# ---------------------------------------------------------------------
# Range Approximation
# ---------------------------------------------------------------------
    def range_approx_one(self):
        """Define the Laplacian-like operator A = delta - div(kappa grad( ))
        and use the MITgcm to:

            1. generate n_rand gaussian random samples, g_i
            2. solve the linear system for z_i: Az_i = g_i
               to get the range of A^{-1}
            3. use to compute the sample variance of A^{-1}

        ... pass to the next step
        """
        jid_list = []
        for Nx in self.NxList:
            for xi in self.xiList:

                _, write_dir, run_dir = self._get_dirs('range_approx_one',Nx,xi)

                self.smooth_writer(write_dir, xi=xi, smooth_apply=False,
                                    num_inputs=self.n_norm)
                matern.write_matern(write_dir,
                                    smoothOpNb=self.smoothOpNb,
                                    Nx=Nx,mymask=self.ctrl.mask,
                                    xdalike=self.mymodel,xi=xi)

                sim = rp.Simulation(name=f'{Nx:02}dx_{xi:02}xi_oi_ra1',
                                    run_dir=run_dir,
                                    obs_dir=write_dir,
                                    **self.dsim)

                # launch job
                sim.link_to_run_dir()

                sim.write_slurm_script()
                jid = sim.submit_slurm(**self.slurm)
                jid_list.append(jid)

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='range_approx_two',mysim=sim,
                               jid_depends=jid_list)

    def range_approx_two(self):
        """Given the output from range_approx_one, the n_rand vectors z_i

            1. Read in all z_i, and the inverse square root of the
                variance associated with A^{-1} as the filternorm, X
            2. For each z_i compute:
                w_i = XF^T\Gamma_{obs}^{-1}FX z_i
            3. Send to the MITgcm to solve the linear system for y_i:
                A y_i = w_i

        ... pass to next step
        """

        jid_list = []
        dslistNx = []
        for Nx in self.NxList:
            dslistXi = []
            for xi in self.xiList:

                # --- Prepare reading and writing
                read_dir, write_dir, run_dir = self._get_dirs('range_approx_two',Nx,xi)

                # --- Read in samples and filternorm
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_norm),
                                               read_filternorm=True)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();
                ds['filternorm'].load();
                evds = rp.RandomDecomposition(n_ctrl=self.ctrl.n_wet,
                                              n_obs=self.obs.n_wet,
                                              n_small=self.n_small,
                                              n_over=self.n_over)

                # Add Nx and xi dimensions
                evds['filternorm'] = rp.to_xda(self.ctrl.pack(ds['filternorm']),evds,
                                               expand_dims={'Nx':Nx,'xi':xi})
                dslistXi.append(evds)

                # --- Now apply obserr & filternorm weighted interpolation operator
                smooth2DInput = []
                filternorm = self.ctrl.pack(ds['filternorm'].values)
                F_norm = self.F*filternorm
                FtWF = (F_norm.T * self.obs_var_inv) @ F_norm
                for s in range(self.n_rand):
                    g_out = FtWF @ self.ctrl.pack(ds['ginv'].isel(sample=s))
                    g_out = self.ctrl.unpack(g_out)
                    smooth2DInput.append(g_out)

                # --- Write matern operator and submit job
                smooth2DInput = xr.concat(smooth2DInput,dim='sample')
                self.smooth_writer(write_dir, xi=xi, num_inputs=self.n_rand)
                jid,sim =self.submit_matern(fld=smooth2DInput,
                                            Nx=Nx,xi=xi,
                                            write_dir=write_dir,
                                            run_dir=run_dir,
                                            run_suff='ra2')
                jid_list.append(jid)

            # --- Keep appending the dataset with filternorm for safe keeping
            dslistNx.append(xr.concat(dslistXi,dim='xi'))

        newds = xr.concat(dslistNx,dim='Nx')
        newds['F'] = rp.to_xda(self.F,newds)
        newds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_filternormInterp.nc')

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='basis_projection_one',
                               jid_depends=jid_list,mysim=sim)


# ---------------------------------------------------------------------
# Form and project onto low dimensional subspace
# ---------------------------------------------------------------------
    def basis_projection_one(self):
        """From the previous steps we have the vectors y_i, which form the columns
        of the matrix Y. This is used to approximate the range of:

            M = A^{-1}XF^T\Gamma_{obs}^{-1}FXA^{-1}

            1. orthonormalize the range approximator Y -> Q
            2. start projecting M onto this orthonormal basis by
                sending to the MITgcm to solve for z_i
                    A z_i = q_i

        ... send to next stage
        """

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_filternormInterp.nc')
        jid_list=[]
        dslistNx = []
        for Nx in self.NxList:
            dslistXi = []
            for xi in self.xiList:

                # --- Prepare read and write
                read_dir, write_dir, run_dir = self._get_dirs('basis_projection_one',Nx,xi)

                # --- Read samples from last stage, form orthonormal basis
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)

                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                # Get randNorm
                C, _ = matern.get_matern(Nx=Nx,xi=xi,mymask=self.ctrl.mask)

                # Form Y, range approximator
                Y = []
                for s in range(self.n_rand):
                    output = C['randNorm']*ds['ginv'].sel(sample=s)
                    Y.append(self.ctrl.pack(output))
                Y = np.array(Y).T

                # do some linalg
                Q,_ = np.linalg.qr(Y)

                # prep for saving
                tmpds = xr.Dataset()
                tmpds['Y'] = rp.to_xda(Y,evds,expand_dims={'Nx':Nx,'xi':xi})
                tmpds['Q'] = rp.to_xda(Q,evds,expand_dims={'Nx':Nx,'xi':xi})
                dslistXi.append(tmpds)

                # --- Write out Q and submit job
                smooth2DInput = []
                for i in range(self.n_rand):
                    q = C['randNorm']*self.ctrl.unpack(Q[:,i],fill_value=0.)
                    smooth2DInput.append(q)

                smooth2DInput = xr.concat(smooth2DInput,dim='sample')
                self.smooth_writer(write_dir, xi=xi, num_inputs=self.n_rand)
                jid,sim =self.submit_matern(fld=smooth2DInput,
                                            Nx=Nx,xi=xi,
                                            write_dir=write_dir,
                                            run_dir=run_dir,
                                            run_suff='proj1')
                jid_list.append(jid)

            # get those datasets
            dslistNx.append(xr.concat(dslistXi,dim='xi'))
        newds = xr.concat(dslistNx,dim='Nx')
        atts = evds.attrs.copy()
        evds = xr.merge([evds,newds])
        evds.attrs = atts
        evds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_proj.nc')

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='basis_projection_two',
                               jid_depends=jid_list,mysim=sim)

    def basis_projection_two(self):
        """From the previous step we have the vectors z_i, which is the
        start of projecting M onto the reduced basis defined by Q, with

            M = A^{-1}XF^T\Gamma_{obs}^{-1}FXA^{-1}

            1. read in z_i
            2. compute w_i = XF^T\Gamma_{obs}^{-1}FX z_i
            3. send to the MITgcm to solve for y_i:
                    A y_i = w_i

        ... send to next stage
        """
        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_proj.nc')
        evds['filternorm'].load();
        evds['F'].load();

        jid_list = []
        for Nx in self.NxList:
            for xi in self.xiList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('basis_projection_two',Nx,xi)

                # --- Read output from last stage, apply obserr,filternorm weight op
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                # Apply operator
                smooth2DInput = []
                filternorm = evds['filternorm'].sel(Nx=Nx,xi=xi).values
                F_norm = evds['F'].values*filternorm
                FtWF = (F_norm.T * self.obs_var_inv) @ F_norm
                for s in range(self.n_rand):
                    # write this out rather than form it every time
                    g_out = FtWF @ self.ctrl.pack(ds['ginv'].sel(sample=s))
                    g_out = self.ctrl.unpack(g_out)
                    smooth2DInput.append(g_out)

                # --- Write out and submit next application
                smooth2DInput = xr.concat(smooth2DInput,dim='sample')
                self.smooth_writer(write_dir, xi=xi, num_inputs=self.n_rand)
                jid,sim =self.submit_matern(fld=smooth2DInput,
                                            Nx=Nx,xi=xi,
                                            write_dir=write_dir,
                                            run_dir=run_dir,
                                            run_suff='proj2')
                jid_list.append(jid)

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='do_the_evd',
                               jid_depends=jid_list,mysim=sim)

# ---------------------------------------------------------------------
# Compute and save the low dimensional EVD
# ---------------------------------------------------------------------
    def do_the_evd(self):
        """From the previous step we have the vectors w_i, which
        forms MQ, where

            M = A^{-1}XF^T\Gamma_{obs}^{-1}FXA^{-1}

            1. read in w_i to form the columns of the matrix W
            2. compute B = Q^T W = Q^T M Q
            3. find the eigenvalue decomposition of B: B = VDV^T
            4. get eigenpairs in order of descending eigenvalues
            5. approximate eigenvectors of M are V <- QV

        ... send to next stage
        """

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_proj.nc')
        evds['Q'].load();
        dslistNx = []
        for Nx in self.NxList:
            dslistXi = []
            for xi in self.xiList:

                # --- Prepare directories
                read_dir, _, _ = self._get_dirs('do_the_evd',Nx,xi)

                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                # Get randNorm
                C, _ = matern.get_matern(Nx=Nx,xi=xi,mymask=self.ctrl.mask)

                # Get Hm Q
                HmQ = []
                for s in range(self.n_rand):
                    output = C['randNorm']*ds['ginv'].sel(sample=s)
                    HmQ.append(self.ctrl.pack(output))
                HmQ = np.array(HmQ).T

                # Apply Q^T
                Q = evds['Q'].sel(Nx=Nx,xi=xi).values
                B = Q.T @ HmQ

                # Do the EVD
                D,Vhat = linalg.eigh(B)

                # eigh returns in ascending order, reverse
                D = D[::-1]
                Vhat = Vhat[:,::-1]

                V = Q @ Vhat

                tmpds = xr.Dataset()
                tmpds['V'] = rp.to_xda(V,evds,expand_dims={'Nx':Nx,'xi':xi})
                tmpds['D'] = rp.to_xda(D,evds,expand_dims={'Nx':Nx,'xi':xi})
                dslistXi.append(tmpds)

            # --- Continue saving eigenvalues
            dslistNx.append(xr.concat(dslistXi,dim='xi'))

        newds = xr.concat(dslistNx,dim='Nx')
        atts = evds.attrs.copy()
        evds = xr.merge([newds,evds])
        evds.attrs = atts
        evds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_evd.nc')

        # --- Pass on to next stage
        sim = rp.Simulation(name='evd',**self.dsim)
        self.submit_next_stage(next_stage='prior_to_misfit',mysim=sim)

    def prior_to_misfit(self):
        """Optional to re-run this from the last stage with another initial guess
        Given the eigenvalue decomposition of

            M = A^{-1}XF^T\Gamma_{obs}^{-1}FXA^{-1} = VDV^T

        Start computing the MAP point:

            m* = m_0 + P^{1/2} (I-VD_{inv}V^T) P^{T/2} F^T \Gamma_{obs}^{1/2}( d - Fm_0)
        for initial guess m_0, and prior covariance given by P = P^{1/2}P^{T/2}, and
            P^{1/2} = sigma  X A^{-1}

            1. Compute misfit2model = X F^T \Gamma_{obs}^{1/2}( d - F m_0 )
            2. Pass to MITgcm to solve for z_i:
                A z_i = misfit2model

        ... pass to next step
        """

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_evd.nc')
        evds['filternorm'].load();

        # --- Compute initial misfit: F m_0 - d, and weighted versions
        initial_misfit = self.obs_mean - evds['F'].values @ self.m0

        # --- Map back to model grid and apply posterior via EVD
        misfit2model = evds['F'].T.values @ (self.obs_var_inv * initial_misfit)

        jid_list = []
        for Nx in self.NxList:
            for xi in self.xiList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('prior_to_misfit',Nx,xi)

                # --- Apply prior^T/2
                filternorm = evds['filternorm'].sel(Nx=Nx,xi=xi)
                smooth2DInput = self.ctrl.unpack(filternorm*misfit2model)

                self.smooth_writer(write_dir, xi=xi, num_inputs=1)
                jid,sim = self.submit_matern(fld=smooth2DInput,
                                             Nx=Nx,xi=xi,
                                             write_dir=write_dir,
                                             run_dir=run_dir,
                                             run_suff='p2m')
                jid_list.append(jid)

        self.submit_next_stage(next_stage='solve_for_map',
                               jid_depends=jid_list,mysim=sim)


    def solve_for_map(self):
        """Optional to re-run this from the last stage with another initial guess

        Continue computing the MAP point:

            m* = m_0 + P^{1/2} (I-VD_{inv}V^T) P^{T/2} F^T \Gamma_{obs}^{1/2}( d - Fm_0)
        for initial guess m_0, and prior covariance given by P = P^{1/2}P^{T/2}, and
            P^{1/2} = sigma  X A^{-1}

            1. read in z_i from previous step: z_i = A^{-1}XF^T \Gamma_{obs}^{1/2}( d - Fm_0)
            2. Compute: ivdv = (I - V D_{inv} V^T) sigma  z_i
            2. Pass to MITgcm to solve for w_i:
                A w_i = ivdv

        ... pass to final stage...
        """

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_evd.nc')
        evds['filternorm'].load();

        evds = _add_map_fields(evds,self.sigma,self.doRegularizeDebug)

        jid_list = []
        for Nx in self.NxList:
            for xi in self.xiList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('solve_for_map',Nx,xi)

                # --- Possibly use lower dimensional subspace and get arrays
                V = evds['V'].sel(Nx=Nx,xi=xi).values[:,:self.n_small]

                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                # Get randNorm
                C, _ = matern.get_matern(Nx=Nx,xi=xi,mymask=self.ctrl.mask)

                prior_misfit2model = self.ctrl.pack(C['randNorm']*ds['ginv'])

                smooth2DInput = []
                for s in self.sigma:

                    # --- Possibly account for lower dimensional subspace
                    Dinv = evds['Dinv'].sel(Nx=Nx,xi=xi,sigma=s).values[:self.n_small]

                    # --- Apply (I-VDV^T)
                    vdv = V @ ( Dinv * ( V.T @ prior_misfit2model))
                    ivdv = prior_misfit2model - vdv
                    ivdv = C['randNorm']*self.ctrl.unpack(ivdv)
                    smooth2DInput.append(ivdv)

                # --- Apply prior half
                smooth2DInput = xr.concat(smooth2DInput,dim='sigma')
                self.smooth_writer(write_dir, xi=xi, num_inputs=len(self.sigma))
                jid,sim = self.submit_matern(fld=smooth2DInput,
                                             Nx=Nx,xi=xi,
                                             write_dir=write_dir,
                                             run_dir=run_dir,
                                             run_suff='map')
                jid_list.append(jid)

        evds.to_netcdf(self.dirs['nctmp']+f'/{self.expWithInitGuess}_map.nc')
        self.submit_next_stage(next_stage='save_the_map',
                               jid_depends=jid_list,mysim=sim)

    def save_the_map(self):
        """Finish computing the MAP point:

            m* = m_0 + P^{1/2} (I-VD_{inv}V^T) P^{T/2} F^T \Gamma_{obs}^{1/2}( d - Fm_0)
        for initial guess m_0, and prior covariance given by P = P^{1/2}P^{T/2}, and
            P^{1/2} = sigma  X A^{-1}

            1. read in w_i from previous step: w_i = A^{-1}(I - VD_{inv}V^T) z_i
            2. compute: m_update= sigma^2 Xw_i
            3. m_map = m0 + m_update
            4. compute misfits of every flavor
            5. be like jesus (save it!)

        """
        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.expWithInitGuess}_map.nc')
        evds['filternorm'].load();

        jid_list = []
        for Nx in self.NxList:
            for xi in self.xiList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('save_the_map',Nx,xi)

                # --- Get arrays
                filternorm = self.ctrl.unpack(evds['filternorm'].sel(Nx=Nx,xi=xi))
                filterstd = xr.where(filternorm!=0,filternorm**-1,0.)

                C,K = matern.get_matern(Nx=Nx,mymask=self.ctrl.mask,xi=xi)

                # for rayleigh quotients, get eigenvectors of Hessian (not PPMH)
                if self.doRayleigh:
                    jid,sim = self.get_vm(evds.V,C['randNorm'],Nx,xi)
                    jid_list.append(jid)

                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(len(self.sigma)),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();
                for i,s in enumerate(self.sigma):

                    m_update = (s**2)*filternorm*ds['ginv'].isel(sample=i)
                    mmap = self.m0 + self.ctrl.pack(m_update)

                    # --- Compute m_map - m0, and weighted version
                    dm = filterstd*m_update
                    dm_normalized = (s**-1) * rp.apply_matern_2d( \
                                                fld_in=dm,
                                                mask3D=self.cds.maskC,
                                                mask2D=self.ctrl.mask,
                                                ds=self.cds,
                                                delta=C['delta'],
                                                Kux=None,Kvy=K['vy'],Kwz=K['wz'])
                    dm_normalized = dm_normalized*(C['randNorm']**-1)

                    dm_normalized = rp.to_xda(self.ctrl.pack(dm_normalized),evds)
                    with xr.set_options(keep_attrs=True):
                        evds['m_map'].loc[{'sigma':s,'xi':xi,'Nx':Nx}] = \
                                rp.to_xda(mmap,evds)
                        evds['reg_norm'].loc[{'sigma':s,'xi':xi,'Nx':Nx}] = \
                            np.linalg.norm(dm_normalized,ord=2)

                    if self.doRegularizeDebug:
                        evds = self.regular_debug_plots(evds=evds, \
                                Nx=Nx,xi=xi,sigma=s,m_update=m_update,
                                C=C,K=K)

                    # --- Compute misfits, and weighted version
                    misfits = evds['F'].values @ mmap - self.obs_mean
                    misfits_model_space = rp.to_xda( \
                            mmap - evds['F'].T.values @ self.obs_mean,evds)
                    misfits_model_space = misfits_model_space.where( \
                                            evds['F'].T.values @ self.obs_mean !=0)

                    with xr.set_options(keep_attrs=True):
                        evds['misfits'].loc[{'sigma':s,'xi':xi,'Nx':Nx}] = \
                                rp.to_xda(misfits,evds)
                        evds['misfits_normalized'].loc[{'sigma':s,'xi':xi,'Nx':Nx}] = \
                            rp.to_xda(self.obs_std_inv * misfits,evds)
                        evds['misfit_norm'].loc[{'sigma':s,'xi':xi,'Nx':Nx}] = \
                            np.linalg.norm(self.obs_std_inv * misfits,ord=2)
                        evds['misfits_model_space'].loc[{'sigma':s,'xi':xi,'Nx':Nx}] = \
                            misfits_model_space

        evds.to_netcdf(self.dirs['netcdf']+f'/{self.expWithInitGuess}_map.nc')

        if self.doRayleigh:
            self.submit_next_stage(next_stage='calc_rayleigh',
                                   jid_depends=jid_list,mysim=sim)

# ---------------------------------------------------------------------
# Some optional stuff to compute for post-process analysis
# ---------------------------------------------------------------------
    def regular_debug_plots(self,evds,Nx,xi,sigma,m_update,C,K):

        # --- Extra regularization terms
        var = matern.calc_variance(Nx=Nx)
        SHalfmat = filterstd / np.sqrt(var)
        dm = filterstd*m_update
        rlap = (sigma**-1)*rp.apply_laplacian_2d( \
                fld_in = dm,
                mask3D=self.cds.maskC,
                mask2D=self.ctrl.mask,
                ds=self.cds,
                Kux=None,Kvy=K['vy'],Kwz=K['wz'])
        rdelta =(sigma**-1)* C['delta'] * dm
        rslap = (sigma**-1)*rp.apply_laplacian_2d( \
                fld_in = SHalfmat*m_update,
                mask3D=self.cds.maskC,
                mask2D=self.ctrl.mask,
                ds=self.cds,
                Kux=None,Kvy=K['vy'],Kwz=K['wz'])
        rsdelta = (sigma**-1)*C['delta'] * (SHalfmat * m_update)
        with xr.set_options(keep_attrs=True):
            evds['reg_delta'].loc[{'sigma':sigma,'xi':xi,'Nx':Nx}]= \
                    np.linalg.norm(rdelta,ord=2)
            evds['reg_laplacian'].loc[{'sigma':sigma,'xi':xi,'Nx':Nx}]= \
                    np.linalg.norm(rlap,ord=2)
            evds['reg_S_delta'].loc[{'sigma':sigma,'xi':xi,'Nx':Nx}]= \
                    np.linalg.norm(rsdelta,ord=2)
            evds['reg_S_laplacian'].loc[{'sigma':sigma,'xi':xi,'Nx':Nx}]= \
                    np.linalg.norm(rslap,ord=2)
        return evds

    def get_vm(self,V,randNorm,Nx,xi):
        """get eigenvectors of inverse covariance matrix, i.e. the generalized
        eigenvectors of the matrix pair (Hm, \Gamma_{prior}^{-1})

            v^m_i = \Gamma_{prior}^{1/2} v_i

        where v_i are eigenvectors of prior preconditioned misfit Hessian
        This just does the first part:

            v^1_i = A^{-1}T v_i

        v^m_i = Sigma X v^1_i
        is computed in calc_rayleigh
        """
        # --- Prepare directories
        _, write_dir, run_dir = self._get_dirs('get_vm',Nx,xi)

        # --- Apply prior^T/2
        smooth2DInput = []
        for i in V.rand_ind.values:
            Tvi = randNorm * self.ctrl.unpack(V.sel(rand_ind=i,Nx=Nx,xi=xi))
            smooth2DInput.append(Tvi)

        # --- Apply prior half
        smooth2DInput = xr.concat(smooth2DInput,dim='sample')
        self.smooth_writer(write_dir, xi=xi, num_inputs=len(V.rand_ind))

        jid,sim = self.submit_matern(fld=smooth2DInput,
                                     Nx=Nx,xi=xi,
                                     write_dir=write_dir,
                                     run_dir=run_dir,
                                     run_suff='gvm')
        return jid, sim

    def calc_rayleigh(self):
        """Compute Rayleigh quotients of misfit Hessian and inverse prior
        components of the GN posterior:

            Rm(i) = <v_i , Hm v_i> / <v_i,v_i>
            Rp(i) = <v_i , \Gamma^{-1}_prior v_i> / <v_i,v_i>

        where v_i are eigenvectors of inverse posterior covariance
        """
        def _ray_fields(ds):
            ds['Vm'] = xr.zeros_like(ds['sigma']*ds['V'])
            ds['Rm'] = xr.zeros_like(ds['Vm'].isel(ctrl_ind=0).drop_vars('ctrl_ind'))
            ds['Rp'] = xr.zeros_like(ds['Vm'].isel(ctrl_ind=0).drop_vars('ctrl_ind'))
            return ds

        evds = xr.open_dataset(self.dirs['netcdf']+f'/{self.expWithInitGuess}_map.nc')

        evds['filternorm'].load();
        evds['V'].load();

        # tmporary...
        evds=_ray_fields(evds) if 'Vm' not in evds else evds

        for Nx in self.NxList:
            for xi in self.xiList:
                read_dir, _, _ = self._get_dirs('calc_rayleigh',Nx,xi)

                # --- Get arrays
                X = evds['filternorm'].sel(Nx=Nx,xi=xi).values
                filternorm = self.ctrl.unpack(X)

                # --- load first part from get_vm
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                SXATV = evds['sigma']*filternorm*ds['ginv']

                # get misfit hessian
                FtWF = (evds['F'].values.T * self.obs_var_inv) @ evds['F'].values

                for ri in evds.rand_ind.values:
                    V = evds['V'].sel(Nx=Nx,xi=xi,rand_ind=ri).values
                    pri_numer = V.T @ V

                    for s in self.sigma:
                        Vm = self.ctrl.pack(SXATV.sel(sigma=s,sample=ri))
                        evds['Vm'].loc[{'Nx':Nx,'xi':xi,'sigma':s,'rand_ind':ri}]=Vm

                        # get misfit rayleigh
                        numer = Vm.T @ FtWF @ Vm
                        denom = Vm.T @ Vm

                        evds['Rm'].loc[{'Nx':Nx,'xi':xi,'sigma':s,'rand_ind':ri}] =\
                                numer/denom
                        evds['Rp'].loc[{'Nx':Nx,'xi':xi,'sigma':s,'rand_ind':ri}] =\
                                pri_numer/denom

        evds.to_netcdf(self.dirs['netcdf']+f'/{self.expWithInitGuess}_ray.nc')


# ---------------------------------------------------------------------
# Stuff for organizing each stage of the run
# ---------------------------------------------------------------------
    def smooth_writer(self,write_dir,xi,smooth_apply=True,num_inputs=1000):
        """write the data.smooth file
        """
        ndims = len(self.mymodel.dims)
        smooth = f'smooth{ndims}D'
        alg = 'matern'
        alg = alg if not smooth_apply else alg+'apply'
        if xi not in self.sorDict.keys():
            warnings.warn(f'sor parameter value unknown for xi={xi}, setting sor=1 (=Gauss-Seidel)')
            sor=1
        else:
            sor = self.sorDict[xi]
        file_contents = ' &SMOOTH_NML\n'+\
            f' {smooth}Filter({self.smoothOpNb})=1,\n'+\
            f' {smooth}Dims({self.smoothOpNb})=\'{self.smooth2DDims}\',\n'+\
            f' {smooth}Algorithm({self.smoothOpNb})=\'{alg}\',\n'+\
            f' {smooth}NbRand({self.smoothOpNb})={num_inputs},\n'+\
            f' {smooth}JacobiMaxIters({self.smoothOpNb}) = {self.jacobi_max_iters},\n'+\
            f' {smooth}SOROmega({self.smoothOpNb}) = {sor},\n'+\
            ' &'
        fname = write_dir+f'/data.smooth'
        with open(fname,'w') as f:
            f.write(file_contents)

    def submit_matern(self,
                      fld,
                      Nx,xi,
                      write_dir,run_dir,
                      run_suff):
        """Use the MITgcm to smooth a small number of fields, wait for result

        Inputs
        ------
        fld : xarray DataArray
            with possibly multiple records,
        Nx, xi : int
            specifies the prior
        write_dir, run_dir : str
            telling the simulation where to do stuff
        """

        nrecs = fld.shape[0] if fld.shape!=self.ctrl.mask.shape else 1

        # --- Write and submit
        fld = fld.reindex_like(self.mymodel).values
        fname = f'{write_dir}/smooth2DInput{self.smoothOpNb:03}'
        wrmds(fname,arr=fld,dataprec=self.dataprec,nrecords=nrecs)
        matern.write_matern(write_dir,
                 smoothOpNb=self.smoothOpNb,Nx=Nx,mymask=self.ctrl.mask,
                 xdalike=self.mymodel,xi=xi)
        sim = rp.Simulation(name=f'{Nx:02}dx_{xi:02}xi_{run_suff}',
                            run_dir=run_dir,
                            obs_dir=write_dir,**self.dsim)

        sim.link_to_run_dir()
        sim.write_slurm_script()
        jid = sim.submit_slurm(**self.slurm)

        return jid,sim


    def submit_next_stage(self,next_stage, mysim, jid_depends=None):
        """Write a bash script and submit, which will execute next stage
        of experiment

        Parameters
        ----------
        next_stage : str
            a string with the next stage to execute, see init or start
        jid_depends : list of ints or int
            with slurm job id's to wait on before submitting the next stage
        mysim : rosypig.Simulation
            use the last simulation to create another one
        """

        bashname = self.write_bash_script(stage=next_stage,mysim=mysim)

        if jid_depends is not None:
            jid_depends = [jid_depends] if isinstance(jid_depends,int) else jid_depends
            jid_depends = str(jid_depends).replace(', ',':').replace('[','').replace(']','')
            # catch case of 0 length list
            if jid_depends == '':
                slurmstring=f'sbatch {bashname}'
            else:
                slurmstring=f'sbatch --dependency=afterany:{jid_depends} {bashname}'
        else:
            slurmstring=f'sbatch {bashname}'

        pout = mysim.launch_the_job(slurmstring)

    def write_bash_script(self,stage,mysim):
        """Write a bash script for the next experiment stage
        """
        file_contents = '#!/bin/bash\n\n' +\
            f'#SBATCH -J {stage}\n' +\
            f'#SBATCH -o {stage}.%j.out\n' +\
            f'#SBATCH -e {stage}.%j.err\n' +\
            '#SBATCH -N 1\n' +\
            f'#SBATCH -n {mysim.procs_per_node}\n' +\
            f'#SBATCH -p {mysim.queue_name}\n' +\
            f'#SBATCH -t {mysim.time}\n'

        file_contents += f'\n\neval "$(conda shell.bash hook)"\n'+\
                f'conda activate {self.conda_env}\n\n'+\
                f'python3 -c '+\
                '"from pych.pigmachine import OIDriver;'+\
                f'oid = OIDriver(\'{self.experiment}\',\'{stage}\')"\n'

        fname = self.dirs['main_run']+f'/submit_{self.experiment}.sh'
        with open(fname,'w') as f:
            f.write(file_contents)
        return fname



# ----
# Helpers
# -----
    def _get_dirs(self,stage,Nx,xi):
        """return read_dir, write_dir, and run_dir for a specific stage"""

        if stage == 'range_approx_one':
            read_str  = None
            write_str = 'maternC' if 'uvel' not in self.experiment else 'maternW'

        elif stage == 'range_approx_two':
            read_str  = 'maternC' if 'uvel' not in self.experiment else 'maternW'
            write_str = self.experiment + '/range2'

        elif stage == 'basis_projection_one':
            read_str  = self.experiment + '/range2'
            write_str = self.experiment + '/project1'

        elif stage == 'basis_projection_two':
            read_str  = self.experiment + '/project1'
            write_str = self.experiment + '/project2'

        elif stage == 'do_the_evd':
            read_str  = self.experiment + '/project2'
            write_str = None

        elif stage == 'prior_to_misfit':
            read_str = self.experiment + '/project2'
            write_str = self.expWithInitGuess + '/p2m'

        elif stage == 'solve_for_map':
            read_str = self.expWithInitGuess + '/p2m'
            write_str = self.expWithInitGuess + '/map'

        elif stage == 'save_the_map':
            read_str = self.expWithInitGuess + '/map'
            write_str = None

        elif stage == 'get_vm':
            read_str = None
            write_str = self.experiment + '/gvm'

        elif stage == 'calc_rayleigh':
            read_str = self.experiment + '/gvm'
            write_str = None

        else:
            raise NameError(f'Unexpected stage for directories: {stage}')

        numsuff = f'.{Nx:02}dx.{xi:02}xi'
        if read_str is not None:
            read_str += '/run' + numsuff
            read_dir = self.dirs["main_run"] + '/' + read_str
        else:
            read_dir = None

        if write_str is not None:
            write_dir = write_str + '/input' + numsuff
            run_dir   = write_str + '/run'   + numsuff
            write_dir = _dir(self.dirs['main_run']+'/'+write_dir)
            run_dir   = self.dirs['main_run']+'/'+run_dir
        else:
            write_dir=None
            run_dir=None

        return read_dir, write_dir, run_dir

def _dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return dirname


def _add_map_fields(ds,sigma,doRegularizeDebug):
    """Helper routine to define some container fields

    """

    # parameter uncertainty and regularization
    ds['sigma'] = xr.DataArray(sigma,coords={'sigma':sigma},dims=('sigma',))
    ds['betaSq'] = ds.sigma**-2

    # Recompute eigenvalues
    ds['Dorig'] = ds['D'].copy()
    ds['D'] = ds.sigma**2 * ds.Dorig
    ds['Dinv'] = ds.D / (1+ ds.D)

    bfn = ds['sigma']*ds['xi']*ds['Nx']
    ds['m_map'] = xr.zeros_like(bfn*ds['ctrl_ind'])
    ds['reg_norm'] = xr.zeros_like(bfn)
    ds['misfits'] = xr.zeros_like(bfn*ds['obs_ind'])
    ds['misfits_normalized'] = xr.zeros_like(bfn*ds['obs_ind'])
    ds['misfit_norm'] = xr.zeros_like(bfn)
    ds['misfits_model_space'] = xr.zeros_like(bfn*ds['ctrl_ind'])

    if doRegularizeDebug:
        # extra on the regularization term
        ds['reg_delta'] = xr.zeros_like(bfn)
        ds['reg_laplacian'] = xr.zeros_like(bfn)
        ds['reg_S_delta'] = xr.zeros_like(bfn)
        ds['reg_S_laplacian'] = xr.zeros_like(bfn)

    # --- some descriptive attributes
    for fld in ds.keys():
        ds[fld].attrs = get_nice_attrs(fld)
    for fld in ds.coords.keys():
        ds[fld].attrs = get_nice_attrs(fld)

    return ds


def _unpack_field(fld,packer=None):
    if packer is not None:
        if len(fld.shape)>1 or len(fld)!=packer.n_wet:
            fld = packer.pack(fld)
    else:
        if len(fld.shape)>1:
            fld = fld.flatten() if isinstance(fld,np.ndarray) else fld.values.flatten()
    return fld
