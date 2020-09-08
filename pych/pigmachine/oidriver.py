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
    dataprec = 'float64'
    NxList  = [5, 10, 15, 20, 30, 40]
    FxyList = [0.5,   1,   2]#,   5]
    sorList = [1.8, 1.6, 1.3]#, 1.2]
    n_beta = 9
    beta = 10**np.linspace(-5,-3,n_beta)
    smoothOpNb = 1
    smooth2DDims = 'yzw'
    jacobi_max_iters = 20000
    conda_env = 'py38_tim'
    def __init__(self, experiment,stage=None):
        """Should this initialize? Or do we want another method to do it?
        
        Parameters
        ----------
        experiment : str
            an identifying name for the experiment
        stage : str
            the stage to start or pickup at when this is called

            'range_approx_one' : first stage of range approximation
            'range_approx_two'
            'basis_projection_one'
            'basis_projection_two'
            'do_the_evd'
            'solve_for_map'
            'save_the_evd'
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
                if key == 'beta' and isinstance(val,list):
                    val = np.array(val)
                self.__dict__[key] = val

        self.sordict = dict(zip(self.FxyList,self.sorList))

    def _send_to_stage(self,stage):
        possible_stages = ['range_approx_one','range_approx_two', 
                           'basis_projection_one','basis_projection_two', 
                           'do_the_evd','prior_to_misfit',
                           'solve_for_map','save_the_map']

        if stage in possible_stages:
            self.pickup()
            eval(f'self.{stage}()')
        else:
            if stage != 'start_experiment' and stage is not None:
                raise NameError(f'Incorrectly specified stage: {stage}.\n'+\
                    'Available possibilities are: '+str(possible_stages))


# ---------------------------------------------------------------------
# Range Approximation
# ---------------------------------------------------------------------
    def range_approx_one(self):
        jid_list = []
        for nx in self.NxList:
            for fxy in self.FxyList:

                _, write_dir, run_dir = self._get_dirs('range_approx_one',nx,fxy)

                self.smooth_writer(write_dir, Fxy=fxy, smooth_apply=False)
                matern.write_matern(write_dir,
                                    smoothOpNb=self.smoothOpNb,
                                    Nx=nx,mymask=self.ctrl.mask,
                                    xdalike=self.mymodel,Fxy=fxy) 

                sim = rp.Simulation(name=f'{nx:02}dx_{fxy:02}fxy_oi_ra1',
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

        jid_list = []
        dslistNx = []
        for nx in self.NxList:
            dslistFxy = []
            for fxy in self.FxyList:

                # --- Prepare reading and writing
                read_dir, write_dir, run_dir = self._get_dirs('range_approx_two',nx,fxy)

                # --- Read in samples and filternorm
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=True)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();
                ds['filternorm'].load();
                evds = rp.RandomDecomposition(n_ctrl=self.ctrl.n_wet,
                                              n_obs=self.obs.n_wet,
                                              n_small=self.n_small,
                                              n_over=self.n_over)

                # Add Nx and Fxy dimensions
                evds['filternorm'] = rp.to_xda(self.ctrl.pack(ds['filternorm']),evds,
                                               expand_dims={'Nx':nx,'Fxy':fxy})
                dslistFxy.append(evds)

                # --- Now apply obserr & filternorm weighted interpolation operator
                smooth2DInput = []
                filternorm = self.ctrl.pack(ds['filternorm'].values)
                F_norm = self.F*filternorm
                FtWF = (F_norm.T * self.obs_var_inv) @ F_norm
                for s in ds.sample.values:
                    g_out = FtWF @ self.ctrl.pack(ds['ginv'].sel(sample=s))
                    g_out = self.ctrl.unpack(g_out)
                    smooth2DInput.append(g_out)

                # --- Write matern operator and submit job
                smooth2DInput = xr.concat(smooth2DInput,dim='sample')
                self.smooth_writer(write_dir, Fxy=fxy)
                jid,sim =self.submit_matern(fld=smooth2DInput,
                                   Nx=nx,Fxy=fxy,
                                   write_dir=write_dir,
                                   run_dir=run_dir,
                                   run_suff='ra2')
                jid_list.append(jid)

            # --- Keep appending the dataset with filternorm for safe keeping
            dslistNx.append(xr.concat(dslistFxy,dim='Fxy'))

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

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_filternormInterp.nc')
        jid_list=[]
        dslistNx = []
        for nx in self.NxList:
            dslistFxy = []
            for fxy in self.FxyList:
    
                # --- Prepare read and write
                read_dir, write_dir, run_dir = self._get_dirs('basis_projection_one',nx,fxy)

                # --- Read samples from last stage, form orthonormal basis
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)

                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();
        
                # Form Y, range approximator
                Y = []
                for s in ds.sample.values:
                    Y.append(self.ctrl.pack(ds['ginv'].sel(sample=s)))
                Y = np.array(Y).T
        
                # do some linalg
                Q,_ = np.linalg.qr(Y)
        
                # prep for saving
                tmpds = xr.Dataset()
                tmpds['Y'] = rp.to_xda(Y,evds,expand_dims={'Nx':nx,'Fxy':fxy})
                tmpds['Q'] = rp.to_xda(Q,evds,expand_dims={'Nx':nx,'Fxy':fxy})
                dslistFxy.append(tmpds)
        
                # --- Write out Q and submit job
                smooth2DInput = []
                for i in range(evds.n_rand):
                    q = self.ctrl.unpack(Q[:,i],fill_value=0.)
                    smooth2DInput.append(q)

                smooth2DInput = xr.concat(smooth2DInput,dim='sample')
                self.smooth_writer(write_dir, Fxy=fxy)
                jid,sim =self.submit_matern(fld=smooth2DInput,
                                   Nx=nx,Fxy=fxy,
                                   write_dir=write_dir,
                                   run_dir=run_dir,
                                   run_suff='proj1')
                jid_list.append(jid)
        
            # get those datasets
            dslistNx.append(xr.concat(dslistFxy,dim='Fxy'))
        newds = xr.concat(dslistNx,dim='Nx')
        atts = evds.attrs.copy()
        evds = xr.merge([evds,newds])
        evds.attrs = atts
        evds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_proj1.nc')

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='basis_projection_two',
                               jid_depends=jid_list,mysim=sim)

    def basis_projection_two(self):
        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_proj1.nc')
        evds['filternorm'].load();
        evds['F'].load();

        jid_list = []
        for nx in self.NxList:
            for fxy in self.FxyList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('basis_projection_two',nx,fxy)
            
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
                filternorm = evds['filternorm'].sel(Nx=nx,Fxy=fxy).values
                F_norm = evds['F'].values*filternorm
                FtWF = (F_norm.T * self.obs_var_inv) @ F_norm
                for s in ds.sample.values:
                    # write this out rather than form it every time
                    g_out = FtWF @ self.ctrl.pack(ds['ginv'].sel(sample=s))
                    g_out = self.ctrl.unpack(g_out)
                    smooth2DInput.append(g_out)

                # --- Write out and submit next application
                smooth2DInput = xr.concat(smooth2DInput,dim='sample')
                self.smooth_writer(write_dir, Fxy=fxy)
                jid,sim =self.submit_matern(fld=smooth2DInput,
                                   Nx=nx,Fxy=fxy,
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

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_proj1.nc')
        evds['Q'].load();
        jid_list = []
        dslistNx = []
        for nx in self.NxList:
            dslistFxy = []
            for fxy in self.FxyList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('do_the_evd',nx,fxy)

                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                # Get Hm Q
                HmQ = []
                for s in ds.sample.values:
                    HmQ.append(self.ctrl.pack(ds['ginv'].sel(sample=s)))
                HmQ = np.array(HmQ).T

                # Apply Q^T
                Q = evds['Q'].sel(Nx=nx,Fxy=fxy).values
                B = Q.T @ HmQ

                # Do the EVD
                D,Uhat = linalg.eigh(B)

                # eigh returns in ascending order, reverse
                D = D[::-1]
                Uhat = Uhat[:,::-1]

                U = Q @ Uhat

                tmpds = xr.Dataset()
                tmpds['U'] = rp.to_xda(U,evds,expand_dims={'Nx':nx,'Fxy':fxy})
                tmpds['D'] = rp.to_xda(D,evds,expand_dims={'Nx':nx,'Fxy':fxy})
                dslistFxy.append(tmpds)

            # --- Continue saving eigenvalues
            dslistNx.append(xr.concat(dslistFxy,dim='Fxy'))

        newds = xr.concat(dslistNx,dim='Nx')
        atts = evds.attrs.copy()
        evds = xr.merge([newds,evds])
        evds.attrs = atts
        evds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_evd.nc')

        # --- Pass on to next stage
        sim = rp.Simulation(name='evd',
                **self.dsim)
        self.submit_next_stage(next_stage='prior_to_misfit',
                               jid_depends=jid_list,mysim=sim)

    def prior_to_misfit(self):
        """Optional to re-run this from the last stage with another initial guess"""

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_evd.nc')
        evds['filternorm'].load();

        # --- Compute initial misfit: F m_0 - d, and weighted versions
        initial_misfit = self.obs_mean - evds['F'].values @ self.m0

        # --- Map back to model grid and apply posterior via EVD
        misfit2model = evds['F'].T.values @ (self.obs_var_inv * initial_misfit)

        jid_list = []
        for nx in self.NxList:
            for fxy in self.FxyList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('prior_to_misfit',nx,fxy)

                # --- Possibly use lower dimensional subspace and get arrays
                filternorm = self.ctrl.unpack(evds['filternorm'].sel(Nx=nx,Fxy=fxy))

                # --- Apply prior
                smooth2DInput = self.ctrl.unpack(misfit2model)
                self.smooth_writer(write_dir, Fxy=fxy, num_inputs=1)
                jid,sim = self.submit_matern( \
                                    fld=smooth2DInput,
                                    Nx=nx,Fxy=fxy,
                                    write_dir=write_dir,
                                    run_dir=run_dir,
                                    run_suff='p2m')
                jid_list.append(jid)

        self.submit_next_stage(next_stage='solve_for_map',
                               jid_depends=jid_list,mysim=sim)


    def solve_for_map(self):
        """Optional to re-run this from the last stage with another initial guess"""

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_evd.nc')
        evds['filternorm'].load();

        evds = _add_map_fields(evds,self.beta)

        jid_list = []
        for nx in self.NxList:
            for fxy in self.FxyList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('solve_for_map',nx,fxy)

                # --- Possibly use lower dimensional subspace and get arrays
                U = evds['U'].sel(Nx=nx,Fxy=fxy).values[:,:self.n_small]

                filternorm = self.ctrl.unpack(evds['filternorm'].sel(Nx=nx,Fxy=fxy))


                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                prior_misfit2model = filternorm*ds['ginv'].isel(sample=0)
                prior_misfit2model = self.ctrl.pack(prior_misfit2model)

                smooth2DInput = []
                for b in self.beta:
        
                    # --- Possibly account for lower dimensional subspace
                    Dinv = evds['Dinv'].sel(Nx=nx,Fxy=fxy,beta=b).values[:self.n_small]

                    # --- Apply (I-UDU^T)
                    udu = U @ ( Dinv * ( U.T @ (b*prior_misfit2model)))
                    iudu = b*prior_misfit2model - udu

                    iudu = self.ctrl.unpack(iudu)
                    smooth2DInput.append(iudu)

                # --- Apply prior half
                smooth2DInput = xr.concat(smooth2DInput,dim='beta')
                self.smooth_writer(write_dir, Fxy=fxy, num_inputs=len(self.beta))
                jid,sim = self.submit_matern(fld=smooth2DInput,
                                         Nx=nx,Fxy=fxy,
                                         write_dir=write_dir,
                                         run_dir=run_dir,
                                         run_suff='map')
                jid_list.append(jid)

        evds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_map.nc')
        self.submit_next_stage(next_stage='save_the_map',
                               jid_depends=jid_list,mysim=sim)

    def save_the_map(self):
        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_map.nc')
        evds['filternorm'].load();

        for nx in self.NxList:
            for fxy in self.FxyList:

                # --- Prepare directories
                read_dir, write_dir, run_dir = self._get_dirs('save_the_map',nx,fxy)

                # --- Get arrays
                filternorm = self.ctrl.unpack(evds['filternorm'].sel(Nx=nx,Fxy=fxy))
                filterstd = xr.where(filternorm!=0,filternorm**-1,0.)

                C,K = matern.get_matern(Nx=nx,mymask=self.ctrl.mask,Fxy=fxy)

                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(len(self.beta)),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();
                for s,b in enumerate(self.beta):

                    m_update = b*filternorm*ds['ginv'].sel(sample=s)
                    mmap = self.m0 + self.ctrl.pack(m_update)

                    # --- Compute m_map - m0, and weighted version
                    dm = filterstd*m_update
                    dm_normalized = (b**-1) * rp.apply_matern_2d( \
                                                fld_in=dm,
                                                mask3D=self.cds.maskC,
                                                mask2D=self.ctrl.mask,
                                                ds=self.cds,
                                                delta=C['delta'],
                                                Kux=None,Kvy=K['vy'],Kwz=K['wz'])

                    dm_normalized = rp.to_xda(self.ctrl.pack(dm_normalized),evds)
                    with xr.set_options(keep_attrs=True):
                        evds['m_map'].loc[{'beta':b,'Fxy':fxy,'Nx':nx}] = \
                                rp.to_xda(mmap,evds)
                        evds['reg_norm'].loc[{'beta':b,'Fxy':fxy,'Nx':nx}] = \
                            np.linalg.norm(dm_normalized,ord=2)

                    # --- Compute misfits, and weighted version
                    misfits = evds['F'].values @ mmap - self.obs_mean
                    misfits_model_space = rp.to_xda( \
                            mmap - evds['F'].T.values @ self.obs_mean,evds)
                    misfits_model_space = misfits_model_space.where( \
                                            evds['F'].T.values @ self.obs_mean !=0)

                    with xr.set_options(keep_attrs=True):
                        evds['misfits'].loc[{'beta':b,'Fxy':fxy,'Nx':nx}] = \
                                rp.to_xda(misfits,evds)
                        evds['misfits_normalized'].loc[{'beta':b,'Fxy':fxy,'Nx':nx}] = \
                            rp.to_xda(self.obs_std_inv * misfits,evds)
                        evds['misfit_norm'].loc[{'beta':b,'Fxy':fxy,'Nx':nx}] = \
                            np.linalg.norm(self.obs_std_inv * misfits,ord=2)
                        evds['misfits_model_space'].loc[{'beta':b,'Fxy':fxy,'Nx':nx}] = \
                            misfits_model_space

        evds.to_netcdf(self.dirs['netcdf']+f'/{self.experiment}_map.nc')

# ---------------------------------------------------------------------
# Stuff for organizing each stage of the run
# ---------------------------------------------------------------------
    def smooth_writer(self,write_dir,Fxy,smooth_apply=True,num_inputs=1000):
        """write the data.smooth file
        """
        ndims = len(self.mymodel.dims)
        smooth = f'smooth{ndims}D'
        alg = 'matern'
        alg = alg if not smooth_apply else alg+'apply'
        sor = self.sordict[Fxy]
        file_contents = ' &SMOOTH_NML\n'+\
            f' {smooth}Filter({self.smoothOpNb})=1,\n'+\
            f' {smooth}Dims({self.smoothOpNb})=\'{self.smooth2DDims}\',\n'+\
            f' {smooth}Algorithm({self.smoothOpNb})=\'{alg}\',\n'+\
            f' {smooth}NbRand({self.smoothOpNb})={num_inputs},\n'+\
            f' {smooth}JacobiMaxIters(1) = {self.jacobi_max_iters},\n'+\
            f' {smooth}SOROmega(1) = {sor},\n'+\
            ' &'
        fname = write_dir+f'/data.smooth'
        with open(fname,'w') as f:
            f.write(file_contents)

    def submit_matern(self,
                      fld,
                      Nx,Fxy,
                      write_dir,run_dir,
                      run_suff):
        """Use the MITgcm to smooth a small number of fields, wait for result

        Inputs
        ------
        fld : xarray DataArray
            with possibly multiple records,
        Nx, Fxy : int
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
                 xdalike=self.mymodel,Fxy=Fxy)
        sim = rp.Simulation(name=f'{Nx:02}dx_{Fxy:02}fxy_{run_suff}',
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
            slurmstring=f'sbatch --dependency=afterany:{jid_depends} {bashname}'
        else:
            slurmstring=f'sbatch {bashname}'

        pout = mysim.launch_the_job(slurmstring)

    def write_bash_script(self,stage,mysim):
        """Write a bash script for the next experiment stage
        """
        file_contents = '#!/bin/bash\n\n' +\
            '#SBATCH -J oidriver\n' +\
            '#SBATCH -o oidriver.%j.out\n' +\
            '#SBATCH -e oidriver.%j.err\n' +\
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
    def _get_dirs(self,stage,nx,fxy):
        """return read_dir, write_dir, and run_dir for a specific stage"""

        if stage == 'range_approx_one':
            read_str = None
            write_str = 'matern'

        elif stage == 'range_approx_two':
            read_str = 'matern'
            write_str = 'range2'

        elif stage == 'basis_projection_one':
            read_str = 'range2'
            write_str = 'project1'

        elif stage == 'basis_projection_two':
            read_str = 'project1'
            write_str = 'project2'

        elif stage == 'do_the_evd':
            read_str = 'project2'
            write_str = 'evd'

        elif stage == 'prior_to_misfit':
            read_str = 'project2'
            write_str = 'p2m'

        elif stage == 'solve_for_map':
            read_str = 'p2m'
            write_str = 'map'

        elif stage == 'save_the_map':
            read_str = 'map'
            write_str = None

        else:
            raise NameError(f'Unexpected stage for directories: {stage}')

        if read_str is not None:
            read_suff = f'/run.{self.experiment}' if stage !='range_approx_two' else '/run'
            read_dir = self.dirs["main_run"]+read_suff+\
                f'.{read_str}.{nx:02}dx.{fxy:02}fxy'
        else:
            read_dir = None

        if write_str is not None:
            write_suff = self.experiment +'.' if stage != 'range_approx_one' else ''
            write_suff += f'{write_str}.{nx:02}dx.{fxy:02}fxy'
            write_dir = _dir(self.dirs['main_run']+'/'+write_suff)
            run_dir   = self.dirs['main_run']+'/run.'+write_suff
        else:
            write_dir=None
            run_dir=None

        return read_dir, write_dir, run_dir

def _dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return dirname


def _add_map_fields(ds,beta):
    """Helper routine to define some container fields

    """

    ds['beta'] = xr.DataArray(beta,coords={'beta':beta},dims=('beta',))

    # Recompute eigenvalues
    ds['Dorig'] = ds['D'].copy()
    ds['D'] = ds.beta**2 * ds.Dorig 
    ds['Dinv'] = ds.D / (1+ ds.D)

    bfn = ds['beta']*ds['Fxy']*ds['Nx']
    ds['m_map'] = xr.zeros_like(bfn*ds['ctrl_ind'])
    ds['reg_norm'] = xr.zeros_like(bfn)
    ds['misfits'] = xr.zeros_like(bfn*ds['obs_ind'])
    ds['misfits_normalized'] = xr.zeros_like(bfn*ds['obs_ind'])
    ds['misfit_norm'] = xr.zeros_like(bfn)
    ds['misfits_model_space'] = xr.zeros_like(bfn*ds['ctrl_ind'])

    # --- some descriptive attributes
    ds['beta'].attrs = {'label':r'$\beta$','description':'regularization parameter'}
    ds['m_map'].attrs = {'label':r'$\mathbf{m}_{MAP}$',
            'description':r'Maximum a Posteriori solution for control parameter $\mathbf{m}$'}
    ds['reg_norm'].attrs = {'label':r'$||\mathbf{m}_{MAP} - \mathbf{m}_0||_{\Gamma_{prior}^{-1}}$',
            'label2':r'$||\Gamma_{prior}^{-1/2}(\mathbf{m}_{MAP} - \mathbf{m}_0)||_2$',
            'description':'Normed difference between initial and MAP solution, weighted by prior uncertainty'}
    ds['misfits'].attrs = {'label':r'$F\mathbf{m}_{MAP} - \mathbf{d}$',
            'description':'Difference between MAP solution and observations'}
    ds['misfits_normalized'].attrs = {'label':r'$\dfrac{F\mathbf{m}_{MAP} - \mathbf{d}}{\sigma_{obs}}$',
            'description':'Difference between MAP solution and observations, normalized by observation uncertainty'}
    ds['misfit_norm'].attrs = {'label':r'$||F\mathbf{m}_{MAP} - \mathbf{d}||_{\Gamma_{obs}^{-1}}$',
            'label2':r'$||\Gamma_{obs}^{-1/2}(F\mathbf{m}_{MAP} - \mathbf{d})||_2$',
            'description':'Normed difference between MAP solution and observations, weighted by observational uncertainty'}
    ds['misfits_model_space'].attrs = {'label':r'$\mathbf{m}_{MAP} - F^T\mathbf{d}$',
            'description':'Difference between MAP solution and observations, in model domain'}

    return ds


def _unpack_field(fld,packer=None):
    if packer is not None:
        if len(fld.shape)>1 or len(fld)!=packer.n_wet:
            fld = packer.pack(fld)
    else:
        if len(fld.shape)>1:
            fld = fld.flatten() if isinstance(fld,np.ndarray) else fld.values.flatten()
    return fld
