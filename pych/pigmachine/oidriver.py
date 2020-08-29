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
    NxList = [10,15,20,30]
    FxyList = [0.5,1,2,5]
    smoothOpNb = 1
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
            'save_the_evd'
        """
        self.experiment = experiment
        self._send_to_stage(stage)


    def start(self,dirs,dsim,mymodel,obs_mask,obs_std,
              startat='range_approx_one',**kwargs):
        """Start the experiment by writing everything we need to file,
        to be read in later by "pickup"

        Parameters
        ----------
        dirs : dict
            containing the key directories for the experiment
            'main_run' : the directory where MITgcm files are written
                and simulations are started
            'namelist' : with namelist files for the first range approximation stage
                i.e. where smoothing package pushes through random samples
            'namelist_apply' : with namelist files for all other stages, where
                the smoothing operator is applied to an input vector
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
        obs_mask : xarray DataArray
            mask field denoting where observations are taken, on an "observation grid"
            Note: dimensions are assumed to be ordered like
            (vertical, meridional, zonal)
        obs_std : xarray DataArray
            containing observational uncertainties (of course, at the mask points!)
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
        mymodel.to_netcdf(dirs['nctmp']+f'/{self.experiment}_ctrl.nc')
        myobs = xr.Dataset({'obs_mask':obs_mask,'obs_std':obs_std})
        myobs.to_netcdf(dirs['nctmp']+f'/{self.experiment}_obs.nc')

        # --- "pickup" experiment at startat
        self.pickup()
        startsim = rp.Simulation('startmeup',namelist_dir=dirs['namelist'],**dsim)
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

        mymodel = xr.open_dataarray(dirs['nctmp']+f'/{self.experiment}_ctrl.nc')
        myobs = xr.open_dataset(dirs['nctmp']+f'/{self.experiment}_obs.nc')

        modeldims=list(mymodel.dims)
        obsdims = list(myobs['obs_mask'].dims)

        # --- Carry these things around
        self.dirs = dirs
        self.dsim = dsim
        self.mymodel = mymodel
        self.obs_std = myobs['obs_std']
        self.ctrl = rp.ControlField('ctrl',mymodel.sortby(modeldims))
        self.obs = rp.ControlField('obs',
                                   myobs['obs_mask'].sortby(obsdims).astype('bool'))

        # --- Get the interpolation operator
        mdimssort = [mymodel[dim].sortby(dim) for dim in modeldims]
        odimssort = [myobs[dim].sortby(dim) for dim in obsdims]
        self.F = pm.interp_operator_2d(mdimssort,odimssort,
                                       pack_index_in=self.ctrl.wet_ind,
                                       pack_index_out=self.obs.wet_ind)

        # --- If kwargs exist, use to rewrite default attributes
        if kwargs is not None:
            for key,val in kwargs.items():
                self.__dict__[key] = val

    def _send_to_stage(self,stage):
        possible_stages = ['range_approx_one','range_approx_two', 
                           'basis_projection_one','basis_projection_two', 
                           'do_the_evd','save_the_evd']

        if stage in possible_stages:
            self.pickup()
            eval(f'self.{stage}()')
        else:
            if stage != 'start_experiment' and stage is not None:
                raise NameError(f'Incorrectly specified stage: {stage}.\n'+\
                    'Available possibilities are: '+str(possible_stages))


# ---------------------------------------------------------------------
# The actual routines!
# ---------------------------------------------------------------------

# --- Range Approximation
    def range_approx_one(self):
        jid_list = []
        for nx in self.NxList:
            for fxy in self.FxyList:

                stage_suff = self.experiment+f'.range1.{nx:02}dx.{fxy:02}fxy'
                write_dir = _dir(self.dirs["main_run"]+'/'+stage_suff)

                matern.write_matern(write_dir,
                                    smoothOpNb=self.smoothOpNb,
                                    Nx=nx,mymask=self.ctrl.mask,
                                    xdalike=self.mymodel,Fxy=fxy) 

                sim = rp.Simulation(name=f'{nx:02}dx_{fxy:02}fxy_oi_ra1',
                                 namelist_dir=self.dirs['namelist'],
                                 run_dir=self.dirs["main_run"]+f'/run.{stage_suff}',
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
                read_suff = self.experiment+f'.range1.{nx:02}dx.{fxy:02}fxy'
                read_dir = self.dirs["main_run"]+f'/run.{read_suff}'

                write_suff = self.experiment+f'.range2.{nx:02}dx.{fxy:02}fxy'
                write_dir = _dir(self.dirs["main_run"]+'/'+write_suff)
                run_dir = self.dirs["main_run"]+'run.'+write_suff

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
                        n_small=self.n_small,n_over=self.n_over)

                # Add Nx and Fxy dimensions
                evds['filternorm'] = rp.to_xda(self.ctrl.pack(ds['filternorm']),evds,
                                       expand_dims={'Nx':nx,'Fxy':fxy})
                dslistFxy.append(evds)

                # --- Now apply obserr & filternorm weighted interpolation operator
                smooth2DInput = []
                for s in ds.sample.values:
                    g_out = pm.apply_ppmh(self.F,fld=ds['ginv'].sel(sample=s),
                                  filternorm=ds['filternorm'],
                                  obs_std=self.obs_std,
                                  input_packer=self.ctrl,output_packer=self.obs)
                    # Prepare for I/O
                    g_out = g_out.reindex_like(self.mymodel)
                    smooth2DInput.append(g_out.values)

                # --- Write inputs and submit job
                smooth2DInput = np.stack(smooth2DInput,axis=0)
                fname = f'{write_dir}/smooth2DInput{self.smoothOpNb:03d}'
                wrmds(fname,arr=smooth2DInput,dataprec=self.dataprec,
                      nrecords=len(ds.sample))

                # Write matern operator and submit job
                matern.write_matern(write_dir,smoothOpNb=self.smoothOpNb,
                        Nx=nx,mymask=self.ctrl.mask,xdalike=self.mymodel)

                sim = rp.Simulation(name=f'{nx:02}dx_{fxy:02}fxy_ra2_oi',
                                    namelist_dir=self.dirs['namelist_apply'],
                                    run_dir=run_dir,
                                    obs_dir=write_dir,
                                    **self.dsim)
                # launch job
                sim.link_to_run_dir()

                sim.write_slurm_script()
                jid =sim.submit_slurm(**self.slurm)
                jid_list.append(jid)

            # --- Keep appending the dataset with filternorm for safe keeping
            dslistNx.append(xr.concat(dslistFxy,dim='Fxy'))

        newds = xr.concat(dslistNx,dim='Nx')
        newds['F'] = rp.to_xda(self.F,newds)
        newds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_filternormInterp.nc')

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='basis_projection_one',
                               jid_depends=jid_list,mysim=sim)


# --- Form the orthonormal basis Q and use it to project to low dim subspace
    def basis_projection_one(self):

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_filternormInterp.nc')
        jid_list=[]
        dslistNx = []
        for nx in self.NxList:
            dslistFxy = []
            for fxy in self.FxyList:
    
                # --- Prepare read and write
                read_suff =  self.experiment+f'.range1.{nx:02}dx.{fxy:02}fxy'
                read_dir = self.dirs['main_run']+'/run.'+read_suff
                write_suff = self.experiment+f'.project1.{nx:02}dx.{fxy:02}fxy'
                write_dir = _dir(self.dirs['main_run']+'/'+write_suff)
                run_dir = self.dirs['main_run']+'/run.'+write_suff

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
                    q = q.reindex_like(self.mymodel).values
                    smooth2DInput.append(q)

                smooth2DInput = np.stack(smooth2DInput,axis=0)
                wrmds(f'{write_dir}/smooth2DInput{smoothOpNb:03d}',
                      arr=smooth2DInput,dataprec=self.dataprec)
            
                # set up the run
                matern.write_matern(write_dir,smoothOpNb=self.smoothOpNb,
                                    Nx=nx,mymask=self.ctrl.mask,xdalike=self.mymodel)

                sim = rp.Simulation(name=f'{nx:02}dx_{fxy:02}fxy_proj1_oi',
                                    namelist_dir=self.dirs['namelist_apply'],
                                    run_dir=run_dir,
                                    obs_dir=write_dir,
                                    **self.dsim)
        
                # launch job
                sim.link_to_run_dir()
                sim.write_slurm_script()
                jid = sim.submit_slurm(**self.slurm)
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
        jid_list = []
        for nx in self.NxList:
            for fxy in self.FxyList:

                # --- Prepare directories
                read_suff = self.experiment+f'.project1.{nx:02}dx.{fxy:02}fxy'
                read_dir = self.dirs['main_run']+'/'+read_suff
                write_suff = self.experiment+f'.project2.{nx:02}dx.{fxy:02}fxy'
                write_dir = _dir(self.dirs['main_run']+'/'+write_suff)
                run_dir = self.dirs['main_run']+'/run.'+write_suff


            
                # --- Read output from last stage, apply obserr,filternorm weight op
                ds = matern.get_matern_dataset(read_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)
                ds = ds.sortby(self.mymodel.dims)
                ds['ginv'].load();

                # Apply operator
                smooth2DInput = []
                filternorm = self.ctrl.unpack(evds['filternorm'].sel(Nx=nx,Fxy=fxy))
                for s in ds.sample.values:
                    g_out = pm.apply_ppmh(self.F,fld=ds['ginv'].sel(sample=s),
                                          filternorm=filternorm,
                                          obs_std=self.obs_std,
                                          input_packer=self.ctrl,
                                          output_packer=self.obs)

                    smooth2DInput.append(g_out.reindex_like(self.mymodel).values)
                smooth2DInput = np.stack(smooth2DInput,axis=0)

                # --- Write out and submit next application
                fname = f'{write_dir}/smooth2DInput{smoothOpNb:03d}'
                wrmds(fname,arr=smooth2DInput,dataprec=self.dataprec)

                matern.write_matern(write_dir,smoothOpNb=self.smoothOpNb,
                                    Nx=nx,mymask=self.ctrl.mask,xdalike=self.mymodel)

                sim = rp.Simulation(name=f'{nx:02d}dx_{fxy:02}fxy_proj2_oi',
                                    namelist_dir=self.dirs['namelist_apply'],
                                    run_dir=run_dir,
                                    obs_dir=write_dir,
                                    **self.dsim)

                # launch job
                sim.link_to_run_dir()
                sim.write_slurm_script()
                jid = sim.submit_slurm(**self.slurm)
                jid_list.append(jid)

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='do_the_evd',
                               jid_depends=jid_list,mysim=sim)

    def do_the_evd(self):

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_proj1.nc')
        evds['Q'].load();
        jid_list = []
        dslistNx = []
        for nx in self.NxList:
            dslistFxy = []
            for fxy in self.FxyList:

                # --- Prepare directories
                read_suff = self.experiment+f'.project2.{nx:02}dx.{fxy:02}fxy'
                read_dir = self.dirs['main_run']+'/'+read_suff
                write_suff = self.experiment+f'.evd.{nx:02}dx.{fxy:02}fxy'
                write_dir = _dir(self.dirs['main_run']+'/'+write_suff)
                run_dir = self.dirs['main_run']+'/run.'+write_suff

                ds = matern.get_matern_dataset(run_dir,
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
                Q = evds['Q'].sel(Nx=nx,Fxy=fxy).T.values
                B = Q @ HmQ

                # Do the EVD
                D,Uhat = linalg.eigh(B)

                # eigh returns in ascending order, reverse
                D = D[::-1]
                Uhat = Uhat[:,::-1]

                U = Q @ Utilde

                tmpds = xr.Dataset()
                tmpds['U'] = rp.to_xda(U,evds,expand_dims={'Nx':nx,'Fxy':fxy})
                tmpds['D'] = rp.to_xda(D,evds,expand_dims={'Nx':nx,'Fxy':fxy})
                dslistFxy.append(tmpds)

                # --- Write out U to apply prior one last time
                smooth2DInput = []
                for ii in range(evds.n_rand):
                    u = obcsw.unpack(U[:,ii]).reindex_like(self.mymodel).values
                    smooth2DInput.append(u)
                smooth2DInput = np.stack(smooth2DInput,axis=0)
                fname = f'{write_dir}/smooth2DInput{smoothOpNb:03}'
                wrmds(fname,smooth2DInput,dataprec=self.dataprec)

                matern.write_matern(write_dir,smoothOpNb=self.smoothOpNb,
                                    Nx=nx,mymask=self.ctrl.mask,xdalike=self.mymodel)

                sim = rp.Simulation(name=f'{nx:02d}dx_{fxy:02}fxy_evd_oi',
                                    namelist_dir=self.dirs['namelist_apply'],
                                    run_dir=run_dir,
                                    obs_dir=write_dir,
                                    **self.dsim)

                # launch job
                sim.link_to_run_dir()
                sim.write_slurm_script()
                jid = sim.submit_slurm(**self.slurm)
                jid_list.append(jid)

            # --- Continue saving eigenvalues
            dslistNx.append(xr.concat(dslistFxy,dim='Fxy'))

        newds = xr.concat(dslistNx,dim='Nx')
        atts = evds.attrs.copy()
        evds = xr.merge([newds,evds])
        evds.attrs = atts
        evds.to_netcdf(self.dirs['nctmp']+f'/{self.experiment}_proj2.nc')

        # --- Pass on to next stage
        self.submit_next_stage(next_stage='save_the_evd',
                               jid_depends=jid_list,mysim=sim)

    def save_the_evd(self):

        evds = xr.open_dataset(self.dirs['nctmp']+f'/{self.experiment}_proj2.nc')
        evds['filternorm'].load();

        utildeNx = []
        for nx in self.NxList:
            utildeFxy = []
            for fxy in self.FxyList:

                # --- Prepare directories
                read_suff = self.experiment+f'.evd.{nx:02}dx.{fxy:02}fxy'
                read_dir = self.dirs['main_run']+'/'+read_suff
                ds = matern.get_matern_dataset(run_dir,
                                               smoothOpNb=self.smoothOpNb,
                                               xdalike=self.mymodel,
                                               sample_num=np.arange(self.n_rand),
                                               read_filternorm=False)
                ds = ds.sortby(list(self.mymodel.dims))
                ds['ginv'].load();

                Utilde = []
                filternorm = evds['filternorm'].sel(Nx=nx,Fxy=fxy).values
                for s in ds.sample.values:
                    ut = self.ctrl.pack(ds['ginv'].sel(sample=s))*filternorm
                    Utilde.append(ut)
                Utilde = np.array(Utilde).T
                utildeFxy.append(rp.to_xda(Utilde,evds,
                                           expand_dims={'Nx':nx,'Fxy':fxy}))

            utildeNx.append(xr.concat(utildeFxy,dim='Fxy'))

        evds['Utilde'] = xr.concat(utildeNx,dim='Nx')
        evds.to_netcdf(self.dirs['netcdf']+f'/{self.experiment}_evd.nc')

# ---------------------------------------------------------------------
# Stuff for organizing each stage of the run
# ---------------------------------------------------------------------
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

        fname = self.dirs['main_run']+'/submit_oi.sh'
        with open(fname,'w') as f:
            f.write(file_contents)
        return fname



# ----
# Helpers
# -----

def _dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return dirname

