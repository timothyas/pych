"""
SampleDriver class, which defines an object to aid in herding these experiments
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

class SampleDriver:
    """Defines the driver for getting an Eigenvalue decomposition
    associated with the optimal interpolation problem

    Example
    -------
    To start an experiment

        >>> myoi = SampleDriver('myoi')
        >>> myoi.start(dirsdict,simdict,mymodel,obs_std)

    see the start method for descriptions on input parameters.
    By default, this starts the whole show, but a different starting point
    can be specified.

    To pick up at a specific stage, e.g. specified by the string mystage

        >>> myoi = SampleDriver('myoi',stage=mystage)

    """
    slurm = {'be_nice':True,
             'max_job_submissions':9,
             'dependency':'afterany'}
    n_samples = 1000
    dataprec = 'float64'
    NxList  = [5, 10, 15, 20, 30, 40]
    xiList  = [0.5,   1,   2]#,   5]
    isotropic = False
    sorDict = {0.5:1.8, 1:1.6, 2:1.3, 4:1.2, 5:1.2}
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
            this should identify:
            1. the observable quantity
            2. the observation set used (e.g. just obs from one year, or multiple?)
            3. the observational uncertainty profile used
        stage : str
            the stage to start or pickup at when this is called

            'range_approx_one' : first stage of range approximation
        """
        self.experiment = experiment
        self._send_to_stage(stage)


    def start(self, dirs, dsim, mymodel, ctrl_ds,
              startat='range_approx_one', **kwargs):
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
        myctrl = xr.Dataset({'mymodel':mymodel})
        myctrl.to_netcdf(dirs['nctmp']+f'/{self.experiment}_ctrl.nc')
        ctrl_ds.to_netcdf(dirs['nctmp']+f'/{self.experiment}_cds.nc')

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

        modeldims=list(myctrl['mymodel'].dims)

        # --- Carry these things around
        self.dirs = dirs
        self.dsim = dsim
        self.mymodel = myctrl['mymodel']
        self.cds = cds

        self.ctrl = rp.ControlField('ctrl',self.mymodel.sortby(modeldims))

        # --- Determine grid location
        gridloc = 'C'
        if 'uvel' in self.experiment:
            gridloc = 'W'
        elif 'vvel' in self.experiment:
            gridloc = 'S'
        self.gridloc = gridloc

        # --- If kwargs exist, use to rewrite default attributes
        if kwargs is not None:
            for key,val in kwargs.items():
                setattr(self, key, val)

    def _send_to_stage(self,stage):
        possible_stages = ['range_approx_one']

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

            1. generate n_samples gaussian random samples, g_i
            2. solve the linear system for z_i: Az_i = g_i
               to get the range of A^{-1}
            3. use to compute the sample variance of A^{-1}

        ... pass to the next step
        """
        jid_list = []
        for Nx in self.NxList:
            for xi in self.xiList:

                _, write_dir, run_dir = self._get_dirs('range_approx_one',Nx,xi)

                self.smooth_writer(write_dir,
                                   xi=xi,
                                   smooth_apply=False,
                                   num_inputs=self.n_samples)

                matern.write_matern(write_dir,
                                    smoothOpNb=self.smoothOpNb,
                                    Nx=Nx,
                                    mymask=self.ctrl.mask,
                                    xdalike=self.mymodel,
                                    xi=xi,
                                    isotropic=self.isotropic)

                sim = rp.Simulation(name=f'{Nx:02}dx_{xi:02}xi_oi_ra1',
                                    run_dir=run_dir,
                                    obs_dir=write_dir,
                                    **self.dsim)

                # launch job
                sim.link_to_run_dir()

                sim.write_slurm_script()
                jid = sim.submit_slurm(**self.slurm)
                jid_list.append(jid)

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
        maskName = 'mask'+self.gridloc
        if xi not in self.sorDict.keys():
            warnings.warn(f'sor parameter value unknown for xi={xi}, setting sor=1 (=Gauss-Seidel)')
            sor=1
        else:
            sor = self.sorDict[xi]

        smooth2ddims = '' if self.smooth2DDims is None else \
            f' {smooth}Dims({self.smoothOpNb})=\'{self.smooth2DDims}\',\n'

        file_contents = ' &SMOOTH_NML\n'+\
            f' {smooth}Filter({self.smoothOpNb})=1,\n'+\
            smooth2ddims +\
            f' {smooth}Algorithm({self.smoothOpNb})=\'{alg}\',\n'+\
            f' {smooth}NbRand({self.smoothOpNb})={num_inputs},\n'+\
            f' {smooth}JacobiMaxIters({self.smoothOpNb}) = {self.jacobi_max_iters},\n'+\
            f' {smooth}SOROmega({self.smoothOpNb}) = {sor},\n'+\
            f' {smooth}MaskName({self.smoothOpNb}) = "{maskName}",\n'+\
            ' &'
        fname = write_dir+f'/data.smooth'
        with open(fname,'w') as f:
            f.write(file_contents)


    def submit_matern(self,
                      fld,
                      Nx,
                      xi,
                      write_dir,
                      run_dir,
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
                 xdalike=self.mymodel,xi=xi, isotropic=self.isotropic)
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
                '"from pych.pigmachine import SampleDriver;'+\
                f'oid = SampleDriver(\'{self.experiment}\',\'{stage}\')"\n'

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
            write_str = 'matern'+self.gridloc

        else:
            raise NameError(f'Unexpected stage for directories: {stage}')

        numsuff = f'.{Nx:02}dx.{xi:02}xi'
        if read_str is not None:

            # --- Add isotropic suffix here
            if self.isotropic:
                read_str = read_str + '.isotropic'

            read_str += '/run' + numsuff
            read_dir = self.dirs["main_run"] + '/' + read_str
        else:
            read_dir = None

        if write_str is not None:

            # --- Add isotropic suffix here
            if self.isotropic:
                write_str = write_str + '.isotropic'

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


def _unpack_field(fld,packer=None):
    if packer is not None:
        if len(fld.shape)>1 or len(fld)!=packer.n_wet:
            fld = packer.pack(fld)
    else:
        if len(fld.shape)>1:
            fld = fld.flatten() if isinstance(fld,np.ndarray) else fld.values.flatten()
    return fld
