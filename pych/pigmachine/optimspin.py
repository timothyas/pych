"""Driver for optimization using mjlosch/optim_m1qn3
Quasi-Newton optimization with MITgcm
"""

import os
from os.path import join
from copy import deepcopy
import sys
import json
import subprocess
import warnings
import numpy as np
from shutil import rmtree, copyfile
import xarray as xr
from xmitgcm import open_mdsdataset
import rosypig as rp

from .optim import (
        _dir, _jid_from_pout, _symlink)

class OptimSpinDriver:
    """Defines the driver for optimizing some control variable in the MITgcm
    Do a little spinup between each simulation...

    Example
    -------
    To start an experiment

        >>> myoptim = OptimSpinDriver('myoptim')
        >>> myoptim.start(main_dir,adm_sim,...)

    see the start method for descriptions on input parameters.
    By default, this starts the whole show, but a different starting point
    can be specified.

    To pick up at a specific stage, e.g. specified by the string mystage

        >>> myoi = OIDriver('myoi',stage=mystage,optim_iter=optimcycle)

    """
    slurm = {'be_nice':False,
             'max_job_submissions':10,
             'dependency':'afterok'}
    conda_env = 'py38_tim'
    optim_iter_max = 1000
    g0 = 1000

    # Spinup stuff
    nIter0 = 0
    nIterSpinFinal = 0

    # Control Package
    obcs_nml=''
    genarr2d_nml=''
    genarr3d_nml=''
    gentim2d_nml=''
    ctrl_packname='pm_ctrl'
    cost_packname='pm_cost'

    # Optim m1qn3
    dfminFrac   = 0.1   # cold start, expected fractional reduction in cost
    numiter     = 100   # max num. iterations to determine step direction
    nfunc       = 100   # max num. fun calls (mitgcmuv_ad) per iter
    epsg        = 1e-6  # stopping criterion for ||g_k||/||g_0||
    nupdate     = 8     # number of gradients and ctrl vectors to approx Hess.Inv.
                        # this is m in m1qn3 documentation
                        # m1qn3 requires working vector of size:
                        # ndz = nupdate*(2*Nctr l+ 1) + 4*Nctrl

    iprint      = 5     # how much to print, 5 is max printing
    epsx        = 1e-6  # if sup. norm is used, minimum resolution for ctrl vector
                        # for two points to be indistinguishable
    eps         = -1    # if >0, overwrites epsg and epsx
    def __init__(self, experiment,stage=None,optim_iter=None):
        """Should this initialize? Or do we want another method to do it?

        Parameters
        ----------
        experiment : str
            an identifying name for the experiment
            this should identify:
            1. the control vector
            2. potentially any uncertainty info different than "normal"
        stage : str, optional
            the optimization iteration, and "adm", "fwd", or "m1qn3" for starting
            with the MITgcm run, or the optimization step for that iter
        optim_iter : int, optional
            iteration number
        """
        self.experiment = experiment
        self._send_to_stage(stage,optim_iter)

    def start(self,main_dir,
              adm_sim,fwd_sim,
              stage='adm',optim_iter=0,
              **kwargs):
        """Start the experiment by writing everything we need to file,
        to be read in later by "pickup"

        Parameters
        ----------
        main_dir : str
            main directory where all the mitgcm run directories will launch from
        adm_sim, fwd_sim : dict
            containing the base parameters for a rosypig.Simulation
            with everything but run_dir
            see rosypig.Simulation for details
        kwargs
            Are passed to override the default class attributes
            Additionally:
            optimx : str, optional
                path to m1qn3 executable, default is main_dir/build.m1qn3/optim.x
        """

        # --- make some dirs
        slurm_dir = os.path.join(main_dir,'slurm')
        for mydir in ['json',main_dir,slurm_dir]:
            _dir(mydir);

        dirs={'main_dir':main_dir,'slurm_dir':slurm_dir}
        self.write_json(dirs,'_dirs.json')
        self.write_json(adm_sim,'_adm_sim.json')
        self.write_json(fwd_sim,'_fwd_sim.json')

        # Requiring these arguments, but throwing them into kwargs b/c it's easy
        if kwargs !={}:
            self.write_json(kwargs,'_kwargs.json')

        # --- "pickup" experiment at stage
        self.pickup()
        adm_sim.pop('name')
        startsim = rp.Simulation('startmeup',**adm_sim)
        self.submit_next_stage(next_stage=stage,optim_iter=optim_iter,
                               mysim=startsim)

    def pickup(self):
        """Read in the files saved in start, prepare self for next stage
        """

        dirs = self.read_json('_dirs.json')
        adm_sim = self.read_json('_adm_sim.json')
        fwd_sim = self.read_json('_fwd_sim.json')
        kwargs = self.read_json('_kwargs.json')

        # --- Carry these things around
        self.main_dir = dirs['main_dir']
        self.slurm_dir=dirs['slurm_dir']
        self.adm_sim = adm_sim
        self.fwd_sim = fwd_sim
        self.optimx = os.path.join(self.main_dir,'build.m1qn3','optim.x')

        # --- If kwargs exist, use to rewrite default attributes
        if kwargs is not None:
            for key,val in kwargs.items():
                self.__dict__[key] = val

    # --- Write/Read dictionaries to start/pickup
    def write_json(self,mydict,mysuff):
        json_file = os.path.join('json',self.experiment+mysuff)
        with open(json_file,'w') as f:
            json.dump(mydict, f)

    def read_json(self,mysuff):
        json_file = os.path.join('json',self.experiment+mysuff)
        if not os.path.isfile(json_file):
            return None
        with open(json_file,'r') as f:
            mydict = json.load(f)
        return mydict

# ----------------------------------------------------------------
# Methods for organizing each stage
# ----------------------------------------------------------------
    def _send_to_stage(self,stage,optim_iter):
        possible_stages = ['adm','fwd','m1qn3']

        if stage in possible_stages:
            self.pickup()
            if stage in ('adm','fwd'):
                eval(f'self.run_gcm(optim_iter={optim_iter}, gcm_type="{stage}")')
            else:
                eval(f'self.run_m1qn3(optim_iter={optim_iter}')
        else:
            if stage is not None:
                raise NameError(f'Incorrectly specified stage: {stage}.\n'+\
                    'Available possibilities are: '+str(possible_stages))

    def submit_next_stage(self,next_stage, optim_iter, mysim, jid_depends=None):
        """Write a bash script and submit, which will execute next stage
        of experiment

        Parameters
        ----------
        next_stage : str
            a string with the next stage to execute, see init or start
        optim_iter : int
            optimization cycle iteration
        mysim : rosypig.Simulation
            use the last simulation to create another one
        jid_depends : list of ints or int
            with slurm job id's to wait on before submitting the next stage
        """

        bashname = self.write_bash_script(stage=next_stage,optim_iter=optim_iter,
                                          mysim=mysim)

        if jid_depends is not None:
            jid_depends = [jid_depends] if isinstance(jid_depends,int) else jid_depends
            jid_depends = str(jid_depends).replace(', ',':').replace('[','').replace(']','')
            # catch case of 0 length list
            if jid_depends == '':
                slurmstring=f'sbatch {bashname}'
            else:
                slurmstring=f'sbatch --dependency={self.slurm["dependency"]}:{jid_depends} {bashname}'
        else:
            slurmstring=f'sbatch {bashname}'

        pout = mysim.launch_the_job(slurmstring)

    def write_bash_script(self,stage,optim_iter,mysim):
        """Write a bash script for the next experiment stage
        """
        name=f'opt{optim_iter:04d}_{stage}'
        file_contents = '#!/bin/bash\n\n' +\
            f'#SBATCH -J {name}\n' +\
            f'#SBATCH -o {name}.%j.out\n' +\
            f'#SBATCH -e {name}.%j.err\n' +\
            '#SBATCH -N 1\n' +\
            f'#SBATCH -n {mysim.procs_per_node}\n' +\
            f'#SBATCH -p {mysim.queue_name}\n' +\
            f'#SBATCH -t {mysim.time}\n'

        file_contents += f'\n\neval "$(conda shell.bash hook)"\n'+\
                f'conda activate {self.conda_env}\n\n'+\
                f'python3 -c '+\
                '"from pych.pigmachine import OptimSpinDriver;'+\
                f'optim = OptimSpinDriver(\'{self.experiment}\',\'{stage}\',{optim_iter})"\n'

        # get current directory, move slurm output to slurm_dir
        file_contents+=f'\nmv $SLURM_JOB_NAME.$SLURM_JOB_ID.* {self.slurm_dir}/'

        fname = os.path.join(self.main_dir,f'submit_{self.experiment}.sh')
        with open(fname,'w') as f:
            f.write(file_contents)
        return fname

# ----------------------------------------------------------------
# Methods for actually running the MITgcm or optim_m1qn3
# ----------------------------------------------------------------
    def run_gcm(self, optim_iter, gcm_type):
        """run the MITgcm at the given optimization iteration"""

        # iter0
        # 0. create sim
        run_dir = self.get_run_dir(optim_iter, gcm_type)
        m1qn3_dir = self.get_m1qn3_dir()
        dsim = self.adm_sim if gcm_type =='adm' else self.fwd_sim

        # Handle pickups
        if  (optim_iter > 0 and gcm_type == 'adm') or \
            (optim_iter > 1 and gcm_type == 'fwd'):
            dsim['pickup_dir'] = None
            if gcm_type == 'fwd':
                previous_fwd_dir = self.get_run_dir(optim_iter-1, 'fwd')
            else:
                previous_fwd_dir = self.get_run_dir(optim_iter, 'fwd')

            for suffix in ['meta','data']:

                old_pickup = os.path.join(previous_fwd_dir,
                                          f'pickup.{nIterSpinFinal:010d}.{suffix}')
                new_pickup = os.path.join(run_dir,
                                          f'pickup.{nIter0:010d}.{suffix}')
                copyfile(old_pickup, new_pickup)


        sim = rp.Simulation(run_dir=run_dir,**dsim)

        # 1. write data.ctrl
        self.write_ctrl_namelist(optim_iter, gcm_type)

        # 2. write data.optim
        self.write_optim_namelist(optim_iter, gcm_type)

        # 3. link and launch
        sim.link_to_run_dir()
        if optim_iter>0:
            fname=self.ctrl_packname+f'_MIT_CE_000.opt{optim_iter:04d}'
            _symlink(join(m1qn3_dir,fname), join(run_dir,fname))

        sim.write_slurm_script()
        jid = sim.submit_slurm(**self.slurm)

        next_stage = 'm1qn3' if gcm_type == 'adm' else 'adm'
        self.submit_next_stage(next_stage=next_stage,optim_iter=optim_iter,
                               jid_depends=jid,mysim=sim)

    def run_m1qn3(self,optim_iter):
        run_dir = self.get_run_dir(optim_iter, gcm_type='m1qn3')
        m1qn3_dir = self.get_m1qn3_dir()

        # link optim.x to run directory
        exe = os.path.basename(self.optimx)
        _symlink(self.optimx, join(m1qn3_dir,exe))

        # link over cost vector
        fname = self.cost_packname+f'_MIT_CE_000.opt{optim_iter:04d}'
        _symlink(join(run_dir,fname), join(m1qn3_dir,fname))

        # iter 0: get ctrl vector too
        if optim_iter == 0:
            fname = self.ctrl_packname+f'_MIT_CE_000.opt{optim_iter:04d}'
            _symlink(join(run_dir,fname), join(m1qn3_dir,fname))

        # copy data.ctrl, data.optim
        for fname in ['data.ctrl','data.optim']:
            copyfile(join(run_dir,fname), join(m1qn3_dir,fname))

        # move to run dir and run
        pwd = os.getenv('PWD')
        os.chdir(m1qn3_dir)
        run_cmd = f'./{exe} > stdout.{optim_iter:04d} 2> stderr.{optim_iter:04d}'
        subprocess.run(run_cmd,shell=True)

        # check stderr
        fname=f'stderr.{optim_iter:04d}'
        with open(fname,'r') as f:
            stderr = f.read()
        if len(stderr)>0:
            raise RuntimeError(f'm1qn3 failed, see: {os.path.realpath(fname)}')
        os.chdir(pwd)

        # read gradient norm
        with open(f'{m1qn3_dir}/m1qn3_output.txt','r') as f:
            for line in f.readlines():
                if 'two-norm of g' in line:
                    g_norm = float(line.split('=')[1][:-1].replace('D','E'))

        # set ||g0||
        if optim_iter==0:
            self.g0 = g_norm
            self.write_json
            kwargs = self.read_json('_kwargs.json')
            kwargs = kwargs if kwargs is not None else {}
            kwargs['g0'] = g_norm
            self.write_json(kwargs,'_kwargs.json')

        # submit next optim iter
        if optim_iter+1 < self.optim_iter_max and g_norm > self.epsg*self.g0:
            sim=rp.Simulation(**self.fwd_sim)
            self.submit_next_stage(next_stage='fwd',optim_iter=optim_iter+1, mysim=sim)
        else:
            if g_norm < self.epsg * self.g0:
                print('\t\t---\t\tCongratulations!!\t\t---')
                print('')
                print(f'     ||g_{optim_iter}||_2 / ||g_0||_2 < eps_g = {self.epsg}')
                print('')
                print(f'            g_norm = {g_norm:1.6e}')
                print(f'            optim_iter = {optim_iter}')

            else:
                print(f' --- Reached Maximum Optimization Iterations: {self.optim_iter_max} ---')
                print(' exiting ...')


    def get_run_dir(self, optim_iter, gcm_type):
        basename = 'run_ad' if gcm_type in ('adm','m1qn3') else 'run'
        return os.path.join(self.main_dir,f'{basename}.{optim_iter:04d}')

    def get_m1qn3_dir(self):
        return _dir(os.path.join(self.main_dir,f'run.m1qn3'))

    def write_ctrl_namelist(self, optim_iter, gcm_type):
        """write data.ctrl into run directory"""

        doInitxx = '.TRUE.' if optim_iter==0 else '.FALSE.'
        doMainUnpack = '.FALSE.' if optim_iter==0 else '.TRUE.'
        doMainPack = '.TRUE.' if gcm_type == 'adm' else '.FALSE.'

        nml =   " &CTRL_NML\n"+\
               f" doInitxx      = {doInitxx},\n"+\
               f" doMainUnpack  = {doMainUnpack},\n"+\
               f" doMainPack    = {doMainPack},\n"+\
                "# --- OBCS\n"+\
                self.obcs_nml+\
                " &\n"+\
                " &CTRL_PACKNAMES\n"+\
               f" ctrlname = '{self.ctrl_packname}'\n"+\
               f" costname = '{self.cost_packname}'\n"+\
                " &\n"+\
                " &CTRL_NML_GENARR\n"+\
                "# --- GENARR2D\n"+\
                self.genarr2d_nml+\
                "# --- GENARR3D\n"+\
                self.genarr3d_nml+\
                "# --- GENTIM2D\n"+\
                self.gentim2d_nml+\
                " &"

        run_dir = self.get_run_dir(optim_iter, gcm_type)
        fname = os.path.join(run_dir,'data.ctrl')
        with open(fname,'w') as f:
            f.write(nml)

    def write_optim_namelist(self, optim_iter, gcm_type):
        """write data.optim into run_directory"""
        dmf = f" dfminFrac = {self.dfminFrac:f}\n" if optim_iter==0 else ""

        nml =   f" &OPTIM\n"+\
                f" optimcycle   = {optim_iter},\n"+\
                dmf+\
                f" numiter      = {self.numiter},\n"+\
                f" nfunc        = {self.nfunc},\n"+\
                f" epsg         = {self.epsg},\n"+\
                f" nupdate      = {self.nupdate},\n"+\
                f" iprint       = {self.iprint},\n"+\
                f" epsx         = {self.epsx},\n"+\
                f" eps          = {self.eps},\n"+\
                 " &\n &M1QN3\n &"

        run_dir = self.get_run_dir(optim_iter, gcm_type)
        fname = os.path.join(run_dir,'data.optim')
        with open(fname,'w') as f:
            f.write(nml)
