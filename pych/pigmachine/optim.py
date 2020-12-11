"""Driver for optimization using mjlosch/optim_m1qn3
Quasi-Newton optimization with MITgcm
"""

import os
import sys
import json
import subprocess
import warnings
import numpy as np
from shutil import rmtree
import rosypig as rp

class OptimDriver:
    """Defines the driver for optimizing some control variable in the MITgcm

    Example
    -------
    To start an experiment

        >>> myoptim = OptimDriver('myoptim')
        >>> myoptim.start(main_dir,dsim,...)

    see the start method for descriptions on input parameters.
    By default, this starts the whole show, but a different starting point
    can be specified.

    To pick up at a specific stage, e.g. specified by the string mystage

        >>> myoi = OIDriver('myoi',stage=mystage)

    """
    slurm = {'be_nice':True,
             'max_job_submissions':9,
             'dependency':'afterany'}
    conda_env = 'py38_tim'
    obcs_nml=''
    genarr2d_nml=''
    genarr3d_nml=''
    gentim2d_nml=''
    ctrl_packname='pigmachine_ctrl'
    cost_packname='pigmachine_cost'
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
            the optimization iteration, and "gcm" or "optim" for starting
            with the MITgcm run, or the optimization step for that iter
        optim_iter : int, optional
            iteration number
        """
        self.experiment = experiment
        self._send_to_stage(stage,optim_iter)

    def start(self,main_dir,dsim,
              startat='gcm',optim_iter=0,**kwargs):
        """Start the experiment by writing everything we need to file,
        to be read in later by "pickup"

        Parameters
        ----------
        main_dir : str
            main directory where all the mitgcm run directories will launch from
        dsim : dict
            containing the base parameters for a rosypig.Simulation
            with everything but run_dir
            see rosypig.Simulation for details
        kwargs
            Are passed to override the default class attributes
        """

        # --- make some dirs
        for mydir in ['json',main_dir]:
            _dir(mydir);

        # --- Write the directories
        def write_json(mydict,mysuff):
            json_file = f'json/{self.experiment}'+mysuff
            with open(json_file,'w') as f:
                json.dump(mydict, f)

        # make dirs dict, for now?
        dirs={'main_dir':main_dir}
        write_json(dirs,'_dirs.json')
        write_json(dsim,'_sim.json')
        if kwargs !={}:
            write_json(kwargs,'_kwargs.json')

        # --- "pickup" experiment at startat
        self.pickup()
        dsim.pop('name')
        startsim = rp.Simulation('startmeup',**dsim)
        self.submit_next_stage(next_stage=startat,optim_iter=optim_iter,
                               mysim=startsim)

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

        # --- Carry these things around
        self.main_dir = dirs['main_dir']
        self.dsim = dsim

        # --- If kwargs exist, use to rewrite default attributes
        if kwargs is not None:
            for key,val in kwargs.items():
                self.__dict__[key] = val

# ----------------------------------------------------------------
# Methods for organizing each stage
# ----------------------------------------------------------------
    def _send_to_stage(self,stage,optim_iter):
        possible_stages = ['gcm','optim']

        if stage in possible_stages:
            self.pickup()
            eval(f'self.run_{stage}({optim_iter})')
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
                slurmstring=f'sbatch --dependency=afterany:{jid_depends} {bashname}'
        else:
            slurmstring=f'sbatch {bashname}'

        pout = mysim.launch_the_job(slurmstring)

    def write_bash_script(self,stage,optim_iter,mysim):
        """Write a bash script for the next experiment stage
        """
        name=f'optim{optim_iter:03d}'
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
                '"from pych.pigmachine import OptimDriver;'+\
                f'optim = OptimDriver(\'{self.experiment}\',\'{stage}\',{optim_iter})"\n'

        fname = os.path.join(self.main_dir,f'submit_{self.experiment}.sh')
        with open(fname,'w') as f:
            f.write(file_contents)
        return fname

# ----------------------------------------------------------------
# Methods for actually running the MITgcm or optim_m1qn3
# ----------------------------------------------------------------
    def run_gcm(self,optim_iter):
        """run the MITgcm at the given optimization iteration"""

        # iter0
        # 0. create sim
        run_dir = self.get_run_dir(optim_iter)
        sim = rp.Simulation(run_dir=run_dir,**self.dsim)

        # 1. write data.ctrl
        self.write_ctrl_namelist(optim_iter)

        # 2. write data.optim
        self.write_optim_namelist(optim_iter)

        # 3. link and launch
        sim.link_to_run_dir()
        sim.write_slurm_script()
        jid = sim.submit_slurm(**self.slurm)

        self.submit_next_stage(next_stage='optim',optim_iter=optim_iter,
                               jid_depends=jid,mysim=sim)

    def run_optim(self,optim_iter):
        print(' -- run_optim --')
        print(self.__dict__)

    def get_run_dir(self,optim_iter):
        return os.path.join(self.main_dir,f'run_ad.{optim_iter:03d}')

    def write_ctrl_namelist(self,optim_iter):
        """write data.ctrl into run directory"""

        doInitxx = '.TRUE.' if optim_iter==0 else '.FALSE.'
        doMainUnpack = '.FALSE.' if optim_iter==0 else '.TRUE.'

        nml =   " &CTRL_NML\n"+\
               f" doInitxx      = {doInitxx},\n"+\
               f" doMainUnpack  = {doMainUnpack},\n"+\
                " doMainPack    = .TRUE.,\n"+\
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

        fname = os.path.join(self.get_run_dir(optim_iter),'data.ctrl')
        with open(fname,'w') as f:
            f.write(nml)

    def write_optim_namelist(self,optim_iter):
        """write data.optim into run_directory"""

        nml =   f" &OPTIM\n optimcycle = {optim_iter},\n &"
        fname = os.path.join(self.get_run_dir(optim_iter),'data.optim')
        with open(fname,'w') as f:
            f.write(nml)


def _dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return dirname
