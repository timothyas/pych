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

        >>> myoi = OIDriver('myoi',stage=mystage,optim_iter=optimcycle)

    """
    slurm = {'be_nice':True,
             'max_job_submissions':9,
             'dependency':'afterok'}
    conda_env = 'py38_tim'
    optim_iter_max = 1000
    g0 = 1000

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
            the optimization iteration, and "gcm" or "m1qn3" for starting
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
        self.write_json(dsim,'_gcm_sim.json')
        if kwargs !={}:
            self.write_json(kwargs,'_kwargs.json')

        # --- "pickup" experiment at startat
        self.pickup()
        dsim.pop('name')
        startsim = rp.Simulation('startmeup',**dsim)
        self.submit_next_stage(next_stage=startat,optim_iter=optim_iter,
                               mysim=startsim)

    def pickup(self):
        """Read in the files saved in start, prepare self for next stage
        """

        dirs = self.read_json('_dirs.json')
        dsim = self.read_json('_gcm_sim.json')
        kwargs = self.read_json('_kwargs.json')

        # --- Carry these things around
        self.main_dir = dirs['main_dir']
        self.slurm_dir=dirs['slurm_dir']
        self.dsim = dsim
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
        possible_stages = ['gcm','m1qn3']

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
                '"from pych.pigmachine import OptimDriver;'+\
                f'optim = OptimDriver(\'{self.experiment}\',\'{stage}\',{optim_iter})"\n'

        # get current directory, move slurm output to slurm_dir
        file_contents+=f'\nmv $SLURM_JOB_NAME.$SLURM_JOB_ID.* {self.slurm_dir}/'

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
        m1qn3_dir = self.get_m1qn3_dir()
        sim = rp.Simulation(run_dir=run_dir,**self.dsim)

        # 1. write data.ctrl
        self.write_ctrl_namelist(optim_iter)

        # 2. write data.optim
        self.write_optim_namelist(optim_iter)

        # 3. link and launch
        sim.link_to_run_dir()
        if optim_iter>0:
            fname=self.ctrl_packname+f'_MIT_CE_000.opt{optim_iter:04d}'
            _symlink(join(m1qn3_dir,fname), join(run_dir,fname))

        sim.write_slurm_script()
        jid = sim.submit_slurm(**self.slurm)

        self.submit_next_stage(next_stage='m1qn3',optim_iter=optim_iter,
                               jid_depends=jid,mysim=sim)

    def run_m1qn3(self,optim_iter):
        run_dir = self.get_run_dir(optim_iter)
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
            sim=rp.Simulation(**self.dsim)
            self.submit_next_stage(next_stage='gcm',optim_iter=optim_iter+1, mysim=sim)
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


    def get_run_dir(self,optim_iter):
        return os.path.join(self.main_dir,f'run_ad.{optim_iter:04d}')

    def get_m1qn3_dir(self):
        return _dir(os.path.join(self.main_dir,f'run.m1qn3'))

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
        fname = os.path.join(self.get_run_dir(optim_iter),'data.optim')
        with open(fname,'w') as f:
            f.write(nml)


def _dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return dirname

def _jid_from_pout(pout):
    # make a string from pout to get jid
    pout_str = pout.stdout.decode('utf-8').replace('\n','')
    try:
        # convert to list of str and grab last one for job ID
        jid = int(pout_str.split(' ')[-1])
    except ValueError as err:
        jid = -1
        err.args+=(f'Could not convert last word of string to int: ',pout_str)
        raise err
    return jid

def _symlink(src,destination):
    if not os.path.isfile(src):
        raise OSError(f'Cannot symlink from {src}, does not exist.')
    if not os.path.isfile(destination):
        os.symlink(src,destination)

class OptimDataset():
    """Run through all possible optim iteration directories, grabbing:
        - cost function: misfits and ctrl variables separated
        - xx_*, adxx_* ...
    """
    Noptim = 0
    def __init__(self, main_dir,
                 tikh=None,
                 ctrl_packname='pm_ctrl',
                 cost_packname='pm_cost'):
        """OptimDataset

        Parameters
        ----------
        main_dir : str
            which contains the sub-run-directories: run_ad.XXXX, run.m1qn3,
            XXXX = iter
        tikh : float, optional
            tikh parameter
        ctrl_packname, cost_packname : str,optional
            the ctrl and cost packed vector names
            precedes the standard _MIT_CE_000.opt{optim_number} suffix


        """
        self.main_dir = main_dir
        self.tikh = tikh
        self.ctrl_packname = ctrl_packname
        self.cost_packname = cost_packname

        # figure out which step we're on
        lsmain = os.listdir(main_dir)
        self.adj_dirs = [x for x in lsmain if '_ad' in x]
        self.m1qn3_dir = join(main_dir,'run.m1qn3') if 'run.m1qn3' in lsmain else None

        # checks
        try:
            assert len(self.adj_dirs)>0
        except AssertionError:
            raise AssertionError(f'Could not find any run_ad directories in {main_dir}, found: {lsmain}')

        try:
            assert self.m1qn3_dir is not None
        except AssertionError:
            raise AssertionError(f'Could not find run.m1qn3 in {main_dir}, found: {lsmain}')

        # get iterations and current stage
        self.iters = [int(x.split('.')[1]) for x in self.adj_dirs]
        self.Noptim = np.max(self.iters)
        costpack = self.cost_packname+f'_MIT_CE_000.opt{self.Noptim:04d}'
        self.stage = 'gcm' if costpack not in self.m1qn3_dir else 'm1qn3'

    def __repr__(self):
        return  ' --- M1qn3 Optimization --- \n'+\
               f'stage:\t\t{self.stage}\n'+\
               f'iter:\t\t{self.Noptim}\n'+\
               f'Tikhonov\n  parameter:\t{self.tikh}'

    def get_dataset(self,**kwargs):
        """ get a dataset with the goods

        Parameters
        ----------
        kwargs : dict, optional
            passed to xmitgcm.open_mdsdataset

        Returns
        -------
        ds : xarray.Dataset
            with cost, ctrl, and sensitivity fields by iteration
        """
        cost = self.read_cost_function()
        ctrl = self.read_the_xx(**kwargs)
        both = xr.merge([cost,ctrl])

        # flag the Newton's
        _,simuls = self.read_m1qn3()
        both['isNewtonStep'] = xr.full_like(both.iter,True,dtype=bool)
        for myiter in both.iter.values:
            if myiter not in simuls:
                both['isNewtonStep'].loc[{'iter':myiter}] = False

        return both

    def read_cost_function(self):
        """ return a dataset with just costfunction stuff

        Returns
        -------
        ds : xarray.Dataset
            with just cost fields from costfunctionXXXX
        """

        iter = xr.DataArray(np.array(self.iters),
                                  {'iter':np.array(self.iters)},
                                  dims=('iter',))
        dslist=[]
        for k in iter.values:
            rundir = join(self.main_dir,f'run_ad.{k:04d}')

            fname = f'{rundir}/costfunction{k:04d}'
            tmpds=xr.Dataset({'iter':k})
            if os.path.isfile(fname):
                with open(fname,'r') as f:
                    for line in f.readlines():
                        key = line.split('=')[0].replace(' ','')
                        key = key.split('(')[0] if '(' in key else key
                        val = float(line.split('=')[1].split()[0].replace('D','E'))
                        tmpds[key]=val
                dslist.append(tmpds)
            else:
                for fld in dslist[-1].data_vars:
                    if fld != 'iter':
                        tmpds[fld] = np.nan*dslist[-1][fld]
                dslist.append(tmpds)

        ds = xr.concat(dslist,dim='iter')

        # rename for safe merge with controls
        for f in ds.data_vars:
            if 'xx_' in f:
                ds = ds.rename({f:f+'_cost'})

        if self.tikh is not None:
            ds['tikh'] = self.tikh
            ds = ds.set_coords('tikh')


            # `'fc'` term in `costfunctionXXXX` has regularization
            # (i.e. all `mult_` factors) taken into account,
            # but this is not taken into account for each individual term.
            # Apply it here manually.
            ds['regularization'] = xr.zeros_like(ds['fc'])
            for f in ds.data_vars:
                if 'xx_' in f:
                    ds[f] = ds['tikh']*ds[f]
                    ds['regularization'] += ds[f]

            ds['misfit'] = ds['fc']-ds['regularization']

        return ds

    def read_the_xx(self, **kwargs):
        """get ctrl and sensitivity fields

        Parameters
        ----------
        kwargs : dict, optional
            passed to xmitgcm.open_mdsdataset

        Returns
        -------
        ds : xarray.Dataset
            with xx_*, adxx_*, and any diagnostics
        """
        iter = xr.DataArray(np.array(self.iters),
                                  {'iter':np.array(self.iters)},
                                  dims=('iter',))
        droplist=['time','iter']
        dslist=[]
        flddict={}
        for k in iter.values:
            rundir = join(self.main_dir,f'run_ad.{k:04d}')
            lsdir = os.listdir(rundir)

            # limit to just read adxx, xx_
            prefix = [x for x in lsdir if 'xx_' in x]
            prefix = [x.split('.')[0] if ('effective' not in x) and ('reg' not in x) else \
                      x.split('.')[0]+'.'+x.split('.')[1] for x in prefix]
            prefix = list(np.unique(prefix))

            tmpds = open_mdsdataset(rundir,prefix=prefix,**kwargs).squeeze()
            tmpds = tmpds if not set(droplist).issubset(tmpds.coords) else tmpds.drop(droplist)

            # read diagnostics
            kw = deepcopy(kwargs)
            grid_dir=kw.pop('grid_dir', rundir)
            try:
                diagds = open_mdsdataset(join(rundir,'diags'),grid_dir,**kw).squeeze()
            except IndexError:
                diagds = xr.Dataset()

            diagds = diagds if not set(droplist).issubset(tmpds.coords) else diagds.drop(droplist)

            # stick together
            tmpds = tmpds.merge(diagds)

            # get list of all fields
            for f in tmpds.data_vars:
                if f not in flddict.keys():
                    flddict[f] = tmpds[f].copy(deep=True)

            tmpds['iter'] = k
            tmpds = tmpds.set_coords('iter')
            tmpds = tmpds.expand_dims('iter')
            dslist.append(tmpds)

        # copy with nans
        for ds in dslist:
            for key,val in flddict.items():
                if key not in ds:
                    ds[key] = np.nan*val

        ds = xr.concat(dslist,dim='iter')

        # tikhonov
        if self.tikh is not None:
            ds['tikh'] = self.tikh
            ds = ds.set_coords('tikh')
        return ds

    def read_m1qn3(self):
        """read m1qn3_output.txt for quasi-newton details"""
        fname = join(self.m1qn3_dir,'m1qn3_output.txt')

        iterlist = []
        simullist= []
        with open(fname,'r') as f:
            for line in f.readlines():
                if 'iter' in line and 'simul' in line:
                    iterlist.append( int(line.split(',')[0].split()[-1]))
                    simullist.append(int(line.split(',')[1].split()[-1]))
        return iterlist, simullist
