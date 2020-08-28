import os
from shutil import rmtree
import pytest
import xarray as xr
from xarray.testing import assert_equal
from pych.pigmachine.interp_obcs import *
from rosypig import ControlField, Simulation
from rosypig.test.test_common import get_pig_grid
from .test_common import get_many_obs, get_single_obs
from ..oidriver import OIDriver

_test_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.abspath(os.path.join(_test_dir,'../../../'))
_write_dir = os.path.abspath(os.path.join(_test_dir,'oidriver_test'))
_expected_dir = os.path.abspath(os.path.join(_test_dir,'oidriver_expected'))

## TODO:
#   - binary dir is actually on figshare ... for eventual generalities
#   - what to do about mitgcmuv executable search...


def test_start_experiment(get_many_defaults):

    dirs,dsim,expected = get_many_defaults
    ctrl = ControlField('ctrl',expected['mymodel'].sortby(['Z','YC']))
    obs = ControlField('obs',expected['obs_mask'].sortby(['Zobs','Yobs']))
    F = interp_operator_2d([expected['mymodel']['Z'].sortby('Z'),
                            expected['mymodel']['YC'].sortby('YC')],
                           [expected['obs_mask']['Zobs'].sortby('Zobs'),
                            expected['obs_mask']['Yobs'].sortby('Yobs')],
                           pack_index_in=ctrl.wet_ind,
                           pack_index_out=obs.wet_ind)


    testoid = OIDriver('test_start')
    assert testoid.experiment == 'test_start'

    testoid.start(dirs,dsim,
                  mymodel=expected['mymodel'],
                  obs_mask=expected['obs_mask'],
                  obs_std=expected['obs_std'],
                  startat=None)

    assert testoid.dirs==dirs
    assert testoid.dsim==dsim
    assert_equal(testoid.mymodel,expected['mymodel'])
    assert_equal(testoid.obs_std,expected['obs_std'])
    assert_equal(testoid.ctrl.mask,ctrl.mask)
    assert_equal(testoid.obs.mask,expected['obs_mask'])
    assert_equal(testoid.obs.mask,obs.mask)
    assert np.all(F==testoid.F)

def test_change_attributes(get_defaults):

    dirs,dsim,expected = get_defaults
    testoid = OIDriver('test_attrs')

    myslurm={'be_nice':False,'max_job_submissions':100,'dependency':'afterok'}
    n_small=10
    n_over = 5
    n_rand=15
    dataprec='float32'
    NxList=[1,5,10,20,300]
    FxyList=[.25,200]
    conda_env='something'

    testoid.start(dirs,dsim,
                  mymodel=expected['mymodel'],
                  obs_mask=expected['obs_mask'],
                  obs_std=expected['obs_std'],
                  startat=None,
                  slurm=myslurm,n_small=n_small,n_over=n_over,n_rand=n_rand,
                  dataprec=dataprec,NxList=NxList,FxyList=FxyList,
                  conda_env=conda_env)

    assert testoid.slurm==myslurm
    assert testoid.n_small==n_small
    assert testoid.n_over==n_over
    assert testoid.n_rand==n_rand
    assert testoid.dataprec==dataprec
    assert testoid.NxList ==NxList
    assert testoid.FxyList ==FxyList
    assert testoid.conda_env ==conda_env

    # Test now with dict assignment... I guess more as a template
    kwargdict = {'slurm':myslurm,'n_small':n_small,'n_over':n_over,'n_rand':n_rand,
                 'dataprec':dataprec,'NxList':NxList,'FxyList':FxyList,
                 'conda_env':conda_env}
    
    testoid = OIDriver('test_attrs2')
    testoid.start(dirs,dsim,
                  mymodel=expected['mymodel'],
                  obs_mask=expected['obs_mask'],
                  obs_std=expected['obs_std'],
                  startat=None,
                  **kwargdict)
    for key,val in kwargdict.items():
        assert testoid.__dict__[key] == val

def test_slurm_script(get_defaults):

    dirs,dsim,expected = get_defaults

    testoid = OIDriver('test_write')
    testoid.start(dirs,dsim,
                  mymodel=expected['mymodel'],
                  obs_mask=expected['obs_mask'],
                  obs_std=expected['obs_std'],
                  startat=None)
    mysim = Simulation('test',namelist_dir=dirs['namelist'],**dsim)
    test_file = testoid.write_bash_script('range_approx_one',mysim)
    
    with open(_expected_dir+'/main_run/submit_range_approx_one.sh') as f:
        expected_slurm = f.read()

    with open(test_file) as f:
        test_slurm = f.read()

    assert test_slurm==expected_slurm
    
    

@pytest.fixture(scope='function')
def get_many_defaults(get_pig_grid,get_many_obs):
    """test against a bunch of observation types"""

    ds = get_pig_grid
    obsds = get_many_obs
    expected = {}
    expected['mymodel'] = ds['maskC'].isel(XC=0)
    expected['ctrl_mask'] = ds['maskC'].isel(XC=0).sortby(['Z','YC'])
    expected['obs_std'] = obsds['obs_std']
    expected['obs_mask']= obsds['mask']

    dirs = {'main_run':_write_dir+'/main_run',
            'namelist':_expected_dir+'/input',
            'namelist_apply':_expected_dir+'/input.apply',
            'netcdf':_write_dir+'/ncdir'}
    dsim = {'machine':'sverdrup',
            'n_procs':60,
            'exe_file':_expected_dir+'build.fake/mitgcmuv',
            'binary_dir':_expected_dir+'bin.fake',
            'time':'72:00:00'}
    
    yield dirs,dsim,expected

    if os.path.isdir(_write_dir):
        rmtree(_write_dir)
    if os.path.isdir('json'):
        rmtree('json')

@pytest.fixture(scope='function')
def get_defaults(get_pig_grid,get_single_obs):
    """test against a single observation type because it doesn't matter"""

    ds = get_pig_grid
    obsds = get_single_obs
    expected = {}
    expected['mymodel'] = ds['maskC'].isel(XC=0)
    expected['ctrl_mask'] = ds['maskC'].isel(XC=0).sortby(['Z','YC'])
    expected['obs_std'] = obsds['obs_std']
    expected['obs_mask']= obsds['mask']

    dirs = {'main_run':_write_dir+'/main_run',
            'namelist':_expected_dir+'/input',
            'namelist_apply':_expected_dir+'/input.apply',
            'netcdf':_write_dir+'/ncdir'}
    dsim = {'machine':'sverdrup',
            'n_procs':60,
            'exe_file':_expected_dir+'build.fake/mitgcmuv',
            'binary_dir':_expected_dir+'bin.fake',
            'time':'72:00:00'}
    
    yield dirs,dsim,expected

    if os.path.isdir(_write_dir):
        rmtree(_write_dir)
    if os.path.isdir('json'):
        rmtree('json')


