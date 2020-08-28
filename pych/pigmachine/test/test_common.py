"""
Helper functions for all tests
"""
import os
import pytest
import xarray as xr

# Following xmitgcm's lead here
# some expected domains
_experiments = {
    'uvel': {'obs_mean': 'uvel',
             'obs_std':'verr'},
    'vvel': {'obs_mean':'vvel',
             'obs_std':'verr'},
    'theta':{'obs_mean':'theta',
             'obs_std':'thetaerr'},
    'salt': {'obs_mean':'salt',
             'obs_std':'salterr'}
    }

@pytest.fixture(scope='module',params=['uvel'])
def get_single_obs(request):

    fname = f'datasets/test_obs_{request.param}.nc'
    ds = xr.open_dataset(fname)
    for key,val in _experiments[request.param].items():
        ds = ds.rename({val:key})
    return ds

@pytest.fixture(scope='module',params=['uvel','vvel','theta','salt'])
def get_many_obs(request):

    fname = f'datasets/test_obs_{request.param}.nc'
    ds = xr.open_dataset(fname)
    for key,val in _experiments[request.param].items():
        ds = ds.rename({val:key})
    return ds
