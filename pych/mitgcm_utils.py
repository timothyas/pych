"""some random one off functions"""

import os
from glob import glob
from .utils import search_and_replace

def set_data_singleCPUIO(fname):

    # Run through file and replace single cpu io false to true
    found = search_and_replace(fname,
                               search= ' useSingleCpuIO=.FALSE.',
                               replace=' useSingleCpuIO=.TRUE.')
    if not found:
        found = search_and_replace(fname,
                           search= ' &PARM01',
                           replace=' &PARM01\n useSingleCpuIO=.TRUE.')

def all_verification_singleCPUIO(mitgcm_dir):

    all_data_files = glob(mitgcm_dir+'/verification/*/input*/data')
    for datafile in all_data_files:
        set_data_singleCPUIO(datafile)
