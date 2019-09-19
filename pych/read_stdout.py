#! /workspace/anaconda3/bin/python

# A script to parse grdhck output from STDOUT.0000
# This page was very useful: https://www.vipinajayakumar.com/parsing-text-with-python/

import re
import pandas as pd
import numpy as np

def read_stdout_timing(filepath):
    """
    Parse timing from STDOUT file

    Parameters
    ----------
    filepath : str
        path to STDOUT.* file

    Returns
    -------
    timing_dict : dict
        dictionary containing timing for each section of code
    """

    re_dict = {
        'section': re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\s+Seconds\sin\ssection\s\"(\w+\/?\w+\s?\(?\w+\)?)\s+\[.+\]\"'),
        'time': re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\s+Wall\sclock\stime\:\s+(\d+\.\d+E?.?\d+)')}

    # Parse line by line, when get to the 'timing' part of the STDOUT file
    # first grab the "section"= the part of the code timing is counted toward
    # then the lines after that give the time, grab the Wall Clock Time
    line_dict = {'section':[],'time':[]}
    with open(filepath,'r') as f:
        for line in f:
            key, val = _parse_line(line,re_dict)
            if key is not None:
                line_dict[key].append(val.group(1))

    # Now make a dictionary by 'section'
    # Doing it this way because section and time are not read in as pairs
    timing_dict = {}
    for key, val in zip(line_dict['section'],line_dict['time']):
        timing_dict[key] = np.double(val)

    return timing_dict

def read_grdchk_from_stdout(filepath):
    """
    Parse gradient check information from STDOUT.0000 files

    Parameters
    ----------
    filepath : str
        path to STDOUT.0000 file to be parsed

    Returns
    -------
    grdchk_data : pandas.DataFrame
        grdchk output as a pandas Dataframe
        
    """

    # Step 1: define regular expression dictionary
    rx_dict = {
        'ind' : re.compile(r"\(\w+\.\w+\s\d+\.\d+\)\s=+\sStarts\sgradient-check\snumber\s+(\d+)\s+\(=ichknum\)"),
        'loc' : re.compile(r"\(\w+\.\w+\s\d+\.\d+\)\sgrdchk\spos\:\s+i,j,k=\s+(\d+)\s+(\d+)\s+(\d+)\s+\;\s+bi,bj=\s+(\d+)\s+(\d+)\s+\;\s+iobc=\s+(\d+)\s+\;\s+rec=\s+(\d+)"),
        'fc_plus' : re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\sgrdchk\sperturb\(\+\)fc\:\s+fcpertplus\s+=\s+(.\d+\.\d+E.\d+)'),
        'fc_minus' : re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\sgrdchk\sperturb\(\-\)fc\:\s+fcpertminus\s+=\s+(.\d+\.\d+E.\d+)'),
        'fc_ref' : re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\s+ADM\s+ref_cost_function\s+=\s+(.\d+\.\d+E.\d+)'),
        'grad_fd' : re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\s+ADM\s+finite\-diff\_grad\s+=\s+(.\d+\.\d+E.\d+)'),
        'grad_ad' : re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\s+ADM\s+adjoint\_gradient\s+=\s+(.\d+\.\d+E.\d+)'),
        'stop' : re.compile(r'\(\w+\.\w+\s\d+\.\d+\)\s=+\sEnd\sof\sgradient-check\snumber\s+(\d+)\s+\(ierr=\s+(\d+)\)')
    }



    # Step 2: setup empty container to be filled with results
    grdchk_data = []

    # Step 3: open up the file, parse by line, add to simple conatiner
    with open(filepath,'r') as file_object:

        # Much faster than the suggested content on the website above ...
        for line in file_object:

            # Read individual line
            key, match = _parse_line(line,rx_dict)

            # Extract matching groups according to keys in regex dict
            if key == 'ind':
                ind = match.group(1)
                ind = int(ind)

            if key == 'loc':
                i_loc = match.group(1)
                i_loc = int(i_loc)
                j_loc = match.group(2)
                j_loc = int(j_loc)
                k_loc = match.group(3)
                k_loc = int(k_loc)
                bi_loc= match.group(4)
                bi_loc = int(bi_loc)
                bj_loc= match.group(5)
                bj_loc = int(bj_loc)
                iobc  = match.group(6)
                iobc = int(iobc)
                rec   = match.group(7)
                rec = int(rec)

            if key == 'fc_plus':
                fc_plus = match.group(1)
                fc_plus = float(fc_plus)


            if key == 'fc_minus':
                fc_minus = match.group(1)
                fc_minus = float(fc_minus)

            if key == 'fc_ref':
                fc_ref = match.group(1)
                fc_ref = float(fc_ref)

            if key == 'grad_fd':
                grad_fd = match.group(1)
                grad_fd = float(grad_fd)

            if key == 'grad_ad':
                grad_ad = match.group(1)
                grad_ad = float(grad_ad)
    
            if key == 'stop':
                # This key indicates that we've found all the info
                # for this gradient check index

                # Returns ierr, if error associated with this
                # index, return warning
                ierr = match.group(2)
                ierr = int(ierr)
                if ierr != 0:
                    print('WARNING: Found ierr = ',ierr,' at grdchk #',ind)

                # Now put all info into simple data container as dict
                single_grdchk = {
                    'ind'       : ind,
                    'i'         : i_loc,
                    'j'         : j_loc,
                    'k'         : k_loc,
                    'bi'        : bi_loc,
                    'bj'        : bj_loc,
                    'iobc'      : iobc,
                    'rec'       : rec,
                    'fc_plus'   : fc_plus,
                    'fc_minus'  : fc_minus,
                    'fc_ref'    : fc_ref,
                    'grad_fd'   : grad_fd,
                    'grad_ad'   : grad_ad,
                    'accuracy'  : _compute_gradient_accuracy(grad_fd,grad_ad),
                }

                grdchk_data.append(single_grdchk)


    ## Step 4: Make a Pandas dataframe object
    grdchk_data = pd.DataFrame(grdchk_data)

    ## Index by ind
    grdchk_data.set_index('ind')

    return grdchk_data

def _parse_line(line,rx_dict):
    """
    Regex search based on all defined regular expressions, return key and matching groups
    of first match in regex

    Input one line from stdout file
    """


    for key,value in rx_dict.items():
        match = value.search(line)
        if match:
            return key, match

    # else no matches, return None
    return None, None

def _compute_gradient_accuracy(grad_fd,grad_ad):
    """
    Compute 1 - fd / ad
    """

    if grad_ad == 0:
        return 0
    else:
        return 1 - grad_fd / grad_ad

#read_grdchk_from_stdout('grdchk-files/grdchk_parse_test.data')

