"""
Some basic operations for temporally averaging/filtering/grouping the observations
"""

import numpy as np
import xarray as xr

def calc_variability(xda):
    """Return dataset with averaged and filtered fields
    averaged over the time axis


        annual_mean :  dim = 'year'
            yearly average of field
        subannual_var : dim = 'time' (original)
            variability about the annual average (xda - annual_avg)
        monthly_mean : dim = 'month'
            annual cycle, climatology, (annual average removed)
        submonthly_var : dim = 'time' (original)
            deviation from annual average and climatology (xda-annual_avg-monthly_mean)
        interannual_var : dim = 'time' (original)
            deviation from climatology (seasonal or annual cycle) (xda - monthly_mean)
        daily_mean : dim = 'dayofyear'
            average submonthly variability for that day of year

    All mean values have a standard deviation partner '*_std'


    Parameters
    ----------
    xda : xarray DataArray
        field to be temporally filtered

    Returns
    -------
    ds : xarray Dataset
        withe the fields described above
    """

    ds = xr.Dataset()

    # Annual averaging and filtering
    ds['annual_mean'] = xda.groupby('time.year').mean('time')
    ds['annual_std'] = xda.groupby('time.year').std('time')
    ds['subannual_var'] = xda.groupby('time.year') - ds['annual_mean']

    # Monthly averaging and filtering
    ds['monthly_mean'] = ds['subannual_var'].groupby('time.month').mean('time')
    ds['monthly_std'] = ds['subannual_var'].groupby('time.month').std('time')
    ds['submonthly_var'] = ds['subannual_var'].groupby('time.month') - ds['monthly_mean']
    ds['interannual_var'] = xda.groupby('time.month') - ds['monthly_mean']

    # Daily averaging
    ds['daily_mean'] = ds['submonthly_var'].groupby('time.dayofyear').mean('time')
    ds['daily_std'] = ds['submonthly_var'].groupby('time.dayofyear').std('time')

    return ds
