"""PIGMachine utilities"""

_rhoFresh       = 1000 # kg/m^3
_sec_per_year   = 3600*24*30*12 # s/yr
_kg_per_mt      = 1e9

def convert_units(xda,units_out,keep_attrs=True):
    """convert units to 'units_out'

    Parameters
    ----------
    xda : xarray.DataArray
        with the input field
    units_out : str
        desired units to have xda in
    keep_attrs : bool, optional
        if True, keeps all other meta information, only changing units

    Returns
    -------
    xda : xarray.DataArray
        with the desired units...
    """
    try:
        units=xda.attrs['units']
    except KeyError:
        raise KeyError(f'Could not find units in {xda.name} DataArray attributes')

    # catch easy
    if units == units_out:
        return xda

    # this is just easier than xarray.set_options...
    attrs = xda.attrs if keep_attrs else {}

    if units == 'kg/m^2/s':
        if units_out=='Mt/m^2/yr':
            xda = xda / _kg_per_mt * _sec_per_year
        elif units_out=='m/yr':
            xda = xda / _rhoFresh * _sec_per_year
        else:
            raise NotImplementedError()
    elif units =='Mt/m^2/yr':
        if units_out == 'kg/m^2/s':
            xda = xda * _kg_per_mt / _sec_per_year
        elif units_out=='m/yr':
            xda = xda * _kg_per_mt / _rhoFresh
        else:
            raise NotImplementedError()
    elif units == 'm/yr':
        if units_out == 'kg/m^2/s':
            xda = xda * _rhoFresh / _sec_per_year
        elif units_out == 'Mt/m^2/yr':
            xda = xda / _kg_per_mt * _rhoFresh
        else:
            raise NotImplementedError()
    elif ('dJ/' in units or '[objective_function_units]/' in units) \
         and units_out in ['Mt/yr']:
        print("Assuming sensitivity of meltrate to control variable,"+\
              " with meltrate units [m^3/s]")
        if units_out=='Mt/yr':
            xda = xda / _kg_per_mt * _sec_per_year * _rhoFresh
        else:
            raise NotImplementedError()

        # reset with converted dJ units replaced
        units = units.replace('[objective_function_units]','dJ/')
        ctrlvar_units = units.split('dJ/')[-1]
        units_out = f'({units_out})/'+ctrlvar_units

    else:
        raise NotImplementedError()

    xda.attrs=attrs
    xda.attrs['units'] = units_out
    return xda
