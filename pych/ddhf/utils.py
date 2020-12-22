"""helpers"""

try:
    from class_data.netcdf import DataNetCDF
except ImportError:
    pass

def get_dataobj(thredd,params,add_noise=True):
    """return DataNetCDF object from data-driven-collab
    based on ThreddsDataset

    Parameters
    ----------
    thredd : ThreddsDataset
        describing NetCDF data to get
    params : dict
        for initializing the DataNetCDF object
    add_noise : bool, optional
        add noise or not, noise level defined in params

    Returns
    -------
    dataobj : ddc.DataNetCDF
    """

    xtrain = thredd.get_dataset(decode_times=False).squeeze()['water_temp']
    vtrain = xtrain.stack(x=('lat','lon')).values.T

    dataobj = DataNetCDF(myparams)
    dataobj.set_values(vtrain)
    dataobj.set_times(xtrain['time'].values)
    dataobj.normalize()
    if add_noise:
        dataobj.add_noise()

    return dataobj

def get_params(delta_t=3,
               Nxi=4,
               Nxo=None,
               tikh=0.1,
               noise=0.01
               reservoir_dimension=500,
               spectral_radius=0.9,
               sparsity=0.2,
               leak_rate=1.,
               sigma=0.9,
               random_state=11111,
               training_method='least-squares'):

    # For now using Nx to define system dimension
    input_dimension=Nxi**2
    output_dimension=Nxo**2 if Nxo is not None else input_dimension

    params = {'data':
                  {'input_dimension': input_dimension,
                   'output_dimension': output_dimension,
                   'time_dimension': 0,
                   'noise':noise,
                   'delta_t': delta_t,
                   'plot_label':'truth'},
              'tools':
                  {'input_dimension': input_dimension,
                   'output_dimension': output_dimension,
                   'training_method':training_method,
                   'reservoir_dimension': reservoir_dimension,
                   'spectral_radius':spectral_radius,
                   'sparsity': sparsity,
                   'leak_rate': leak_rate,
                   'random_state': random_state,
                   'sigma': sigma,
                   'tikhonov_parameter':tikh,
                   'store_matrices':True},
              'model':
                  {'time_dimension': None,
                   'training_method': 'pinv',
                   'num_layers': 1,
                   'toolkit': None,
                   #'group_size': 4,
                   #'local_halo': 2,
                   #'num_groups': 3,
                   #'group_inlist': [],
                   #'group_outlist': [],
                   #'system_dimension': output_dimension,
                   #'system_dimensions': [Nxo,Nxo],
                    },
              'plot':
                  {'local system_dimension':output_dimension,
                   'training_method': training_method,
                   'reservoir_dimension': reservoir_dimension,
                   'spectral_radius':spectral_radius,
                   'sparsity': sparsity,
                   'leak_rate': leak_rate,
                   'random_state': random_state}
             }
    return params
