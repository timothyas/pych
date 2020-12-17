"""
I/O routines
"""

import xarray as xr

_vars = {'water_u':'tzyx',
         'water_v':'tzyx',
         'water_temp':'tzyx',
         'salinity':'tzyx',
         'surf_el':'tyx'}

class ThreddsDataset():
    """Access HYCOM+NCODA Gulf of Mexico 1/25 degree reanalysis
    (GOMu0.04/expt_50.1)
    from THREDDS Data Server:
    https://ncss.hycom.org/thredds/catalogs/GOMu0.04/expt_50.1.html?dataset=GOMu0.04-expt_50.1-2012

    Reanalysis details:
    https://www.hycom.org/data/gomu0pt04/expt-50pt1
    """
    def __init__(self,
                 lon=(0,541,1),
                 lat=(0,346,1),
                 depth=(0,40,1),
                 time=(0,2927,1),
                 varlist=['water_u','water_v','water_temp','salinity','surf_el']):

        def _istuple(x):
            return x if isinstance(x,tuple) else (0,x,1)
        self.lon = _istuple(lon)
        self.lat = _istuple(lat)
        self.depth = _istuple(depth)
        self.time = _istuple(time)
        self.varlist = list(varlist)
        self.url = self.set_url()

    def __repr__(self):
        return f'Thredds dataset size {self.gb:.3f} GB from:\n{self.url}'

    def get_gigs(self):
        """Estimate size of dataset in GigaBytes

        Returns
        -------
        dsgb : float
            full dataset size in GB
        """
        single_prec=4
        double_prec=8
        coords = (self.lon[1]+self.lat[1]+self.depth[1]+self.time[1])*double_prec

        var3d = (self.lon[1]*self.lat[1]*self.depth[1]*self.time[1])*single_prec
        var2d = (self.lon[1]*self.lat[1]*self.time[1])*single_prec

        dsgb = coords
        for key in self.varlist:
            try:
                mysize = var3d if 'z' in _vars[key] else var2d
            except KeyError:
                raise NotImplementedError(f'Need to add {key} to _vars dict for shape')
            dsgb+=mysize
        return dsgb*1e-9


    def set_url(self,lon=None,lat=None,depth=None,time=None,varlist=None):
        """set the url based on how to subset the data
        any changes change the object's property

        Parameters
        ----------
        lon, lat, depth, time : tuple, optional
            set these parameters to get appropriate url with datasubset

        Returns
        -------
        url : str
            with url to netcdf data on THREDDS Server
        """
        self.lon = lon if lon is not None else self.lon
        self.lat = lat if lat is not None else self.lat
        self.depth = depth if depth is not None else self.depth
        self.time = time if time is not None else self.time
        self.varlist = list(varlist) if varlist is not None else self.varlist
        self.gb = self.get_gigs()

        url = "https://tds.hycom.org/thredds/dodsC/GOMu0.04/expt_50.1/data/netcdf/2012?"
        for name, index in zip(['depth','lat','lon','time'],
                               [self.depth, self.lat, self.lon, self.time]):
            url += f"{name}[{index[0]}:{index[-1]}:{index[1]-1}],"

        for var in self.varlist:
            url+=var+","
        url = url[:-1] if url[-1]==',' else url
        return url

    def get_dataset(self,**kwargs):
        """Actually get the xarray dataset

        Returns
        -------
        ds : xarray.Dataset
            with the goods
        """
        ds = xr.open_dataset(self.url,**kwargs)

        for fld,myslice in zip(['time','depth'],
                               [slice(*self.time),slice(*self.depth)]):
            f1=f'{fld}_1'
            ds = ds.sel({f1:myslice}).assign_coords({f1:ds[fld].rename({fld:f1})}).drop(fld).rename({f1:fld})
        return ds
