"""
Use pyproj and cartopy to convert South Polar Stereographic to lat lon
"""

import numpy as np
from pyproj import Proj, transform
import cartopy.crs as ccrs

# define starting projection
inProj = Proj(ccrs.SouthPolarStereo().proj4_init)
outProj = Proj(init='epsg:4326') #<- regular latlon

# Assume ds has dataset with usual xmitgcm coordinates... 
# except XC,YC are not lon,lat
# they are in weird stereographic projection coordinates (m)
xc,yc = np.meshgrid(ds.XC.values,ds.YC.values)


# Sometimes there is an offset in the stereographic units
# if it's in lat/lon, then this goes into the ccrs call above
# e.g. for this Dotson/Crosson domain it's
x_offset = -1703000
y_offset = -733000
xc += x_offset
yc += y_offset

# finally, do conversion
lon,lat = transform(inProj,outProj,xc,yc)
