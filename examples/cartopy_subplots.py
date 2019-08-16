#!/bin/bash

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# There are two identical ways to create a bunch of subplots
# this shows both

# Also this explains how to set_extent for cartopy in both
# situations

# Note that "extent" is unique to a GeoAxesSubplot rather than
# just an AxesSubplot. This happens by adding a "projection" or "crs" argument
# when creating the subplot

Nrows=3
Ncols=2

# In this example, we just want to set that lon/lat extent
# of the map... 
lon_min_bound = -85
lon_max_bound = -30
lat_min_bound = -65
lat_max_bound = 15


# --- Method 1 
fig, array_of_axes = plt.subplots(nrows=Nrows,
                     ncols=Ncols,
                     figsize=(12,6),
                     sharex='row', # <- this might be useful...
                     sharey='col',
                     gridspec_kw={'hspace':0.02, 'wspace':0.02}, #<- not essential, just showing
                     subplot_kw={'projection':ccrs.PlateCarree} #<- changes AxesSubplot to GeoAxesSubplot
                     )

# Then we can loop over the axes like this...
print(array_of_axes) # to see the shape
array_of_axes = array_of_axes.flatten()
for ax in array_of_axes:

    # Note without setting projection earlier, we wouldn't have access
    # to this 'set_extent' function
    ax.set_extent([lon_min_bound, lon_max_bound, lat_min_bound, lat_max_bound], 
                  crs=ccrs.PlateCarree())

# --- Method 2

fig = plt.figure(figsize=(12,6))

for i in range(Nrows*Ncols):

    ax = fig.add_subplot(Nrows,Ncols,i+1,
            projection=ccrs.PlateCarree()) #<- again, this is the critical GeoAxesSubplot part...

    ax.set_extent([lon_min_bound, lon_max_bound, lat_min_bound, lat_max_bound], 
                  crs=ccrs.PlateCarree())
