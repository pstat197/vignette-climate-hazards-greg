#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Enbo Zhou"

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

#%% read in climatology
chirps_data_dir  = './chpclim.5050.monthly.nc'
clim = xr.open_dataset(chirps_data_dir)

# Calculate the 12-month sum.
totalPrecip = sum(clim.precip[i,:,:] for i in range(12))

# Plot the histogram
totalPrecip.plot.hist()
plt.show()

# Get the max 12-month precip.
print("Max 12 month Precipitation:", totalPrecip.max())
# Get areas over 75% of the max precip.
maxTotalPrecip = totalPrecip.where(totalPrecip >= 0.5*totalPrecip.max())

# Plot the histogram
maxTotalPrecip.plot.pcolormesh()
plt.show()

#filter the area with annual precip over 100
filtered = totalPrecip.where(totalPrecip > 100)

#Visualize the area with precip over 1000
filtered.plot.pcolormesh()
plt.show()

# Calculate the Jan, Feb, March sum.
JFMPrecip = clim.precip[0,:,:] + clim.precip[1,:,:] + clim.precip[2,:,:]
JFMPrecip.plot.pcolormesh()
plt.show()