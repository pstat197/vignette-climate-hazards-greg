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

#filter the area with annual precip over 100
filtered = totalPrecip.where(totalPrecip > 100)

#Visualize the area with precip over 1000
filtered.plot.pcolormesh()
plt.show()