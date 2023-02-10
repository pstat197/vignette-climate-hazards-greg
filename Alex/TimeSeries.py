# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 05:48:55 2023

@author: foamy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 01:05:29 2023

@author: foamy
"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import pymannkendall as mk
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
data_dir = '/Users/foamy/Downloads/CHC/'
infile = 'chirps-v2.0.monthly.nc'

CHIRPS = xr.open_dataset(data_dir+infile)
"""
#set spatial subset dimensions
minlat = 16.; maxlat = 21.
minlon = 24; maxlon = 36.
CHsubS =  CHIRPS.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsubS.load()
CHsubS2 = CHsubS.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubS2.plot()
t1= mk.seasonal_test(CHsubS2, period = 12)
"""
minlat2 = 10.; maxlat2 = 13.
minlon2 = 6; maxlon2 = 10.
CHsubSl =  CHIRPS.sel(latitude=slice(minlat2,maxlat2),longitude=slice(minlon2,maxlon2))
CHsubSl.load()
CHsubSl2 = CHsubSl.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubSl2.plot()
t2= mk.seasonal_test(CHsubSl2, period = 12)
