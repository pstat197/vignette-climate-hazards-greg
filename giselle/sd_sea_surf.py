#pet_intro
#%%

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import percentileofscore
import intake
import holoviews as hv
import cartopy.crs as ccrs
import geoviews as gv
import hvplot.xarray
#%matplotlib inline
import pandas as pd
import pymannkendall as mk
import time
from datetime import date
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
from scipy.stats import spearmanr, pearsonr, mode, linregress

hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
#%% download 
#%% read in climatology
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
PET_dir = '/home/chc-data-out/people/husak/forCapstone/'
import glob
fnames = sorted(glob.glob(PET_dir+'*.nc'))
#%%
PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date').sortby('date')
#PETmonthly_01.nc  PETmonthly_03.nc  PETmonthly_05.nc  PETmonthly_07.nc  PETmonthly_09.nc  PETmonthly_11.nc
#PETmonthly_02.nc  PETmonthly_04.nc  PETmonthly_06.nc  PETmonthly_08.nc  PETmonthly_10.nc  PETmonthly_12.nc

#pet_mon1 = xr.open_dataset(data_dir+infile)
# %% 
chirps = xr.open_dataset(data_dir+infile)