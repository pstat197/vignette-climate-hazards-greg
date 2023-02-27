#forecasting draft
#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import intake
import holoviews as hv
import cartopy.crs as ccrs
import geoviews as gv
import hvplot.xarray
#%matplotlib inline
import pandas as pd
import pymannkendall as mk
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot
import warnings
import itertools
import matplotlib
import pmdarima as pm
#%%
hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
# %% 
chirps = xr.open_dataset(data_dir+infile)
#%%
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
congo =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#%%
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
#plot_month2 = mean_seasonal.mean(dim=["longitude", "latitude"]).to_pandas()
#plot_month = seasonal_mean2_all.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month3 = congo.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month3.plot()