
#Sierra Leone Time Series
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns 

data_dir = '/Users/disha/downloads/'
infile = 'chirps-v2.0.monthly.nc'
CHIRPS = xr.open_dataset(data_dir+infile)
#%%
minlat = 7; maxlat = 10
minlon = -13; maxlon = -10
SierraLeoneSlice =  CHIRPS.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
SierraLeoneSlice.load()
SierraLeoneSlice = SierraLeoneSlice.mean(dim=["longitude", "latitude"]).to_pandas()
SierraLeoneSlice.plot()

#%%
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(SierraLeoneSlice)

#%%
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(SierraLeoneSlice, lags=50)

# %%
