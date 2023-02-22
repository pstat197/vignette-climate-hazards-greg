#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns 
count_bord = cartopy.feature.NaturalEarthFeature('','admin_0_boundary_lines_land','110m',facecolor='none')

#%% read in climatology
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
data_dir = 'C:/Users/ryanc/Downloads/'
infile = 'chirps-v2.0.monthly.nc'

clim = xr.open_dataset(data_dir+infile)



# Time Series Analysis
#%%
minlat = 10.; maxlat = 13.
minlon = 6.; maxlon = 10.
CHsub_leone =  clim.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),)
CHsub_leone.load()
CHsubSl2 = CHsub_leone.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubSl2.plot()

#%%
CHsubSl2 = CHsub_leone.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubSl2.plot()

CHsubSl2[0:23].plot()
# %%
sns.lineplot(CHsubSl2)
# %%

rolling_mean = CHsubSl2.rolling(12).mean()
rolling_std = CHsubSl2.rolling(12).std()
sns.lineplot(CHsubSl2)

autocorrelation_lag1 = CHsubSl2['precip'].autocorr(lag=1)
print("1 Month Lag: ", autocorrelation_lag1)


autocorrelation_lag1 = CHsubSl2['precip'].autocorr(lag=3)
print("6 Month Lag: ", autocorrelation_lag1)

autocorrelation_lag1 = CHsubSl2['precip'].autocorr(lag=12)
print("12 Month Lag: ", autocorrelation_lag1)


#%%
CHsubSl2_diff12 = CHsubSl2['precip'].diff(periods=12) # differnece by lag 12 to remove seasonality
CHsubSl2_diff12 = CHsubSl2_diff12.dropna() # drop NaNs
print(CHsubSl2_diff12)

#%%
rolling_mean = CHsubSl2_diff12.rolling(12).mean()
rolling_std = CHsubSl2_diff12.rolling(12).std()
print(CHsubSl2)
# %%
plt.plot(CHsubSl2_diff12, color="blue",label="Original Precip Data")
plt.plot(rolling_mean, color="red", label="Rolling Mean Precip")
plt.plot(rolling_std, color="black", label = "Rolling Standard Deviation in Precip")
plt.legend(loc='best')

# plot of time series including rolling mean and rolling stddev
# %%
from statsmodels.tsa.stattools import adfuller
adft = adfuller(CHsubSl2_diff12,autolag="AIC")

output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", 
            
                                                        "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
print(output_df)
#test for stationarity (Dickey-Fuller test)
# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(CHsubSl2_diff12,model='additive', period=12)
decompose.plot()
plt.show()

# Decomposition of data
# %%

# ACF
"""

dta = sm.datasets.sunspots.load_pandas().data
print(CHsubSl2.dtyp)
CHsubSl2.index = pd.Index(sm.tsa.datetools.dates_from_range('1981-01-01', '2022-12-01'))
del CHsubSl2[]
sm.graphics.tsa.plot_acf(CHsubSl2.values.squeeze(), lags=40)
plt.show()

"""