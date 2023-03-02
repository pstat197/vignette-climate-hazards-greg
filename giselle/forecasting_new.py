#autocorrelation
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


hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
# %% 
chirps = xr.open_dataset(data_dir+infile)
#%%
#congo: #c(E 16°48'00"--E 29°25'00"/N 5°31'00"--S 11°55'00")
#did I get coordinates right?
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
congo =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#congo.load()
#congo.precip[0,:,:].plot.pcolormesh()
#%%
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
#plot_month2 = mean_seasonal.mean(dim=["longitude", "latitude"]).to_pandas()
#plot_month = seasonal_mean2_all.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month3 = congo.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month3.plot()
#t0= mk.seasonal_test(plot_month, period = 40)

#%% SARIMA 
congo_reset = plot_month3.reset_index()
ts_congo = congo_reset[['time', 'precip']]
ts_congo['time'] = pd.to_datetime(ts_congo['time'])
ts_congo = ts_congo.dropna()
ts_congo.isnull().sum()

ts_congo = ts_congo.set_index('time')
#%%
ts_month_avg = ts_congo['precip'].resample('MS').mean()
ts_month_avg.plot(figsize=(40,6))
#%% training and testing (.8/.2)
congo_train = ts_month_avg.iloc[0:468]
congo_test = ts_month_avg.iloc[468:]
#%%
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(congo_train, model='additive')
fig = decomposition.plot()
plt.show()
#%%
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used',\
    'Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

print(adf_test(congo_train))
#%%
#define function for kpss test
from statsmodels.tsa.stattools import kpss
#define KPSS
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
      kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
print(kpss_test(congo_train)) #kpss p-value too high (0.1 or above. need to shift)
#%%
ts_s_adj = congo_train - congo_train.shift(12)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot(figsize=(40,6))
#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts_s_adj, lags=60)
matplotlib.pyplot.show()
plot_pacf(ts_s_adj, lags=100, method = 'ywm')
matplotlib.pyplot.show()
#.set_ticklabels([0, 2, 4 ,8 ,16 ,32, 64, 128, 256])
#ax.xaxis.set_ticks
#%%
#ARIMA(0, 1, 1)x(2, 1, 0, 12)
#(1,0,0)*(2,1,0, 12): when ran alex's code: also got around same p-value
#%% not diff
mod = sm.tsa.statespace.SARIMAX(congo_train,
                                order=(1,0,0),
                                seasonal_order=(2, 1, 0, 12))

results = mod.fit(method = 'powell')
print(results.summary().tables[1])

# %%
results.plot_diagnostics(figsize=(18, 8))
plt.show()
# %%
#UNTIL 2033
pred_uc = results.get_forecast(steps=150)
pred_ci = pred_uc.conf_int()
ax = ts_month_avg.iloc[350:].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('time')
ax.set_ylabel('precip')
plt.legend()
plt.show()
#%% UNTIL 2025?
pred_uc = results.get_forecast(steps=60) #only steps was changed
pred_ci = pred_uc.conf_int()
ax = ts_month_avg.iloc[350:].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Time')
ax.set_ylabel('Precip')
plt.legend()
plt.show()
#%% 
y_forecasted2 = pred_uc.predicted_mean
y_truth2 = congo_test['2020-01-01':]
mse2 = ((y_forecasted2 - y_truth2) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse2, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse2), 2)))

#195.03 (last two years)
#rmse: 13.97
###IGNORE BELOW!!!!!!!!!!!!!!!!:

# %%
