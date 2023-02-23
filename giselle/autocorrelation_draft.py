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

# %% seasonal time
drop_2mon = congo.drop([np.datetime64('1981-01-01'), np.datetime64('1981-02-01')], dim='time')
mean_seasonal = drop_2mon.rolling(time=3).mean()
#%%
seasonal_mean2_all = congo.groupby(congo.time.dt.year).mean('time')
#%%
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
plot_month2 = mean_seasonal.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month = seasonal_mean2_all.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month3 = congo.mean(dim=["longitude", "latitude"]).to_pandas()
plot_month3.plot()
#t0= mk.seasonal_test(plot_month, period = 40)
##don't run below!!!!
#%% SARIMA 
congo_reset = plot_month3.reset_index()
ts_congo = congo_reset[['time', 'precip']]
ts_congo['time'] = pd.to_datetime(ts_congo['time'])
ts_congo = ts_congo.dropna()
ts_congo.isnull().sum()

ts_congo = ts_congo.set_index('time')
#%%
ts_month_avg = ts_congo['precip'].resample('MS').mean()
ts_month_avg.plot(figsize=(15,6))
#%%
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(ts_month_avg, model='additive')
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

print(adf_test(ts_month_avg))
#%%
ts_s_adj = ts_month_avg - ts_month_avg.shift(12)
ts_s_adj = ts_s_adj.dropna()
ts_s_adj.plot()
#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts_s_adj)
matplotlib.pyplot.show()
plot_pacf(ts_s_adj)
matplotlib.pyplot.show()
#%%
p = range(0, 3)
d = range(1,2)
q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
#%%
#don't run below
#%%
ts_delhi = series_delhi[['Date','AQI']]
#converting 'Date' column to type 'datetime' so that indexing can happen later
ts_delhi['Date'] = pd.to_datetime(ts_delhi['Date'])
ts_delhi.isnull().sum()
ts_delhi = ts_delhi.dropna()
ts_delhi.isnull().sum()

ts_delhi = ts_delhi.set_index('Date')

ts_month_avg = ts_delhi['AQI'].resample('MS').mean()
ts_month_avg.plot(figsize = (15, 6))
plt.show()
# %%
# select DJF
DA_DJF = drop_2mon.sel(time=drop_2mon.time.dt.season=="DJF")

# calculate mean per year
DJF_mean = DA_DJF.groupby(DA_DJF.time.dt.year).mean("time")
# %% plot DJF
minlat = -17.; maxlat = 28.
minlon = 12; maxlon = 20.
plot_djf = DJF_mean.mean(dim=["longitude", "latitude"]).to_pandas()
plot_djf.plot()
t1= mk.seasonal_test(plot_djf, period = 42)

# %% autocorrelation by year
x = sm.tsa.acf(plot_djf.precip, nlags=42)
plt.show()
#%% plot autocorrelation function for dec, jan, feb season
fig1a = tsaplots.plot_acf(x, lags=41)
plt.show()
#%% pacf for dec-jan-feb
series = sm.tsa.pacf(plot_djf.precip, nlags=20)
fig1b = plot_pacf(series, lags=9)
pyplot.show()
#%% run a plot graph of all months in year... long graph?
minlat = -17.; maxlat = 28.
minlon = 12; maxlon = 20.
plot_mean = congo.mean(dim=["longitude", "latitude"]).to_pandas()
plot_mean.plot()
t_testing= mk.seasonal_test(plot_mean, period = 503)
#%%
#congo.load()
#%%
from pandas.plotting import autocorrelation_plot
from pandas import datetime
#import dateparser
#dateparser.parse('12/12/12')
def parser(x):
    if x in range(1980,1990):
        return datetime.strptime('198'+x, '%Y-%m')
    elif x in range(1990,2000):
        return datetime.strptime('199'+x, '%Y-%m')
    elif x in range(2000,2010):
        return datetime.strptime('200'+x, '%Y-%m')
    elif x in range(2010,2020):
        return datetime.strptime('201'+x, '%Y-%m')
    else:
        return datetime.strptime('202'+x, '%Y-%m')
 
#series = 
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
autocorrelation_plot(plot_mean)
pyplot.show()
#%%

# fit an ARIMA model and plot residual errors
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
# load dataset
series.index = plot_mean.index.to_period('M')
#%%
# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())
# %% predict next month:
# Initialize linear regression instance
#linreg = LinearRegression()

# Fit the model to training dataset
#linreg.fit(train_X, train_Y)

# Predict the target variable for training data
#train_pred_Y = linreg.predict(train_X)

# Predict the target variable for testing data
#test_pred_Y = linreg.predict(test_X)
