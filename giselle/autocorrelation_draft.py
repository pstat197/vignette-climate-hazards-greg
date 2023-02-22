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
minlat = -17.; maxlat = 28.
minlon = 12; maxlon = 20.
congo =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
congo.load()

# %% seasonal time
drop_2mon = congo.drop([np.datetime64('1981-01-01'), np.datetime64('1981-02-01')], dim='time')
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
