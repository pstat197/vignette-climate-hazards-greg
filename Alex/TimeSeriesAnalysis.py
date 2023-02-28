# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 04:25:45 2023

@author: foamy
"""

import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn
import cartopy.feature
import matplotlib.pyplot as plt
import pymannkendall as mk
import statsmodels.graphics.tsaplots as sm
import statsmodels.tsa.seasonal as sms
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima

data_dir = '/Users/foamy/Downloads/CHC/'
infile = 'chirps-v2.0.monthly.nc'
# %% 
chirps = xr.open_dataset(data_dir+infile)
#%%
#congo: #c(E 16°48'00"--E 29°25'00"/N 5°31'00"--S 11°55'00")
#did I get coordinates right?
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
congo =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
congo.load()

congo_ps = congo.mean(dim=["longitude", "latitude"]).to_pandas() #convert to pandas

#time series plot 
congo_tst = congo_ps.loc['2021-01-01 00:00:00': '2022-12-01 00:00:00']
congo_tr = congo_ps.loc[: '2020-12-01 00:00:00']
plt.figure(figsize=(10,5))
plt.plot(congo_tr, color='blue',label='Original')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')
#plt.hist(congo_tr)

seaborn.boxplot(x = congo_ps.index.month,  #boxplot
                y = congo_ps['precip'])


#CHECK FOR STATIONARITY
plt.figure(figsize=(20,5))
plt.plot(congo_tr, color='blue',label='Original')   
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')



#CHECK FOR STATIONARITY
plt.figure(figsize=(20,5))
plt.plot(congo_tr, color='blue',label='Original')
plt.plot(congo_tr.rolling(window=12).mean(), label='12-Months Rolling Mean')
plt.plot(congo_tr.rolling(window=12).std(), label='12-Months Rolling STD')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')

print("Observations of Dickey-fuller test")
dftest = adfuller(congo_tr,autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)# p-value is low, reject null hypothesis, stationary data
#decomposition
result=sms.seasonal_decompose(congo_tr, model='additive', period=12) 
result.plot()   


#plot pacf acf
sm.plot_acf(congo_tr,lags=36)
sm.plot_pacf(congo_tr,lags=36, method = "ywm")

#diff at 12, then pacf acf 
congo_12 = congo_tr.diff(periods=12)
plt.plot(congo_12)
sm.plot_acf(congo_12.dropna(),lags=36)
sm.plot_pacf(congo_12.dropna(),lags=36, method = "ywm")
#plt.hist(congo_12)


#CHECK FOR STATIONARITY of differenced 
plt.figure(figsize=(20,5))
plt.plot(congo_12, color='blue',label='Original')
plt.plot(congo_12.rolling(window=12).mean(), label='12-Months Rolling Mean')
plt.plot(congo_12.rolling(window=12).std(), label='12-Months Rolling STD')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')
"""
#create model 
model=auto_arima(congo_12.dropna(), start_p = 0, start_q = 0,D=1, m = 12,
                 seasonal = True, test = "adf",  trace = True, alpha = 0.05,
                 information_criterion = 'aic', suppress_warnings = True, 
                 stepwise = True)
"""
