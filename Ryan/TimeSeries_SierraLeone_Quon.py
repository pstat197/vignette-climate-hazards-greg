#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 10:21:09 2023
this will be some code to keep up with the capstone project
@author: husak
"""
#%%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from pmdarima.arima import ARIMA

count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
data_dir = 'C:/Users/ryanc/Downloads/'
infile = 'chirps-v2.0.monthly.nc'

clim = xr.open_dataset(data_dir+infile)

#set spatial subset dimensions
minlat = 7.; maxlat = 10.
minlon = -13; maxlon = -10.
CHsub_leone =  clim.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),)
CHsub_leone.load()
CHsubSl2 = CHsub_leone.mean(dim=["longitude", "latitude"]).to_pandas()

#%%
#The whole time series
#CHsubSl2.plot()
#Zoom in time series
#CHsubSl2[0:23].plot()
rolling_mean = CHsubSl2.rolling(12).mean()
rolling_std = CHsubSl2.rolling(12).std()
#print(rolling_mean, rolling_std)
#plot_acf(CHsubSl2)
#plot_pacf(CHsubSl2)
#CHsubSl2.hist()

log_CHsubl2 = np.log(CHsubSl2)

#plt.plot(CHsubSl2, color="blue",label="Original Precip Data")
#plt.plot(rolling_mean, color="red", label="Rolling Mean Precip")
#plt.plot(rolling_std, color="black", label = "Rolling Standard Deviation in Precip")
#plt.legend(loc='best')
#decompose = seasonal_decompose(CHsubSl2,model='additive', period=12)
#decompose.plot()

#print(CHsubSl2)
box_data, box_lambda = stats.boxcox(CHsubSl2['precip'])

CHsubSl2['precip'] = box_data
#CHsubSl2.hist()

rolling_mean = CHsubSl2.rolling(12).mean()
#plt.plot(rolling_mean, color="red", label="Rolling Mean Precip")

CHsubSl2.plot()
plot_acf(CHsubSl2)
plot_pacf(CHsubSl2)

#%%
CHsubSl2_diff12 = CHsubSl2.diff(12).dropna()
CHsubSl2_diff12.plot()
plot_acf(CHsubSl2_diff12,lags=100)
plot_pacf(CHsubSl2_diff12,lags = 100)


#CHsubSl2_diff12_diff3 = CHsubSl2_diff12.diff(6).dropna()
#CHsubSl2_diff12_diff3.plot()
#plot_acf(CHsubSl2_diff12_diff3)
#plot_pacf(CHsubSl2_diff12_diff3)

plt.show()



#%%
print(CHsubSl2_diff12)

#%%
train_data = CHsubSl2_diff12.loc[:"2020-12-01"]
test_data= CHsubSl2_diff12.loc["2021-01-01":"2022-12-01"]

print(train_data)
# SARIMA Model
#%%
#(1,0,1)x(5-7,1,1)_12
#(4,0,1)x(5-7,1,1)_12

#%%
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train_data.values, order=(1,0,1),seasonal_order = (6,1,1,12))
model_fit = model.fit()
print(model_fit.summary())

#%%
from statsmodels.tsa.arima.model import ARIMA
model2 = ARIMA(train_data.values, order=(0,0,0),seasonal_order = (7,1,1,12))
model_fit2 = model2.fit()
print(model_fit2.summary())

#%%
# Residual plot
residuals = pd.DataFrame(model_fit2.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


shapiro_test = stats.shapiro(residuals)
shapiro_test

#%%
acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)

plt.show()

#%%

test_data['forecast']=model_fit.predict(start=0,end=23,dynamic=False)
test_data[['precip','forecast']].plot(figsize=(12,8))


from statsmodels.graphics.tsaplots import plot_predict
ax = CHsubSl2_diff12.iloc[400:].plot(ax=ax)
plot_predict(model_fit2,'2021-01-01','2022-12-01', ax = ax)

#%%
pred = model_fit2.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = CHsubSl2_diff12.loc['2021-01-01':].plot(label='observed')

#%%
from statsmodels.tsa.arima.model import get_forecast



#CHsub.fillna(0)
'''
per_25 = CHsub.fillna(0).precip.quantile(0, dim = 'time')
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

map_25 = plt.pcolormesh(per_25.longitude,per_25.latitude,per_25,\
                        vmin=0, vmax=25, cmap='PuBuGn')

ax.set_title('Africa - 25 Percentile Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(map_25,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
#per_0th = CHsub.groupby('time.month').quantile(0, dim = 'time')

plt.show()
'''
# %%
