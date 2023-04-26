#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 10:21:09 2023
this will be some code to keep up with the capstone project
@author: husak
"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import gamma as gamdist
from scipy.stats import norm
from scipy.stats import fisk
from scipy.special import comb, gamma
import glob
import rasterio


#count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
data_dir = 'C:/Users/xusen/Desktop/'

infile = 'chirps-v2.0.monthly.nc'

clim = xr.open_dataset(data_dir+infile)

#set spatial subset dimensions
minlat = -23; maxlat = -15.
minlon = 24; maxlon = 34.
SA_PPT = clim.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
SA_PPT.load()
print(SA_PPT)
SA_Jan = SA_PPT.sel(time = SA_PPT['time.month'].isin(12)).precip.mean(dim=('longitude','latitude')).to_pandas()
SA_timeSeries = SA_PPT.mean(dim = ['longitude', 'latitude']).to_pandas()
SA_timeSeries = SA_timeSeries.dropna()

SA_Jan = SA_Jan.dropna()
print(SA_Jan)
def Precip_2_SPI(ints, MIN_POSOBS=12, NORM_THRESH=160.0):
    ts = np.reshape(ints, len(ints))

    p_norain = np.sum(ts == 0.00) / len(ts)

    poslocs = np.where(ts > 0.000)
    posvals = ts[poslocs]

    if len(poslocs[0]) < MIN_POSOBS:
        return np.zeros(len(ints))
    else:
        a1, loc1, b1 = gamdist.fit(posvals, floc=0.0)
        xi = np.zeros(len(posvals))

        if a1 <= NORM_THRESH:
            xi = gamdist.cdf(posvals, a1, loc=loc1, scale=b1)
        else:
            xi = norm.cdf(posvals, loc=np.mean(posvals), scale=np.std(posvals))

        pxi = np.zeros(len(ts))
        pxi[poslocs] = xi
        prob = p_norain + ((1.0 - p_norain) * pxi)

        if p_norain > 0.5:
            for i in np.argwhere(ts < 7):
                prob[i] = 0.5

        if np.sum(prob >= 1.0000) > 0:
            prob[np.where(prob >= 1.000)] = 0.9999999

        return norm.ppf(prob)

SA_Jan = SA_Jan.iloc[:].values
spi_jan = Precip_2_SPI(SA_Jan, MIN_POSOBS=12, NORM_THRESH=160.0)
print(spi_jan)

ax = sns.displot(spi_jan)

ax.set(title='Distribution of Dec SPI')

plt.show()
'''
plt.plot(spi_jan)
plt.title('SPI Dec')
plt.show()
'''
'''
CHsubSl2 = CHsub_leone.mean(dim=["longitude", "latitude"]).to_pandas()
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

CHsubSl2_diff12 = CHsubSl2.diff(12).dropna()
CHsubSl2_diff12.plot()
plot_acf(CHsubSl2_diff12)
plot_pacf(CHsubSl2_diff12)


#CHsubSl2_diff12_diff3 = CHsubSl2_diff12.diff(6).dropna()
#CHsubSl2_diff12_diff3.plot()
#plot_acf(CHsubSl2_diff12_diff3)
#plot_pacf(CHsubSl2_diff12_diff3)

plt.show()







#CHsub.fillna(0)
'''
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