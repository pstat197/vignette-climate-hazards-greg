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
import matplotlib as mpl
import matplotlib.pyplot as plt
from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
import pandas as pd
import time
from datetime import date
from scipy.stats import spearmanr, pearsonr, mode, linregress

count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

#%% read in climatology
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
data_dir = '/home/chc-data-out/products/CHPclim/netcdf/'
infile = 'chpclim.5050.monthly.nc'

clim = xr.open_dataset(data_dir+infile)

#make a quick visual of the january average rainfall
clim.precip[0,:,:].plot.pcolormesh()

#%% make a better map of the rainfall

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

janmap = plt.pcolormesh(clim.longitude,clim.latitude,clim.precip[0,:,:],\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('January Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#plt.savefig('testmap.png',dpi=200)

#%% find the index of the maximum value
maxind = clim.precip.argmax(dim="time")

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

monmap = plt.pcolormesh(clim.longitude,clim.latitude,maxind,\
                        vmin=-0.5, vmax=11.5, cmap='hsv')

ax.set_title('Month of Maximum Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='white',zorder=100) #color the oceans
cb = plt.colorbar(monmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal',\
                  ticks=[0,1,2,3,4,5,6,7,8,9,10,11],) #add colorbar
cb.ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

## try with discrete colorbar
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

cmap = plt.cm.get_cmap('gist_rainbow')
clevels = np.arange(-0.5,12.,1.0)
monmap = plt.pcolormesh(clim.longitude,clim.latitude,maxind,\
                        vmin=clevels[0], vmax=11.5, cmap=cmap,\
                        norm=mpl.colors.BoundaryNorm(clevels,ncolors=cmap.N))

ax.set_title('Month of Maximum Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='white',zorder=100) #color the oceans
cb = plt.colorbar(monmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal',\
                  ticks=[0,1,2,3,4,5,6,7,8,9,10,11],) #add colorbar
cb.ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])


#%% load in monthly data
data_dir = '/home/chc-data-out/products/CHIRPS-2.0/global_monthly/netcdf/'
infile = 'chirps-v2.0.monthly.nc'

CHIRPSp = xr.open_dataset(data_dir+infile)

#set spatial subset dimensions
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
CHsub = CHIRPSp.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsub.load()

CHsub.precip[0,:,:].plot.pcolormesh()

#monthly average of the data
climatology = CHsub.groupby('time.month').mean('time')  
#each month's difference from the monthly average
anoms = CHsub.groupby('time.month') - climatology  

AOIts = CHsub.precip.sel(latitude=slice(minlat+5,maxlat-5),\
                         longitude=slice(minlon+5,maxlon-5)).\
                     mean(dim=['latitude','longitude'])

#monthly 3-month totals of rainfall
CHsub3mo = CHsub.rolling(time=3).sum()
x=200
y=200
CHsub.precip[0:12,y,x].values
CHsub3mo.precip[0:12,y,x].values

#%% make monthly Hobbins netcdf data

for m in range(1,13):
  
  tic = time.time()
  petblock,PixInfo2,ProjInfo2 = FILL_HOBBINS_DEKAD_GLOBAL((m-1)*3+1,m*3)
  toc = time.time()
  print('{:10.2f} sec elapsed for PET read-in'.format(toc-tic))
  ssnPET = petblock[800:2800,:,:,:].sum(axis=3)   # trim to match dimensions of rainblock
  print('Size of ssnPET is ',ssnPET.shape)

  subNY,subNX,Nyrs = ssnPET.shape
  ssnPET[ssnPET < 0.0] = np.nan

  lons = np.around(np.arange(\
    PixInfo2[0]+(PixInfo2[1]/2),-1*PixInfo2[0],PixInfo2[1]),decimals=3)
  lats = np.around(np.arange(\
    PixInfo2[3]+(PixInfo2[5]/2),-1*PixInfo2[3],PixInfo2[5]),decimals=3)
    
  strt_d8 = date(1981,m,1)
  d8s = pd.date_range(start=strt_d8, periods=Nyrs,freq='12MS')
  pet_da = xr.DataArray(np.flipud(ssnPET),name='PET',\
                         coords=[np.flipud(lats[800:2800]),lons,d8s],\
                         dims=['lats','lons','date'])
  out_dir = '/home/chc-data-out/people/husak/forCapstone/'
  outfile = out_dir+'PETmonthly_{:02d}.nc'.format(m)
  pet_da.to_netcdf(outfile)
  
#%% correlations between PPT and PET

# read in CHIRPS  
PPT_dir = '/home/chc-data-out/products/CHIRPS-2.0/global_monthly/netcdf/'
infile = 'chirps-v2.0.monthly.nc'
PPT = xr.open_dataset(PPT_dir+infile)

# read in Hobbins
PET_dir = '/home/chc-data-out/people/husak/forCapstone/'
PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date')
#this didn't work with the sortby('date') 
#PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date').sortby('date')

#set spatial subset dimensions and specific month
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
moi = 2 # month of interest 1=Jan, 2=Feb, ... 12=Dec

PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'] == moi)
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'] == moi).sortby('date')
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')

#if the numbers aren't the same, then you might have to remove the final year from
# the PPT because it may have been updated
# tmpPPT = tmpPPT[:-1,:,:]

subNY,subNX,nyrs = tmpPET.shape
years = 1981 + np.arange(nyrs)

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

tic = time.time()
for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    

# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
# ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab20')

ax.set_title('Map of Correlation Month = {:02d}'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
plt.savefig('testmap_{:02d}.png'.format(moi),dpi=200)

#%% Identify and map the ROI box
roi_minlat = -23.; roi_maxlat = -14.
roi_minlon = 25.; roi_maxlon = 35.

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='coolwarm')

ax.set_title('Map of Correlation Month = {:02d}'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#now draw the box of interest
boi1 = ax.plot([roi_minlon,roi_minlon,roi_maxlon,roi_maxlon,roi_minlon],\
               [roi_minlat,roi_maxlat,roi_maxlat,roi_minlat,roi_minlat],'k')

plt.savefig('testmap_{:02d}.png'.format(moi),dpi=200,bbox_inches='tight')

#%% GET REGIONAL AVERAGE RAINFALL AND CORRELATE WITH PIXEL-LEVEL SSTS
ppt_ts = PPTsub.precip.sel(latitude=slice(-23.,-14.),longitude=slice(25.,35.)).mean(axis=(1,2)).values
pet_ts = PETsub.PET.sel(lats=slice(-23.,-14.),lons=slice(25.,35.)).mean(axis=(0,1)).values

print(ppt_ts.shape,pet_ts.shape,'these should have the same numbers, but out of order')

#if the numbers aren't the same, then you might have to remove the final year from
# the PPT because it may have been updated
# ppt_ts = ppt_ts[:-1]

ppt_norm = (ppt_ts - ppt_ts.mean()) / ppt_ts.std()
pet_norm = (pet_ts - pet_ts.mean()) / pet_ts.std()

regvals = linregress(ppt_ts,pet_ts)
pet_est = regvals[1] + (regvals[0] * ppt_ts)

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_axes([0.1,0.1,0.85,0.80])
tsplot = ax1.plot(years,pet_ts,alpha=0.5,label='Hobbins PET')
tsplot = ax1.plot(years,ppt_ts,alpha=0.5,label='CHIRPS Precip')
plt.ylabel('Millimeters')
plt.xlabel('Year')
plt.title('Timeseries of OND Precip and PET')
ax1.legend(loc='best')
#plt.savefig('/home/sandbox-people/husak/PythonScripts/TempDelete.png',dpi=200)

fig = plt.figure(figsize = (6,6))
ax2 = fig.add_axes([0.15,0.1,0.80,0.80])
tsscat = ax2.plot(ppt_ts,pet_ts,'o',lw=0)
tsline = ax2.plot([ppt_ts.min(),ppt_ts.max()],[pet_est.max(),pet_est.min()],'--',\
                  label='r^2={:0.6f}'.format(regvals[2]**2))
plt.ylabel('PET (mm)')
plt.xlabel('CHIRPS (mm)')
plt.title('Scatterplot of Precip and PET')
ax2.legend(loc='best')

fig = plt.figure(figsize = (8,6))
ax3 = fig.add_axes([0.1,0.1,0.85,0.80])
tsplot = ax3.plot(years,pet_norm,alpha=0.5,label='Normalized PET')
tsplot = ax3.plot(years,ppt_norm,alpha=0.5,label='Normalized Precip')
ax3.legend(loc='best')

resids = pet_ts - pet_est

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_axes([0.1,0.1,0.85,0.80])
tsplot = ax1.bar(years,resids,alpha=0.5,label='PET Residuals')
plt.ylabel('PET residual (mm)')
plt.xlabel('Year')
plt.title('Timeseries of MAM PET Residuals')
ax1.legend(loc='best')


#%%  put the 4 figures on a single window

fig = plt.figure(figsize = (12,10))
ax1 = plt.subplot(221)
tsplot = ax1.plot(years,pet_ts,alpha=0.5,label='Hobbins PET')
tsplot = ax1.plot(years,ppt_ts,alpha=0.5,label='CHIRPS Precip')
plt.ylabel('Millimeters')
plt.xlabel('Year')
plt.title('Timeseries of OND Precip and PET')
ax1.legend(loc='best')
#plt.savefig('/home/sandbox-people/husak/PythonScripts/TempDelete.png',dpi=200)

ax2 = plt.subplot(222)
tsscat = ax2.plot(ppt_ts,pet_ts,'o',lw=0)
tsline = ax2.plot([ppt_ts.min(),ppt_ts.max()],[pet_est.max(),pet_est.min()],'--',\
                  label='r^2={:0.6f}'.format(regvals[2]**2))
plt.ylabel('PET (mm)')
plt.xlabel('CHIRPS (mm)')
plt.title('Scatterplot of Precip and PET')
ax2.legend(loc='best')

ax3 = plt.subplot(223)
tsplot = ax3.plot(years,pet_norm,alpha=0.5,label='Normalized PET')
tsplot = ax3.plot(years,ppt_norm,alpha=0.5,label='Normalized Precip')
plt.ylabel('Normalized PPT and PET')
ax3.legend(loc='best')

resids = pet_ts - pet_est

ax4 = plt.subplot(224)
tsplot = ax4.bar(years,resids,alpha=0.5,label='PET Residuals')
plt.ylabel('PET residual (mm)')
plt.xlabel('Year')
plt.title('Timeseries of MAM PET Residuals')
ax1.legend(loc='best')

fig.tight_layout()

plt.savefig('/home/sandbox-people/husak/PythonScripts/TempDelete.png',dpi=200)

#%%  link to Sea Surface Temperatures (SSTs)
# file downloaded from https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.html
in_dir = '/home/sandbox-people/husak/PythonScripts/'
infile = in_dir + 'sst.mnmean.nc'

SSTS = xr.open_dataset(infile)

#calculate a rolling 3-month average, and select the average with the endmonth equal to SST_moi
SST_moi = 12 #the final month in the temporal average
#sst3 = SSTS.sst.rolling(time=3).mean().values
sstvals = SSTS.sst.rolling(time=3).mean().sel(time=(SSTS.time['time.month'] == SST_moi)).values

nyrs,NY,NX = sstvals.shape
corr_mat = np.zeros([NY,NX]) * np.nan

tic = time.time()
for x in range(NX):
  for y in range(NY):
    corr_mat[y,x] = linregress(resids[1:],sstvals[1:,y,x])[2]
toc = time.time()
print(toc-tic, ' sec elapsed for correlation calculation')
    
resid_SST_corr = xr.DataArray(corr_mat,coords=[SSTS.lat,SSTS.lon],dims=['lats','lons'])
    
projection = ccrs.PlateCarree(central_longitude=180)
ncolors = 9  # number of colors to display in tshe map
colorcuts = [-0.8,-0.7,-0.6,-0.5,-0.35,0.35,0.5,0.6,0.7,0.8]  # thresholds for colors (ncolors + 1) values in array

fig = plt.figure(figsize = (8,4.5))
ax0 = plt.subplot(111,projection=projection)
tmpgr = ax0.contourf(SSTS.lon,SSTS.lat,corr_mat,colorcuts,\
                     levels=colorcuts,extend='both',transform=ccrs.PlateCarree(),\
                     cmap='coolwarm_r')

plt.title('Correlation w/ Residuals',fontsize=12)
ax0.coastlines(); ax0.add_feature(count_bord,edgecolor='black')
plt.tight_layout()

#cbar = fig.colorbar(tmpgr, cax=fig.add_axes([0.2,0.07,0.6,0.03]), orientation='horizontal',shrink=0.75, aspect=40)
cbar = fig.colorbar(tmpgr, cax=fig.add_axes([0.05,0.04,0.9,0.03]),orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()

plt.savefig('/home/sandbox-people/husak/PythonScripts/PETresid_SST_corr.png',dpi=200)








 