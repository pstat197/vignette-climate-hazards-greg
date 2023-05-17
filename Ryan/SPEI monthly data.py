#%%
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
import glob
from scipy.stats import gamma as gamdist 
from scipy.stats import norm
from scipy.stats import fisk, linregress
from scipy.special import comb, gamma
import xarray
import pandas
import rasterio

#from pmdarima.arima import ARIMA
#from pmdarima.arima import auto_arima

#%%
#CHIRPS

data_dir = 'C:/Users/ryanc/Downloads/'
infile = 'chirps-v2.0.monthly.nc'
clim = xr.open_dataset(data_dir+infile)
#%%
# PET Data
file_paths = ['C:/Users/ryanc/Downloads/PETmonthly_01.nc','C:/Users/ryanc/Downloads/PETmonthly_02.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_03.nc','C:/Users/ryanc/Downloads/PETmonthly_04.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_05.nc','C:/Users/ryanc/Downloads/PETmonthly_06.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_07.nc','C:/Users/ryanc/Downloads/PETmonthly_08.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_09.nc','C:/Users/ryanc/Downloads/PETmonthly_10.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_11.nc','C:/Users/ryanc/Downloads/PETmonthly_12.nc']
PET_monthly = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date')
PET_monthly = PET_monthly.sortby('date')

print(PET_monthly)

#%% Precip_2_SPI function
def Precip_2_SPI(ints, MIN_POSOBS=12, NORM_THRESH=160.0):
    '''This is a recoding of the IDL version of the program to python.  Many of the variables
    have remained the same.  

    
    ints = input timeseries of rainfall, required to be 1-d
    MIN_POSOBS = minimum number of positive observations required to do the calculation
    '''
    
    ts = np.reshape(ints,len(ints))
    
    p_norain = np.sum(ts == 0.00) / len(ts)
    
    poslocs = np.where(ts > 0.000)
    posvals = ts[poslocs]
    
    if len(poslocs[0]) < MIN_POSOBS:
        return np.zeros(len(ints))
    else:
        a1, loc1, b1 = gamdist.fit(posvals,floc=0.0)
        xi = np.zeros(len(posvals))
        
        if a1 <= NORM_THRESH:
            xi = gamdist.cdf(posvals,a1,loc=loc1,scale=b1)
        else: 
            xi = norm.cdf(posvals,loc=np.mean(posvals),scale=np.std(posvals))
        
        pxi = np.zeros(len(ts))
        pxi[poslocs] = xi
        prob = p_norain + ((1.0 - p_norain) * pxi)
        
        if p_norain > 0.5:
            for i in np.argwhere(ts < 7):
                prob[i] = 0.5
        
        if np.sum(prob >= 1.0000) > 0:
            prob[np.where(prob >= 1.000)] = 0.9999999
        
        return norm.ppf(prob)

#%% Get log-logistic distribution parameters calculated manuall
# scipy.stats.fisk doesn't seem to want to get these parameters
def GET_LLD_PARS(input_D):
  zdim = np.isfinite(input_D).sum()
  if zdim > 10:
    tmpD = input_D[np.where(np.isfinite(input_D))]
    n = np.product(tmpD.shape)
    tmpD = np.reshape(tmpD[tmpD.sort()], np.product(tmpD.shape))
    UPWMS = np.zeros(3)  # unbiased probability weighted moments
    for i in range(3):
      FactArr = np.zeros(n)
      for j in range(n):
        FactArr[j] = tmpD[j] * comb(n-(j+1),i) / comb(n-1,i)
      UPWMS[i] = FactArr.sum() / n.astype('float64')
      
    lld_pars = np.zeros(3)
    lld_pars[0] = ((2.0*UPWMS[1])-UPWMS[0]) / ((6.0*UPWMS[1]) - UPWMS[0] - (6.0*UPWMS[2]))
    lld_pars[1] = ((UPWMS[0] - (2.0*UPWMS[1])) * lld_pars[0]) / (gamma(1.0 + (1.0/lld_pars[0])) \
            * (gamma(1.0 - (1.0 / lld_pars[0]))))
    lld_pars[2] = UPWMS[0] - (lld_pars[1] * gamma(1.0 + (1.0 / lld_pars[0])) \
            * gamma(1.0 - (1.0 / lld_pars[0])))
    
    return lld_pars
  else:
    return np.zeros(3) * np.nan

#%% Get the SPEI values from input D
def GET_SPEI_FROM_D(in_D):
  in_D = np.reshape(in_D,np.product(in_D.shape))
  pars = GET_LLD_PARS(in_D)
  if np.isfinite(pars).sum() == 3:
    Fx = 1.0 / (1.0 + ((pars[1] / (in_D - pars[2])) ** pars[0]))
    Zs = norm.ppf(Fx)
    return Zs
  else: 
    return np.zeros(len(in_D)) * np.nan


#%%

PET_monthly = PET_monthly.PET[:, :,0:504]
print(PET_monthly)
#%%
minlat = -23; maxlat = -15.
minlon = 24; maxlon = 34.

months = [1,2,3,4,5,6,7,8,9,10,11,12]
CHsub =  clim.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsub.load()
print(CHsub)
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window



#%%

PET_data = PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
PET_data.load()

#%%
for x in months:
   
  PET_month = PET_data.sel(date=PET_monthly['date.month']== x)
  PET_timeseries = PET_month.mean(dim = ['lats', 'lons']).to_pandas()
  #PET_timeseries = PET_timeseries[0:504]
  #print(PET_timeseries.values)
  
  PET_timeseries = PET_timeseries.dropna()
  PET_array = PET_timeseries.values
  #print(len(PET_array))

  #PET_timeseries.plot()
  #plt.show()

  SA_PPT = clim.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
  SA_PPT.load()

  SA_month = SA_PPT.sel(time=clim['time.month']== x)
  SA_timeSeries = SA_month.mean(dim = ['longitude', 'latitude']).to_pandas()
  #print(SA_timeSeries)

  SA_timeSeries = SA_timeSeries.dropna()
  SA_array = SA_timeSeries['precip'].values


  D = (SA_array-PET_array)
  print(D)


  pd.DataFrame(GET_SPEI_FROM_D(D)).plot(legend=None)
  plt.title("Month " + str(x) + " - SPEI")
  plt.xlabel("Time")
  plt.ylabel("SPEI")
  plt.savefig("month-" + str(x) + ".png")


# %%
print(D)



# Linear Regression
#%%
minlat = -23; maxlat = -15.
minlon = 24; maxlon = 34.

moi = 5 # month of interest 1=Jan, 2=Feb, ... 12=Dec
PPTsub = clim.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=clim['time.month'] == moi)
PETsub = PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET_monthly['date.month'] == moi)

#%%
#PETsub.PET[:,:,0].plot.pcolormesh()

print(PETsub)

#%%
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')


#%%
subNY,subNX = tmpPET[:,:,0].shape

#%%
tmplats = np.arange(subNY+1)*0.05 + minlat #make array of latitude edges for pcolormesh
tmplons = np.arange(subNX+1)*0.05 + minlon #make array of longitude edges for pcolormesh

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])



#%%
print(nvals)
#%%
#tic = time.time()
for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
  
#toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation')    


#print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    

#%%
print(gvals)
len(gvals)
# %%
colors = ['']
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')


projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='BrBG')

ax.set_title('Map of Correlation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

# %%
type(PETsub.lons)
# %%
for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
  plt.plot(x, y, 'o', label='original data')
  plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
  plt.legend()
  plt.show()

# %%
plt.plot(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:],'o')
  m, b = np.polyfit(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:], 1)
  plt.plot(x, m*x+b)





