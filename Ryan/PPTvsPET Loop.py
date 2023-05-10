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
from scipy import stats
import xarray
import pandas
import rasterio
import statistics

#from pmdarima.arima import ARIMA
#from pmdarima.arima import auto_arima

#%%
#CHIRPS

data_dir = 'C:/Users/ryanc/Downloads/'
infile = 'chirps-v2.0.monthly.nc'
clim = xr.open_dataset(data_dir+infile)
PPT_monthly = clim.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})

# PET Data
file_paths = ['C:/Users/ryanc/Downloads/PETmonthly_01.nc','C:/Users/ryanc/Downloads/PETmonthly_02.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_03.nc','C:/Users/ryanc/Downloads/PETmonthly_04.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_05.nc','C:/Users/ryanc/Downloads/PETmonthly_06.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_07.nc','C:/Users/ryanc/Downloads/PETmonthly_08.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_09.nc','C:/Users/ryanc/Downloads/PETmonthly_10.nc',
              'C:/Users/ryanc/Downloads/PETmonthly_11.nc','C:/Users/ryanc/Downloads/PETmonthly_12.nc']
PET_monthly = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date')
PET_monthly = PET_monthly.sortby('date')

#PET_monthly = PET_monthly.PET[:,:,0:504]
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


minlat = -23; maxlat = -22
minlon = 24; maxlon = 25
SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT =  PPT_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT.load()

SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').precip.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').PET.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET_DJF.iloc[:-1] #PET dataset has one more     

"""
PET_PPT = pd.concat([SA_PPT_DJF, SA_PET_DJF],keys = ['PPT','PET'], axis=1) #concat both dataset
seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, line_kws={'color': 'red'}, ci=None)
corr = np.corrcoef(SA_PPT_DJF, SA_PET_DJF)[0, 1]
rsq = corr**2

# Show the plot
plt.show()



resid_df = PET_PPT['PPT']-PET_PPT['PET']
resid_df.plot()


PET_PPT['indate']= np.where((PET_PPT.index >= '2004-12-31'), 'After 2004', 'Before 2004')
g=seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, ci=None, hue = 'indate', fit_reg=False)
seaborn.regplot(x="PPT", y="PET", data=PET_PPT, scatter=False, ax=g.axes[0, 0],ci = None)
plt.show()


minlat = -3.5; maxlat = -2
minlon = 10; maxlon = 13.5
SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT.load()

SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([9, 10, 11])).resample(date='1Y').mean(dim='date').precip.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([9, 10, 11])).resample(date='1Y').mean(dim='date').PET.mean(dim=('lons', 'lats')).to_pandas()

PET_PPT = pd.concat([SA_PPT_DJF, SA_PET_DJF],keys =['PPT','PET'], axis=1) #concat both dataset
seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, line_kws={'color': 'red'}, ci=None)
corr = np.corrcoef(SA_PPT_DJF, SA_PET_DJF)[0, 1]
rsq = corr**2

# Show the plot
plt.show()



resid_df = PET_PPT['PPT']-PET_PPT['PET']
resid_df.plot()


PET_PPT['indate']= np.where((PET_PPT.index >= '1992-12-31') & (PET_PPT.index <= '2010-12-31'), 'Between 1992 and 2010', 'Outisde 1992 and 2010')
g=seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, ci=None, hue = 'indate', fit_reg=False)
seaborn.regplot(x="PPT", y="PET", data=PET_PPT, scatter=False, ax=g.axes[0, 0],ci = None)
plt.show()


D = (SA_PPT_DJF.array-SA_PET_DJF.array).to_numpy()

pd.DataFrame(Precip_2_SPI(SA_PPT_DJF.array)).plot()
pd.DataFrame(GET_SPEI_FROM_D(D)).plot()



"""
SA_PPT_DJF_pre = SA_PPT.sel(date=SA_PPT['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date')
SA_PET_DJF_pre = SA_PET.sel(date=SA_PET['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date')


SA_PET_DJF_pre = SA_PET_DJF_pre.drop('2023-12-31', dim = 'date')
SA_PET_DJF_pre= SA_PET_DJF_pre.reindex(lats=SA_PPT_DJF_pre.lats,
                                       lons=SA_PPT_DJF_pre.lons,
                                       method = 'nearest')

lons = SA_PPT_DJF_pre['lons'].values
lats = SA_PPT_DJF_pre['lats'].values

# Initialize an empty array to store the r^2 values
r_squared = np.zeros((len(lats), len(lons)))

# Loop through each longitude and latitude value and calculate the r^2 value
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # Get the PPT and PET data for the current longitude and latitude value
        ppt = SA_PPT_DJF_pre.sel(lons=lon, lats=lat).precip.values
        pet = SA_PET_DJF_pre.sel(lons=lon, lats=lat).PET.values
        slope, intercept, r, p, se = stats.linregress(ppt,pet)
        r_squared[i,j] = r**2
    
# %%

for i in range(19,20):
  minlat = -22; maxlat = -20
  minlon = i; maxlon = minlon+1
  SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
  SA_PPT =  PPT_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
  SA_PPT.load()

  SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').precip.mean(dim=('lons', 'lats')).to_pandas()
  SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').PET.mean(dim=('lons', 'lats')).to_pandas()
  SA_PET_DJF = SA_PET_DJF.iloc[:-1]

  PET_PPT = pd.concat([SA_PPT_DJF, SA_PET_DJF],keys =['PPT','PET'], axis=1) #concat both dataset
  seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, line_kws={'color': 'red'}, ci=None)
  corr = np.corrcoef(SA_PPT_DJF, SA_PET_DJF)[0, 1]
  rsq = corr**2
  plt.show()

  std_PPT = statistics.stdev(SA_PPT_DJF)
  std_PET = statistics.stdev(SA_PET_DJF)
  PET_range = np.max(SA_PET_DJF) -np.min(SA_PET_DJF)
  PPT_range = np.max(SA_PPT_DJF) - np.min(SA_PPT_DJF)
  range_diff = abs(PET_range - PPT_range)
  print("lat: " + str(minlat) + ' to ' + str(maxlat))
  print("lon: " + str(minlon) + ' to ' + str(maxlon))
  print("PPT STDev: " + str(std_PPT))
  print("PET STDev: " + str(std_PET))
  print("Range Diff: " + str(range_diff))
  print("R^2: " + str(rsq))

  keep = pd.DataFrame(columns = ['minlat','maxlat','minlon','maxlon','rsq','std_PPT','std_PET','range_diff'])
  if((rsq > .7) and  range_diff <= 100):
    list = [minlat,maxlat,minlon,maxlon,rsq,std_PPT,std_PET,range_diff]

    keep = keep.append(list,ignore_index=True)
    print('MEETS CRITERIA')

  else:
    print("Doesn't meet criteria")
    print(" ")

#%%
print(keep)

#plot of the area we are mapping
# %%
moi = 11 # month of interest 1=Jan, 2=Feb, ... 12=Dec
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
#ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window
tmpmap = ax.pcolormesh(SA_PET.lons,SA_PPT.lats,\
                 vmin=-1, vmax=1, cmap='tab20b') #cmap='tab20'

#PETsub.lons,PETsub.lats
ax.set_title('Map of Area" = {:02d}'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
#cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
# %%

# %%
minlat = -23; maxlat = -22
minlon = 24; maxlon = 25
SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT =  PPT_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT.load()

SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').precip.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').PET.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET_DJF.iloc[:-1] #PET dataset has one more     

PET_PPT = pd.concat([SA_PPT_DJF, SA_PET_DJF],keys = ['PPT','PET'], axis=1) #concat both dataset
seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, line_kws={'color': 'red'}, ci=None)
corr = np.corrcoef(SA_PPT_DJF, SA_PET_DJF)[0, 1]
rsq = corr**2

# Show the plot
plt.show()
# %%
