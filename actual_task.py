#actual_task
#task - threshold/classification

#%% read in climatology
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

hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
#"C:\Users\gizra\Downloads\chpclim.5050.monthly.nc"
#"C:\Users\gizra\OneDrive\Documents\netcdf\chirps-v2.0.monthly (1).nc"
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'#'/home/chc-data-out/products/CHPclim/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'

#notes:
#moving metric of variablity (50 yr: increasing)

chirps = xr.open_dataset(data_dir+infile)

# %%
chirps['precip'].shape
#(504, 2000, 7200)
# %%
#reshape at lowest level
#arr = chirps['precip'].values
#arr.shape 
#doesn't work since file is too big lol

# %%
#import netCDF4
#from matplotlib import pyplot as plt
#from xrspatial.classify import natural_breaks
#%matplotlib inline
#plt.rcParams['figure.figsize'] = (8,5)
 
#%%
#step one: get southern hemisphere
#set spatial subset dimensions
minlat = -90.; maxlat = 0. #actually 90 but goes up to 50 here
minlon = -180; maxlon = 180.
s_hemi = chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#s_hemi.load()
#%% split data by season
#da_jja_only = ds.sel(time=ds.time.dt.month.isin([6, 7, 8])) June, July, August
da_jja_only = s_hemi.sel(time=s_hemi.time.dt.month.isin([6,7,8])) #june, july, aug
da_son_only = s_hemi.sel(time=s_hemi.time.dt.month.isin([9,10,11])) #sep, oct, nov
da_djf_only = s_hemi.sel(time=s_hemi.time.dt.month.isin([12,1,2])) #dec, jan, feb
da_mam_only = s_hemi.sel(time=s_hemi.time.dt.month.isin([3,4,5])) #mar, april, may
#%%
#trying to drop first two times from djf_only
#ds_1 = ds.drop([np.datetime64('2020-01-30')], dim='time')
drop_jan_feb = da_djf_only.drop([np.datetime64('1981-01-01'), np.datetime64('1981-02-01')], dim='time')
#drop_feb = drop_jan.drop([np.datetime64('1981-02-01')], dim='time')
#code below takes around 5 min to run each!

#%%find average per year
clim_yr_jja = da_jja_only.groupby('time.year').mean('time') #map?
#%%
'''
CHsubSl2 = CHsubSl.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubSl2.plot()
t2= mk.seasonal_test(CHsubSl2, period = 12)
'''
#%%find anamolies
anoms_jja = da_jja_only.groupby('time.year') - clim_yr_jja
#%% get time series per year
minlat = -90.; maxlat = 0. #actually 90 but goes up to 50 here
minlon = -180; maxlon = 180.
s_winter = da_jja_only.precip.sel(latitude=slice(minlat+5,maxlat-5), longitude=slice(minlon+5,maxlon-5)).\
        mean(dim=['latitude', 'longitude'])

#%% do rolling for sum of precip for june, july, aug
jja_sum = da_jja_only.rolling(time=3).mean()#.dropna('time')
#%%
avg25less = jja_sum.precip.where(jja_sum.values<25)
#jja_sum.precip.plot()
#%% map?: rerun ABOVE!!!
x=200
y=200
print(jja_sum.precip[0:4,y,x].values)
#print(.precip[0:12,y,x].values) 

#%% rolling for sum of precip for sep, oct, nov
son_sum = da_son_only.rolling(time=3).mean().dropna('time')
#%% rolling for sum of precip for dec, jan, feb
djf_sum = drop_feb.rolling(time=3).mean().dropna('time')
#%% rolling for sum of precip for mar, april, may
mam_sum = da_mam_only.rolling(time=3).mean().dropna('time')
# %%
#step 2: divide by season or 4 months (4 months sounds easier)
test1 = s_hemi.rolling(time=3).mean().dropna('time')

#%%
#step 2b: divide by 4 months and add up precip
precip_4 = s_hemi.rolling(time=4).sum().dropna('time')
# %%
#step 3: map it
#%% testing if less than 25mm
AvgPrec = chirps.precip.mean("time")
sumprec25 = precip_4.where(precip_4.values<25)

#%%
# each month’s difference from the monthly average
season_s = s_hemi.groupby("time.season") #- climatology
#arr.groupby("date.season")
#%%

sum_season = s_hemi.groupby("time.season").sum("time")

#%%
# each month’s difference from sum
diff_s = s_hemi.groupby("time.season") - sum_season

#3: get the time series of a wide-area average for each month for the whole 42 years.  In this case, I have taken the center area of my southern Africa window, buffering by 5-degrees on all sides. 

# Area of Interest (AOI) average over lat/long for whole time series
AOIts = s_hemi.precip.sel(latitude=slice(minlat+5,maxlat-5), (longitude=slice(minlon+5,maxlon-5)), \
                     mean(dim=['latitude','longitude']))

4 - create 3-month totals, and then print the first 12 values (Jan-Dec, 1981) for a single point to convince yourself that it has made the moving sum.  CHsub3mo.precip[2,y,x] should equal the sum of CHsub.precip[0:2,y,x].

# monthly 3-month totals of rainfall
CHsub3mo = CHsub.rolling(time=3).sum()
x=200
y=200
print(CHsub.precip[0:12,y,x].values)
print(CHsub3mo.precip[0:12,y,x].values) 

#%%
#find the index of the maximum value
maxind = precip_4.precip.argmax(dim="time")
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window
monmap = plt.pcolormesh(s_hemi.longitude,s_hemi.latitude,maxind,\
                        vmin=-0.5, vmax=9.5, cmap='hsv')
ax.set_title('3 Month Period of Maximum Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='white',zorder=100) #color the oceans
cb = plt.colorbar(monmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal',\
                  ticks=[0,1,2,3,4,5,6,7,8,9],) #add colorbar
cb.ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O'])
