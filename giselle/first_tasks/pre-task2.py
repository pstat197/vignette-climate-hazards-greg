#work for friday

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#%% read in climatology
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
#"C:\Users\gizra\Downloads\chpclim.5050.monthly.nc"
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'#'/home/chc-data-out/products/CHPclim/netcdf/'
infile = 'chirps-v2.0.monthly.nc (1)'

#notes:
#moving metric of variablity (50 yr: increasing)

chirps = xr.open_dataset(data_dir+infile)

#make a quick visual of the january average rainfall
#chirps.precip[0,:,:].plot.pcolormesh()

#%%
#janmap = plt.pcolormesh(clim.longitude,clim.latitude,clim.precip[0,:,:],\
                        #vmin=0, vmax=400, cmap='PuBuGn')

#janmap = plt.pcolormesh(clim.longitude,clim.latitude,clim.precip[0,:,:],\
                        #vmin=0, vmax=400, cmap='PuBuGn')
# %%
#calculate percentile of rainfall
perprec = chirps.precip.mean("time")
perprec.plot.hist()

#%%
print(chirps['precip'].shape) #(504, 2000, 7200)

# %%
chirps["precip"].head()
#latitude: -49.975 to 49.975
#longitude: -180 to 180 (-179.975 to 179.975)


# %%
print("\n50th Percentile of arr, axis = 0(row) : ",
      np.percentile(chirps["precip"], 50, axis=1))
# %%
#testing percentile
my_array1 = [[[1, 3, 7, 8, 2], [3, 5, 1, 7, 9]], 
[[9, 3, 4, 7, 1], [4, 5, 8, 6, 2]], 
[[2, 3, 5, 6, 1], [5, 4, 7, 8, 9]]]
print(np.percentile(my_array1, 50, axis=1))

#output: [[2.  4.  4.  7.5 5.5]
#[6.5 4.  6.  6.5 1.5]
#[3.5 3.5 6.  7.  5. ]]
#%%
a = len(my_array1) #to find how many groups
b = len(my_array1[1]) #2: number of groups within [[a], [b]] a+b= 2
c = len(my_array1[1][1]) #5: number of items in each array

#%%
for i in my_array1:
    b1 = len(my_array1[i])
    #k = #
    #c1 = len(my_array1[i][k])

#my_new = my_array1.reshape(3, 2*5)
# %%
#function for 3d array

#scores_d = scores.reshape(x*y,1)

#percentiles_d = [percentileofscore(d[:, i], scores_d[i]) for i in range(x*y)]
#percentiles_d = np.round(np.array(percentiles_d), 2).reshape(x,y)
#print(percentiles_d)

#%%
data = np.array([[[ 1.,  1.,  1.],
[ 1.,  1.,  1.],
[ 1.,  1.,  1.]],
   [[ 2.,  2.,  2.],
    [ 2.,  2.,  2.],
    [ 2.,  2.,  2.]],
   [[ 3.,  3.,  3.],
    [ 3.,  3.,  3.],
    [ 3.,  3.,  3.]]])
# %%
x = 3 #rows
y = 3 #columns? per row
z = 3 #groups
#x*y = items in the group

d = data.reshape(z, x*y) #z
#scores_d = scores.reshape(x*y,1)
# %%
#ds['temp'].sel(lon=65.5, lat=29.5) select coordinates
#30.5595° S, 22.9375° E (south africa)
#latitude, longitude
#s_africa = chirps['precip'].sel('longitude'==30.5595, 'latitude'==22.9375)
#s_africa = chirps.precip.sel(space='latitude')
namibia = chirps.precip[:,12:20,17:28] #get precip
namibia_mean = namibia.mean()
#coefficient of variation, and duration of time when wet (must define on own)
#nail down wet and dry (consider seasons); changes in variability or rainfall
#trends: "areas getting drier" or opp. 
#peak rainfall more variable
#classification: (D/W) does it make sense? (2/3: 
# ideas: always dry (1)? outliers
# )

#$$c(E 12°33'00"--E 19°39'00"/S 17°33'00"--S 27°42'00")
# %%
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_csv('path_to_file/stock.csv')
#df = pd.DataFrame(data, columns = ['ValueDate', 'Price'])

# Set the Date as Index
#df['ValueDate'] = pd.to_datetime(df['ValueDate'])
#df.index = df['ValueDate']
#del df['ValueDate']
#plot1 = chirps.isel('latitude' == [12:20], 'longitude':[12:28])
data = pd.read_csv('')
chirps.precip[12:20,17:28].plot.line("b-^")
namibia.plot(figsize=(12,4))
chirps['precip'].isel(longitude = 17, latitude = [12, 13, 14, 15, 16, 17, 18, 19, 20]).plot.line(x="time")

#df.plot(figsize=(15, 6))
plt.show()
# %%
#inch or more -- 4 months (25mm)
#absolute amount (as shown in slides)
random1 = chirps.precip[0, 12, 17]
random1.head()

#%%

#set spatial subset dimensions
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
CHsub =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsub.load()

CHsub.precip[0,:,:].plot.pcolormesh()


# %%
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
minprec = 0.; maxprec = 25.
CHsub =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))

CHsub.load()

#inch or more -- 4 months (25mm)
# %%
chirps_three = chirps.precip[0:3,:,:]
#test_where = chirps_three.where((precip))

# %%

#
# small country (slicing)
#1. find avg
#ThrMA = clim.rolling(time=3).mean().dropna('time')
#
