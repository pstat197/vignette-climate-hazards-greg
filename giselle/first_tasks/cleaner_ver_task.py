#cleaner version
#task: nail down wet vs dry areas based on seasonal precip
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
#%matplotlib inline
import pandas as pd
import pymannkendall as mk

hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
chirps = xr.open_dataset(data_dir+infile)
#%%
PPT_dir = '/home/chc-data-out/products/CHIRPS-2.0/global_monthly/netcdf/'
infile = 'chirps-v2.0.monthly.nc'
chirps = xr.open_dataset(PPT_dir+infile)
#%% africa?
#set spatial subset dimensions
minlat = -40.; maxlat = 40.
minlon = -20; maxlon = 50.
CHsub =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsub.load()
# %% drop first two months to create seasonal distribution
drop_2mon = CHsub.drop([np.datetime64('1981-01-01'), np.datetime64('1981-02-01')], dim='time')
#%% using rolling to see mean per season
season_mean = drop_2mon.rolling(time=3).mean()#.dropna('time')
# %%
month_leng = drop_2mon.time.dt.days_in_month

# %%
# Calculate the weights by grouping by 'time.season'.
weights = (
    month_leng.groupby("time.season") / month_leng.groupby("time.season").sum()
)

# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

# Calculate the weighted average
ds_weighted = (drop_2mon * weights).groupby("time.season").sum(dim="time")
# %%
# only used for comparisons
ds_unweighted = drop_2mon.groupby("time.season").mean("time")
ds_diff = ds_weighted - ds_unweighted
# %%
# Quick plot to show the results
notnull = pd.notnull(ds_unweighted["precip"][0])

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 12))
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):
    ds_weighted["precip"].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 0],
        vmin=0,
        vmax=300,
        cmap="Spectral_r",
        add_colorbar=True,
        extend="both",
    )

    ds_unweighted["precip"].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 1],
        vmin=0,
        vmax=300,
        cmap="Spectral_r",
        add_colorbar=True,
        extend="both",
    )

    ds_diff["precip"].sel(season=season).where(notnull).plot.pcolormesh(
        ax=axes[i, 2],
        vmin=-0.1,
        vmax=0.1,
        cmap="RdBu_r",
        add_colorbar=True,
        extend="both",
    )

    axes[i, 0].set_ylabel(season)
    axes[i, 1].set_ylabel("")
    axes[i, 2].set_ylabel("")

for ax in axes.flat:
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.axis("tight")
    ax.set_xlabel("")

axes[0, 0].set_title("season = DJF")
axes[0, 1].set_title("Equal Weighting")
axes[0, 2].set_title("Difference")

plt.tight_layout()

fig.suptitle("Precipitation by Season", fontsize=16, y=1.02)
# %%
# select DJF
DA_DJF = drop_2mon.sel(time=drop_2mon.time.dt.season=="DJF")

# calculate mean per year
DJF_mean = DA_DJF.groupby(DA_DJF.time.dt.year).mean("time")
# %% plot DJF
minlat = -40.; maxlat = 40.
minlon = -20; maxlon = 50.
#CHsub =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#CHsub.load()
plot_djf = DJF_mean.mean(dim=["longitude", "latitude"]).to_pandas()
plot_djf.plot()
t1= mk.seasonal_test(plot_djf, period = 42)
# %% select mam
DA_MAM = drop_2mon.sel(time=drop_2mon.time.dt.season=="MAM")

# calculate mean per year
MAM_mean = DA_MAM.groupby(DA_MAM.time.dt.year).mean("time")
# %% plot DJF
minlat = -40.; maxlat = 40.
minlon = -20; maxlon = 50.
#CHsub =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#CHsub.load()
plot_mam = MAM_mean.mean(dim=["longitude", "latitude"]).to_pandas()
plot_mam.plot()
t2= mk.seasonal_test(plot_mam, period = 42)

# %%
# select jja
DA_JJA = drop_2mon.sel(time=drop_2mon.time.dt.season=="JJA")

# calculate mean per year
JJA_mean = DA_JJA.groupby(DA_JJA.time.dt.year).mean("time")
# %% plot DJF
minlat = -40.; maxlat = 40.
minlon = -20; maxlon = 50.
plot_jja = JJA_mean.mean(dim=["longitude", "latitude"]).to_pandas()
plot_jja.plot()
t3= mk.seasonal_test(plot_jja, period = 42)
# %% select son
DA_SON = drop_2mon.sel(time=drop_2mon.time.dt.season=="SON")

# calculate mean per year
SON_mean = DA_SON.groupby(DA_SON.time.dt.year).mean("time")
# %% plot DJF
minlat = -40.; maxlat = 40.
minlon = -20; maxlon = 50.
#CHsub =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#CHsub.load()
plot_son = SON_mean.mean(dim=["longitude", "latitude"]).to_pandas()
plot_son.plot()
t4= mk.seasonal_test(plot_son, period = 42)
#%%
from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(#time_series_values, lags = 15) 

# %% 25 only
#AvgPrec = chirps.precip.mean("time")
#AvgPrec25 = AvgPrec.where(AvgPrec.values <25)