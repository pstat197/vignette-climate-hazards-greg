#forecasting draft
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
import warnings
import itertools
import matplotlib
import pmdarima as pm
import datetime
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
#%%
hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
# %% 
chirps = xr.open_dataset(data_dir+infile)
#%%
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
congo =  chirps.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#%%
minlat = -17.; maxlat = 5.
minlon = 12; maxlon = 32.
#plot_month2 = mean_seasonal.mean(dim=["longitude", "latitude"]).to_pandas()
#plot_month = seasonal_mean2_all.mean(dim=["longitude", "latitude"]).to_pandas()
congo_mean = congo.mean(dim=["longitude", "latitude"]).to_pandas()
congo_mean.plot()
#%% SARIMA 
congo_reset = congo_mean.reset_index()
ts_congo = congo_reset[['time', 'precip']]
ts_congo['time'] = pd.to_datetime(ts_congo['time'])
ts_congo = ts_congo.dropna()
ts_congo.isnull().sum()

ts_congo = ts_congo.set_index('time')
#%%
sns.set()
#And label the y-axis and x-axis using Matplotlib. We will also rotate the dates 
# on the x-axis so that they’re easier to read:
plt.ylabel('precip')
plt.xlabel('time')
plt.xticks(rotation=45)
#And finally, generate our plot with Matplotlib:

plt.plot(ts_congo.index, ts_congo['precip'], )
# %% test vs train pt. 1
train = ts_congo[ts_congo.index < pd.to_datetime("2020-01-01", format='%Y-%m-%d')]
test = ts_congo[ts_congo.index > pd.to_datetime("2019-12-01", format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('Precipitation')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title("Train/Test split for Precip Data")
plt.show()

# %% test v train pt. 2
y = train['precip']
#To define an ARMA model with the SARIMAX class, we pass in the order parameters of 
# (1, 0 ,1). Alpha corresponds to the significance level of our predictions. 
# Typically, we choose an alpha = 0.05. Here, the ARIMA algorithm calculates upper and 
# lower bounds around the prediction such that there is a 5 percent chance that the 
# real value will be outside of the upper and lower bounds. This means that there is a 
# 95 percent confidence that the real value will be between the upper and lower bounds of 
# our predictions.

ARMAmodel = SARIMAX(y, order = (1, 0, 1))

ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
#%%
#And plot the results:
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.ylabel('Precipitation')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title("Train/Test split for Precip Data")
plt.legend()
plt.show()

# %% rmse to see if prediction is right or on track
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse = np.sqrt(mean_squared_error(test["precip"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

#78.24: surprisingly low, but on graph, just goes down (so not really good at predicting)

# %% random ARIMA model
ARIMAmodel2 = ARIMA(y, order = (2, 2, 2))
ARIMAmodel2 = ARIMAmodel2.fit()

y_pred2 = ARIMAmodel2.get_forecast(len(test.index))
y_pred_df2 = y_pred2.conf_int(alpha = 0.05) 
y_pred_df2["Predictions"] = ARIMAmodel2.predict(start = y_pred_df2.index[0], end = y_pred_df2.index[-1])
y_pred_df2.index = test.index
y_pred_out_2 = y_pred_df2["Predictions"] 
#%% plot along with other lines
#And plot the results:
plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.plot(y_pred_out, color='green', label = 'ARIMA Basic Predictions')
plt.plot(y_pred_out_2, color='yellow', label = 'ARIMA Con Predictions')
plt.ylabel('Precipitation')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title("Train/Test split for Precip Data")
plt.legend()
plt.show()
#%%
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse2 = np.sqrt(mean_squared_error(test["precip"].values, y_pred_df2["Predictions"]))
print("RMSE: ",arma_rmse2)
#54.749591657276646: better but still could do better
# %% OTHER METHOD: SARIMA 
# model configs
#%%
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#%% diff 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# %%
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0,1,2),
                                seasonal_order=(1,1,2, 12))

results = mod.fit(method = 'powell')
print(results.summary().tables[1])

#%%
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1,0,0),
                                seasonal_order=(2,1,0, 12))

results = mod.fit(method = 'powell')
print(results.summary().tables[1])
# %%
#(1,0,0)*(2,1,0, 12)
#%% alex sarima code
#%%multiclass - week 8 in R???? 
#one step - do multistep (recursion)
SARIMAXmodel = SARIMAX(y, order = (1,0,0), seasonal_order=(2,1,0,12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred4 = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df4 = y_pred4.conf_int(alpha = 0.05) 
y_pred_df4["Predictions"] = SARIMAXmodel.predict(start = y_pred_df4.index[0], end = y_pred_df4.index[-1])
y_pred_df4.index = test.index
y_pred_out4 = y_pred_df4["Predictions"] 
#%%
SARIMAXmodel = SARIMAX(y, order = (0,1,2), seasonal_order=(1,1,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred3 = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df3 = y_pred3.conf_int(alpha = 0.05) 
y_pred_df3["Predictions"] = SARIMAXmodel.predict(start = y_pred_df3.index[0], end = y_pred_df3.index[-1])
y_pred_df3.index = test.index
y_pred_out3 = y_pred_df3["Predictions"] 
#%% plot along with other lines
#And plot the results:
plt.plot(train.iloc[410:], color = "black")
plt.plot(test, color = "red")
#plt.plot(y_pred_out, color='green', label = 'ARIMA Basic Predictions')
#plt.plot(y_pred_out_2, color='yellow', label = 'ARIMA Con Predictions')
#plt.plot(y_pred_out3, color='Blue', label = 'SARIMA Prediction')
plt.plot(y_pred_out4, color='blue', label = "SARIMA Predictions pt. 2")
plt.ylabel('Precipitation')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title("Train/Test split for Precip Data")
plt.legend()
plt.show()
#ax = ts_s_adj.iloc[400:].plot(label='observed')
#%%
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse4 = np.sqrt(mean_squared_error(test["precip"].values, y_pred_df4["Predictions"]))
print("RMSE: ",arma_rmse4)
#7.9979

# %%
pred_uc = results.get_forecast(steps=150)
pred_ci = pred_uc.conf_int()
ax = ts_congo.iloc[350:].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('precip')
plt.legend()
plt.show()
#%%
import numpy as np
from sklearn.metrics import mean_squared_error

arma_rmse4 = np.sqrt(mean_squared_error(test["precip"].values, y_pred_df4["Predictions"]))
print("RMSE: ",arma_rmse4)
#13.966

# %%
