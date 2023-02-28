
#Sierra Leone Time Series
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import seaborn as sns 

data_dir = '/Users/disha/downloads/'
infile = 'chirps-v2.0.monthly.nc'
CHIRPS = xr.open_dataset(data_dir+infile)
#%%
## MAKE TIME SERIES OF SIERRA LEONE
minlat = 7; maxlat = 10
minlon = -13; maxlon = -10
SierraLeoneSlice =  CHIRPS.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
SierraLeoneSlice.load()
SierraLeoneSlice = SierraLeoneSlice.mean(dim=["longitude", "latitude"]).to_pandas()
SierraLeoneSlice.plot()

#%%
## MAKE ACF GRAPH
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(SierraLeoneSlice)

#%%
## MAKE PACF GRAPH
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(SierraLeoneSlice, lags=50)

# %%
## FIND DISTRIBUTION OF DATA
SierraLeoneSlice.hist()
plt.title("Sierra Leone Precip Distribution")
plt.xlabel("precip")
plt.ylabel("Months")

# %%
## RUN A LJUNG-BOX TEST 
from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(SierraLeoneSlice, lags=[12], return_df=True)

'''
p value: 0.0 indicates we reject the null and residuals are not independently distributed and exhibit a serial 
correlation; time series does contain an autocorrelation; need to difference the data
'''
# %%
## DIFFERENCING BY 12

diff12 = SierraLeoneSlice.diff(12).dropna()
diff12.plot()
plot_acf(diff12)
plot_pacf(diff12)

'''
time series looks more like white noise; seasonality reduced
'''
# %%
### DICKEY FULLER TEST
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey Fuller Test:')
dftest = adfuller(diff12, autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)

'''
P val < 0.05 indicates data is stationary
'''
# %%
## ARIMA model
import pmdarima as pm
ARIMA_model = pm.auto_arima(diff12, 
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)
# %%
ARIMA_model.plot_diagnostics(figsize=(15,12))
plt.show()
# %%
## SARIMA Model
# Seasonal - fit stepwise auto-ARIMA
SARIMA_model = pm.auto_arima(diff12, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, 
                         m=12, #12 is the frequncy of the cycle
                         start_P=0, 
                         seasonal=True, #set to seasonal
                         d=None, 
                         D=1, #order of the seasonal differencing
                         trace=False,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
# %%
SARIMA_model.plot_diagnostics(figsize=(15,12))
plt.show()
# %%
def forecast(ARIMA_model, periods=24):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(diff12.index[-1] + pd.DateOffset(months=1), periods = n_periods, freq='MS')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(diff12, color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("ARIMA/SARIMA - Sierra Leone Forecast")
    plt.show()
#%%
forecast(SARIMA_model)


# %%
