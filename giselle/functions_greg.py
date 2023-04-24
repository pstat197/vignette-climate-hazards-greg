#functions_greg
#%%
import numpy as np
import matplotlib.pyplot as pp
import glob
from scipy.stats import gamma as gamdist
from scipy.stats import norm
from scipy.stats import fisk
from scipy.special import comb, gamma
import xarray
import pandas

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