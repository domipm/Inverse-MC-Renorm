import numpy as np
import math
import pickle
import gzip

import matplotlib.pyplot as plt


def autocorrelation(data, normalized=True):
    """
    Compute autocorrelation using FFT
    """
    nobs = len(data)
    corr_data = data - data.mean()
    n = 2**int(math.log(nobs, 2))
    corr_data = corr_data[:n]
    Frf = np.fft.fft(corr_data)
    acf = np.fft.ifft(Frf * np.conjugate(Frf))/corr_data.shape[0]
    if normalized:
        acf /= acf[0]
    acf = np.real(acf)
    # only return half of the ACF 
    # (see 4.3.1 "Kreuzkorrelationsfunktion" 
    # of https://github.com/arnolda/padc)
    return acf[:int(corr_data.shape[0]/2)]


def calc_error(data):
    """
    Error estimation for time series of simulation observables and take into
    account that these series are correlated (which
    enhances the estimated statistical error).
    """
    # calculate the normalized autocorrelation function of data
    acf = autocorrelation(data)
    # calculate the integrated correlation time tau_int
    # (Janke, Wolfhard. "Statistical analysis of simulations: Data correlations
    # and error estimation." Quantum Simulations of Complex Many-Body Systems:
    # From Theory to Algorithms 10 (2002): 423-445.)
    tau_int = 0.5
    for i in range(len(acf)):
        tau_int += acf[i]
        if i >= 6 * tau_int:
            break
    # mean value of the time series
    data_mean = np.mean(data)
    # calculate the so called effective length of the time series N_eff
    if tau_int > 0.5:
        N_eff = len(data) / (2.0 * tau_int)
        # finally the error is sqrt(var(data)/N_eff)
        stat_err = np.sqrt(np.var(data) / N_eff)


    else:
        stat_err = np.sqrt(np.var(data) / len(data))
    return data_mean,stat_err,tau_int



