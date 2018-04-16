# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:56:08 2016

@author: Zhaoyi.Shen
"""

import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy import interpolate

RADIUS = 6371.0e3        
    
def calc_powerspectrum(a,**kwargs):
    nfft_time = np.shape(a)[0]
    if 'nfft_time' in kwargs.keys():
        nfft_time = kwargs['nfft_time']
    nlon = np.shape(a)[1]
    fa = fft.fft(a,axis=1)
    nomega = nfft_time/2+1
    nk = nlon/2+1 
    cfa = np.real(fa[:,:nk])
    sfa = -np.imag(fa[:,:nk])
    pp = np.zeros([nomega,nk])
    pn = np.zeros([nomega,nk])
    for i in range(nlon/2):
        omega, pca = signal.welch(cfa[:,i],nperseg=nfft_time)
        omega, psa = signal.welch(sfa[:,i],nperseg=nfft_time)
        omega, pcasa = signal.csd(cfa[:,i],sfa[:,i],nperseg=nfft_time)
        pp[:,i] = pca+psa-2*np.imag(pcasa)
        pn[:,i] = pca+psa+2*np.imag(pcasa)
    return pp,pn,omega