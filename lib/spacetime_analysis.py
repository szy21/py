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

def calc_phasespeed_powerspectrum(a,**kwargs):
    if ('freq_time') in kwargs.keys():
        freq_time = kwargs['freq_time']
    nfft_time = np.shape(a)[0]
    if 'nfft_time' in kwargs.keys():
        nfft_time = kwargs['nfft_time']
    ncp = 20
    if 'ncp' in kwargs.keys():
        ncp = kwargs['ncp']
    pp,pn,omega = calc_powerspectrum(a,nfft_time=nfft_time)
    nomega = np.shape(pp)[0]
    nk = np.shape(pp)[1]
    nlon = np.shape(a)[1]
    k = np.linspace(0,0.5,nlon/2+1)
    cpmin = omega[1]/k[-1]
    cpmax = omega[-1]/k[1]
    cp = np.linspace(cpmin,cpmax,ncp)
    pcpp = np.zeros(ncp)
    pcpn = np.zeros(ncp)
    for i in range(1,nk):
        pcpp = pcpp+i*np.interp(cp*k[i],omega,pp[:,i])
        pcpn = pcpn+i*np.interp(cp*k[i],omega,pn[:,i])
    return pcpp,pcpn,cp

def calc_phasespeed_cospectrum(a,b,**kwargs):
    nfft_time = np.shape(a)[0]
    if 'nfft_time' in kwargs.keys():
        nfft_time = kwargs['nfft_time']
    int_time = 86400
    if 'int_time' in kwargs.keys():
        int_time = kwargs['int_time']
    ki = 4
    if 'nwave' in kwargs.keys():
        ki = kwargs['nwave']
    ncp = 10
    if 'ncp' in kwargs.keys():
        ncp = kwargs['ncp']
    p_all,omega_all = calc_cospectrum(a,b,nfft_time=nfft_time)
    nomega = np.shape(p_all)[0]
    nk = np.shape(p_all)[1]
    nlon = np.shape(a)[1]
    k = np.linspace(0,0.5,nlon/2+1)
    dk = k[1]-k[0]
    cpmin = omega_all[0]/k[ki]
    cpmax = -cpmin
    cp_all = np.linspace(cpmin/2.,cpmax,ncp)
    pcp_all = np.zeros(ncp)
    for i in range(1,nk):
        pcp_all = pcp_all+k[i]*np.interp(cp_all*k[i],omega_all,p_all[:,i])*dk
    sf_cp = (np.pi*RADIUS/ki)/int_time/cp_all[-1]
    cp_all = cp_all*sf_cp
    pcp_all = pcp_all/sf_cp
    return pcp_all,cp_all
        
    
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

def calc_cospectrum(a,b,**kwargs):
    nfft_time = np.shape(a)[0]
    if 'nfft_time' in kwargs.keys():
        nfft_time = kwargs['nfft_time']
    nlon = np.shape(a)[1]
    fa = fft.fft(a,axis=1)
    fb = fft.fft(b,axis=1)
    nomega = nfft_time/2+1
    nk = nlon/2+1 
    cfa = np.real(fa[:,:nk])
    sfa = np.imag(fa[:,:nk])
    cfb = np.real(fb[:,:nk])
    sfb = np.imag(fb[:,:nk])
    pp = np.zeros([nomega,nk])
    pn = np.zeros([nomega,nk])
    for i in range(nk):
        omega, pcacb = signal.csd(cfa[:,i],cfb[:,i],nperseg=nfft_time)
        omega, psasb = signal.csd(sfa[:,i],sfb[:,i],nperseg=nfft_time)
        omega, pcasb = signal.csd(cfa[:,i],sfb[:,i],nperseg=nfft_time)
        omega, psacb = signal.csd(sfa[:,i],cfb[:,i],nperseg=nfft_time)
        pp[:,i] = np.real(pcacb)+np.real(psasb)+np.imag(pcasb)-np.imag(psacb)
        pn[:,i] = np.real(pcacb)+np.real(psasb)-np.imag(pcasb)+np.imag(psacb)
    p_all = np.zeros([nomega*2,nk])
    p_all[:nomega,:] = np.flipud(pn)
    p_all[nomega:,:] = pp
    sigma = 0.25/np.pi*nomega
    x = np.linspace(-nomega/2,nomega/2,nomega)
    gauss = np.exp(-x**2/(2*sigma**2))
    gauss = gauss/np.sum(gauss)
    for i in range(nk):
        p_all[:,i] = np.convolve(p_all[:,i],gauss,mode='same')
    omega_all = np.concatenate((np.flipud(-omega),omega))
    return p_all,omega_all

def normalize(ntime,nlon,ncp,nwave,int_time,nfft_time):
    x = np.linspace(0, np.pi*2, nlon)
    t = np.arange(0, ntime)
    u = np.zeros([ntime,nlon])
    for i, i_x in enumerate(x):
        for j, j_t in enumerate(t):
            u[j,i] = np.sin(5*i_x - j_t/7.*2*np.pi)
    v = u.copy()
    pcp, cp = calc_phasespeed_cospectrum(
    u, v, ncp=ncp,nwave=nwave,int_time=int_time,nfft_time=nfft_time)
    sf = 0.5/(np.sum(pcp)*(cp[1]-cp[0]))
    return sf