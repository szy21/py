#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:41:50 2018

@author: Zhaoyi.Shen
"""
import sys
sys.path.append('/home/z1s/py/lib/')
from signal_processing import peigs, csvd
from lanczos_filter import lanczos_filter
from scipy.signal import butter, filtfilt, lfilter
import numpy as np
from matplotlib import pyplot as plt

#%%
dt = 30
n = 7*24*60/dt
t = np.arange(0,n)*dt
pnoise = 0.30
t1 = 12.4*60
t2 = 24*60
t3 = 15*24*60
tc = 60
xn = 5+3*np.cos(2*np.pi*t/t1)+2*np.cos(2*np.pi*t/t2)+1*np.cos(2*np.pi*t/t3)
xn = xn+pnoise*np.max(xn-np.mean(xn))*(0.5-np.random.rand(np.size(xn)))
xs = lanczos_filter(xn,dt,1./tc)[0]

plt.plot(xn)
plt.plot(xs,'k')
#%%
covtot = np.cov(x,rowvar=False)
(n,p) = x.shape

# center data
x = x - np.nanmean(x,0)[np.newaxis,...]
xs = x * np.transpose(scale)
#%%
covtot = np.transpose(scale)*covtot*scale
pcvec, evl, rest = peigs(covtot, min(n-1,p))
trcovtot = np.trace(covtot)
#%%
pvar = evl/trcovtot*100
# principal component time series
pcs = np.dot(xs, pcvec)
# return EOFs in original scaling as patterns (row vectors)
eof = np.transpose(pcvec)/np.transpose(scale)
#%%
ntr = truncation
f = np.sqrt(np.squeeze(evl)[0:ntr])
s = np.dot(pcvec[:,0:ntr], np.diag(1./f))
sadj = np.dot(np.diag(f), np.transpose(pcvec[:,0:ntr]))
#%%
b,a = butter(5,1./cutoff,btype='low')
t = np.arange(1,n+1)
#t.shape = (1,n)
#t = np.transpose(t)
x_f = xs.copy()
for i in range(xs.shape[1]):
    p = np.polyfit(t,xs[:,i],1)
    tmp = xs[t-1,i]-p[0]*t-p[1]
    tmp1 = np.concatenate((np.flipud(tmp),tmp,np.flipud(tmp)))
    tmp_filt = filtfilt(b,a,tmp)
    x_f[:,i] = tmp_filt+p[0]*t+p[1]
#%%
y = np.dot(x_f, s)
gamma = np.cov(y,rowvar=False)
dummy, r, v = csvd(gamma)
weights = scale * np.dot(s, v)
lfps = np.dot(np.transpose(v), sadj)/np.transpose(scale)