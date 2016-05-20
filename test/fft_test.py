# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:38:35 2016

@author: Zhaoyi.Shen
"""
import numpy as np
import scipy.fftpack as fftpack
from scipy import signal
x = np.linspace(0, np.pi, 128)
t = np.arange(0, 365)
u = np.zeros((365, 128))
for i, i_x in enumerate(x):
    for j, j_t in enumerate(t):
        u[j,i] = np.sin(5*i_x - j_t/7.09)

v = u.copy()

pcp, cp = stan.calc_phasespeed_cospectrum(
    u, v, ncp=50,nwave=5,int_time=86400,nfft_time=30)