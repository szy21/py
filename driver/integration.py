# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:03:00 2016

@author: Zhaoyi.Shen
"""

from scipy.integrate import quad
import numpy as np

def integrand(x,xa,sigma):
    return np.exp(-(x-xa)**2/(2*sigma**2))*np.cos(x)/2.
    #return np.cos(x)

xa = 45./180.*np.pi
width = 30./180.*np.pi
sigma = width/(2*(2*np.log(100))**0.5)
I = quad(integrand,-np.pi/2,np.pi/2,args=(xa,sigma))

lat = np.linspace(-np.pi/2,np.pi/2,65)
sinlat = np.sin(lat)
latc = 0.5*(lat[1:]+lat[:-1])
dlat = lat[1]-lat[0]
dsinlat = sinlat[1:]-sinlat[:-1]
flat = np.exp(-(latc-xa)**2/(2*sigma**2))*np.cos(latc)
M = np.sum(flat*dlat)/2
print M