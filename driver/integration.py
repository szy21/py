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

xa = 15./180.*np.pi
width = 100./180.*np.pi
sigma = width/(2*(2*np.log(100))**0.5)
I = quad(integrand,-np.pi/2,np.pi/2,args=(xa,sigma))

lat = np.linspace(-np.pi/2,np.pi/2,91)
sinlat = np.sin(lat)
latc = 0.5*(lat[1:]+lat[:-1])
dlat = lat[1]-lat[0]
dsinlat = sinlat[1:]-sinlat[:-1]
flat = np.exp(-(latc-xa)**2/(2*sigma**2))*np.cos(latc)
M = np.sum(flat*dlat)/2
print M

xa_lon = 90./180.*np.pi
width_lon = 60./180.*np.pi
sigma_lon = width_lon/(2*(2*np.log(100))**0.5)
lon = np.linspace(0,np.pi*2,144)
lonc = 0.5*(lon[1:]+lon[:-1])
dlon = lon[1]-lon[0]
flon = np.exp(-(lonc-xa_lon)**2/(2*sigma_lon**2))
M_lon = np.sum(flon*dlon)/(2*np.pi)
print M_lon

print M*M_lon