# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:37:34 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import amgrid as grid
from scipy.io import netcdf as nc
import numpy as np
import calendar
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3

basedir = '/archive/Zhaoyi.Shen/fms/ulm_201505_c3/SM2/'
outdir = '/home/z1s/research/nonlinearity/analysis/npz/'
exper = 'SM2_'
pert = ['control_1990','m2c35w30']
plat = '/gfdl.ncrc3-intel-prod-openmp/'
diag = 'atmos_month'
reg = '1x12m0d_30x4a/'
date = '07610101.'
var = 'swdn_toa'
"""
basedir = '/archive/Zhaoyi.Shen/ulm_201510/'
outdir = '/home/z1s/research/nonlinearity/analysis/npz/'
exper = 'imr_t42_'
pert = ['control','m2c25w30','m3c25w30','m2c35w30','m2c40w30','m2c45w30','m2c50w30','m2c60w60','asyme2']
plat = '/gfdl.ncrc3-default-prod/'
diag = 'atmos_daily'
reg = '1x0m2d_32pe/'
date = '00010101.'
var = 'swdn_toa'
"""
npert = np.size(pert)
#npert = 2
nlat = 90
nlon = 144
nt = 2
data = np.zeros([npert,nlat])
swts = np.zeros([npert,nt])
fs = []
for i in range(npert):
    filedir = basedir+exper+pert[i]+plat+reg+'history/'
    filename = filedir+date+diag+'.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    #pfull = fs[-1].variables['pfull'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    tmp = fs[-1].variables[var][:,:,:].astype(np.float64) #t,p,lat,lon
    #tmp[np.where(tmp<-999)] = np.nan
    tmp = np.mean(tmp,2)
    #swts[i,:] = tmp[:,57]
    data[i,:] = np.mean(tmp,0)
    fs[-1].close()
    #%%
nlat = np.size(lat)
weight = np.cos(lat*np.pi/180.)
#weight = np.cos(lat)
weight1 = weight/np.sum(weight)
weight1.shape = [1,nlat]
tmpglb = np.sum(data*weight1,1)

for i in range(npert):
    plt.plot(lat,data[i,:]-data[0,:])
plt.figure()
for i in [0,1]:
    plt.plot(lat,data[i,:])
#%%

