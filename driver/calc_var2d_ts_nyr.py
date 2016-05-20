# -*- coding: utf-8 -*-
"""
Created on Mon May 2 09:37:34 2016

@author: z1s
"""

import sys
sys.path.append('/home/z1s/PythonScripts')
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

basedir = '/archive/Zhaoyi.Shen/ulm_201510/'
outdir = '/home/z1s/research/nonlinearity/analysis/npz/ts/'
exper = 'imr_t42_'
pert = ['control','2xCO2',\
        'm2c25w30','m2c30w30','m2c35w30','m2c40w30','m2c45w30','m2c50w30',\
        '2xCO2+m2c35w30','2xCO2+m2c40w30','2xCO2+m2c45w30',\
        'm3c35w30','m3c40w30']
perto = ['ctrl','CO2',\
        'm2c25w30','m2c30w30','m2c35w30','m2c40w30','m2c45w30','m2c50w30',\
        'CO2+m2c35w30','CO2+m2c40w30','CO2+m2c45w30',\
        'm3c35w30','m3c40w30']
plat = '/gfdl.ncrc3-default-prod/'
diag = 'atmos_level'
diago = 'var2d'
var = 't_surf'
varo = 'tsfc'
npert = np.size(pert)
ind = range(npert)
ind = [11,12]
subdir = 'ts/monthly/6yr/'
yrC = '000101-000612'
#npert = 1
for i in ind:
    atmdir = basedir+exper+pert[i]+plat+'pp/'+diag+'/'
    stafile = atmdir+diag+'.static.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    phalf = fs[-1].variables['phalf'][:].astype(np.float64)
    zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    nphalf = np.size(phalf)
    #%%
    filedir = atmdir+subdir
    
    filename = filedir+diag+'.'+yrC+'.'+var+'.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    #pfull = fs[-1].variables['pfull'][:].astype(np.float64)
    tmp = fs[-1].variables[var][:].astype(np.float64) #t,p,lat,lon
    #tmp[np.where(tmp<-999)] = np.nan
    tmp = np.mean(tmp,2)
    #tmp = np.mean(tmp,0)
    fs[-1].close()
    #%%
    #outfile = outdir+'dim.'+perto[i]+'_sigma.npz'
    #fio.save(outfile,lat=lat,lon=lon,phalf=phalf,pfull=pfull)
    outfile = outdir+diago+'.'+perto[i]+'_sigma.npz'
    fio.save(outfile,**{varo:tmp})
    
