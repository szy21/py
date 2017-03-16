# -*- coding: utf-8 -*-
"""
Created on Mon May 2 09:37:34 2016

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import postprocess as pp
from scipy.io import netcdf as nc
from netCDF4 import Dataset
import numpy as np
import calendar
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3
obs= 'GISTEMP'
indir = '/archive/Zhaoyi.Shen/home/research/climate/observation/'
filename = indir+'GISTEMP/gistemp250.nc'
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz_interp/'+obs+'/'
flag = ['annual','MAM','JJA','SON','DJF']
nflag = np.size(flag)
sub_dict = {'annual':'annual','MAM':'MAM','JJA':'JJA','SON':'SON','DJF':'DJF'}
diago = 'var2d'
var = 'tempanomaly'
varo = 't_ref'
zm = False
init = False
init3d = False
timeo = '1880-2015'
yr_ts = np.arange(1880,2016,1)

fs = []
#fs.append(nc.netcdf_file(filename,'r',mmap=True))
fs.append(Dataset(filename,'r',mmap=True))
lat = np.array(fs[-1].variables['lat'][:].astype(np.float64))
lon = np.array(fs[-1].variables['lon'][:].astype(np.float64))
land_mask = 0
if 'land_mask' in fs[-1].variables:
    land_mask = fs[-1].variables['land_mask'][:].astype(np.float64)
    land_mask = np.array(land_mask)
fs[-1].close()
#%%
files = []
nfile = 1
for fi in range(nfile):
    files.append(filename)
ts = pp.ts_multi_files(files,var,0)
#%%
ts = ts[:-5,:,:]
#ts = np.array(ts)
#ts[np.where(abs(ts)>999)] = np.nan
if zm:
    ts = np.mean(ts,-1)
#%%
if init:
    outfile = outdir+'dim.'+obs+'.npz'
    fio.save(outfile,lat=lat,lon=lon,land_mask=land_mask)
if init3d:
    tmp = nc.netcdf_file(files[-1],'r',mmap=True)
    pfull = tmp.variables['pfull'][:].astype(np.float64)
    outfile = outdir+'dim.'+obs+'.npz'
    fio.save(outfile,pfull=pfull)
for flagi in range(nflag):
    if (flag[flagi] != ''):
        ts_flag = pp.month_to_year(ts,yr_ts,flag[flagi])
        outdir_sub='ts/'+sub_dict[flag[flagi]]+'/'
    outfile = outdir+outdir_sub+diago+'.'+timeo+'.'+obs+'.npz'
    fio.save(outfile,**{varo:ts_flag})

