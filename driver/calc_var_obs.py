# -*- coding: utf-8 -*-
"""
Created on Mon May 2 09:37:34 2016

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import postprocess as pp
import amgrid as grid
from scipy.io import netcdf as nc
from netCDF4 import Dataset
import xarray as xr
import numpy as np
import calendar
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3
obs= 'GISv3_1200'
indir = '/archive/Zhaoyi.Shen/home/research/climate/observation/'
filename = indir+'GISTEMP/gistemp1200_ERSSTv4.nc'
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz_interp/'+obs+'/'
flag = ['JJAS','JJA','SON']
nflag = np.size(flag)
sub_dict = {'':'monthly','annual':'annual',\
    'MAM':'MAM','JJA':'JJA','SON':'SON','DJF':'DJF',\
    'MJJASO':'MJJASO','JJASON':'JJASON','JJAS':'JJAS'}
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
nlat = np.size(lat)
lon = np.array(fs[-1].variables['lon'][:].astype(np.float64))
nlon = np.size(lon)
land_mask = np.array((0))
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
ts[np.where(abs(ts)>300)] = np.nan
ts = ts[0:-5,:,:]
print ts.shape[0]/12.
"""
ts_ann = pp.month_to_year(ts,yr_ts,'annual')
area = grid.calcGridArea(lat,lon)
area.shape = (1,nlat,nlon)
land_mask.shape = (1,nlat,nlon)

for i in range(ts_ann.shape[0]):
    nan_ind = np.where(np.isnan(ts_ann[i,:,:]))
    land_mask[0,nan_ind[0],nan_ind[1]] = 0
ts_ann[np.where(np.isnan(ts_ann))]=0

ts_gm = np.sum(np.nansum(ts_ann*area*land_mask,-1),-1)/np.nansum(area*land_mask)
land_mask.shape = (nlat,nlon)
area.shape = (nlat,nlon)
weight = area/np.sum(area)
"""
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
ts_flag = ts.copy()
for flagi in range(nflag):
    if (flag[flagi] != ''):
        ts_flag = pp.month_to_year(ts,yr_ts,flag[flagi])
    outdir_sub='ts/'+sub_dict[flag[flagi]]+'/'
    outfile = outdir+outdir_sub+diago+'.'+timeo+'.'+obs+'.npz'
    fio.save(outfile,**{varo:ts_flag})

