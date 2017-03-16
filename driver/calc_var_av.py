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
import numpy as np
import calendar
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3

indir = '/archive/z1s/ulm_201505_c3/'
indir_sub = 'ts/annual/20yr/'
outdir = '/home/z1s/research/nonlinearity/npz/SM2/'
outdir_sub='av/0781-0860/'
exper = 'SM2_'
pert = ['control_1990','m2c25w30','m2c50w30']
perto = ['ctrl','c25','c50']
plat = 'gfdl.ncrc3-intel-prod-openmp/'
diag = 'atmos_level'
diago = 'var2d'
var = 'swdn_toa'
varo = 'swdn_toa_zm'
yr1 = np.arange(781,822,20)
yr2 = np.arange(800,861,20)
nyr = np.size(yr1)
yrstr = []
for yri in range(nyr):
    yr1C = ('000'+str(yr1[yri]))[-4:]
    yr2C = ('000'+str(yr2[yri]))[-4:]
    yrC = yr1C+'-'+yr2C+'.'+var
    yrstr.append(yrC)

npert = np.size(pert)
ind = range(npert)

nfile = np.size(yrstr)
init = False
init3d = False
#npert = 1
for i in ind:
    atmdir = indir+exper+pert[i]+'/'+plat+'pp/'+diag+'/'
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
    filedir = atmdir+indir_sub
    files = []
    for fi in range(nfile):
        filename = filedir+diag+'.'+yrstr[fi]+'.nc'
        files.append(filename)
    av = pp.av_multi_files(files,var,0)
    av = np.mean(av,-1)
    #%%
    if init:
        outfile = outdir+'dim.'+perto[i]+'.npz'
        fio.save(outfile,lat=lat,lon=lon,phalf=phalf)  
    if init3d:
        tmp = nc.netcdf_file(files[-1],'r',mmap=True)
        pfull = tmp.variables['pfull'][:].astype(np.float64)
        outfile = outdir+'dim.'+perto[i]+'.npz'
        fio.save(outfile,pfull=pfull)       
    outfile = outdir+outdir_sub+diago+'.'+perto[i]+'.npz'
    fio.save(outfile,**{varo:av})
    
