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
import numpy as np
import calendar
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3

indir = '/archive/Zhaoyi.Shen/fms/ulm/AM2/'
indir_sub = 'ts/monthly/16yr/'
flag = 'annual'
exper = 'AM2_'
#exper = 'c48L48_am3p11_'
pert = ['control_1990','m2c25w30','m6c25w30']
npert = np.size(pert)
ens = ['']
nens = np.size(ens)
plat = 'gfdl.ncrc3-default-prod-openmp/'
diag = 'atmos'
var = 'swdn_toa'
varo = var
sim = []
simo = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)

zm = True
gm = True
init = False
init3d = False
init3dp = False
yr1 = np.arange(1983,1984,1)
yr2 = np.arange(1998,1999,1)
yr_ts = np.ones(16)
nyr1 = np.size(yr1)
nyr = yr2[-1]-yr1[0]+1
yrstr = []
for yri in range(nyr1):
    yr1C = ('000'+str(yr1[yri]))[-4:]
    yr2C = ('000'+str(yr2[yri]))[-4:]
    yrC = yr1C+'01-'+yr2C+'12.'+var
    yrstr.append(yrC)
ind = range(nsim)
nfile = np.size(yrstr)

#npert = 1
land_mask = 0
for i in ind:
    atmdir = indir+exper+sim[i]+'/'+plat+'pp/'+diag+'/'
    stafile = atmdir+diag+'.static.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    if ('lat' in fs[-1].variables):
        lat = fs[-1].variables['lat'][:].astype(np.float64)
        lon = fs[-1].variables['lon'][:].astype(np.float64)
        nlat = np.size(lat)
        nlon = np.size(lon)
    #phalf = fs[-1].variables['phalf'][:].astype(np.float64)
    #zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    if ('land_mask' in fs[-1].variables):
        land_mask = fs[-1].variables['land_mask'][:].astype(np.float64)
    fs[-1].close()
    #%%
    filedir = atmdir+indir_sub
    files = []
    for fi in range(nfile):
        filename = filedir+diag+'.'+yrstr[fi]+'.nc'
        files.append(filename)
    ts = pp.ts_multi_files(files,var,0)
    ts = np.array(ts)
    #%%
    if (flag != ''):
        ts = pp.month_to_year(ts,yr_ts,flag)
    if zm:
        ts_zon = np.mean(ts,-1)
    if gm:
        area = grid.calcGridArea(lat,lon)
        area.shape = [1,nlat,nlon]
        ts_glb = np.sum(ts*area,-1)
        ts_glb = np.sum(ts_glb,-1)
        ts_glb = ts_glb/np.sum(area)
        print ts_glb
    plt.plot(ts_zon[0,:]-ts_zon[0,0])
plt.legend(pert,fontsize=10)
