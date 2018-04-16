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

indir = '/archive/Zhaoyi.Shen/fms/ulm/AM3/'
indir_sub = 'ts/monthly/5yr/'
flag = 'annual'
exper = 'c48L48_am3p11_'
#exper = 'SM2_'
pert = ['1860']#,'allforcr','aeroOnly','1860aeror']
npert = np.size(pert)
ens = ['_A1']
nens = np.size(ens)
plat = 'gfdl.ncrc3-intel-prod-openmp/'
diag = 'atmos_month_aer'
var = 'sulfate_col'
varo = var
sim = []
simo = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)

zonal_mean = False
glb_mean = True
col_integ = False
init = False
init3d = False
init3dp = False
yr_ts = np.ones(1000)
yr1 = np.arange(1870,2011,5)
yr2 = np.arange(1874,2015,5)
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
    if col_integ:
        ts = ts[:,-1,:,:]
    if zonal_mean:
        ts = np.mean(ts,-1)
    if glb_mean:
        area = grid.calcGridArea(lat,lon)
        ts = np.sum(ts*area,-1)
        ts = np.sum(ts,-1)
        ts = ts/np.sum(area)
    #%%
    if (flag != ''):
        ts = pp.month_to_year(ts,yr_ts,flag)
    if (i==0):
        ts0 = ts
    plt.plot(ts)
    print np.mean(ts[-40:])
plt.legend(pert,fontsize=10)
