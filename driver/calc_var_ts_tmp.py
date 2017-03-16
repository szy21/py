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

indir = '/archive/ccsp/ipcc_ar4/'
#indir = '/archive/bw/fms/memphis/narccap/'
indir_sub = 'ts/monthly/140yr/'
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz/perfect_model/'
exper = 'CM2.1U-D4_1860-2000-'
#exper = 'm45_narccap_hist_'
pert_dict = {'AllForc_H2':'CM2','cm2':'AM2'}
pert = ['AllForc_H2']
ens = ['']
enso = ens

flag = ['']
nflag = np.size(flag)
sub_dict = {'':'monthly','annual':'annual','MAM':'MAM','JJA':'JJA','SON':'SON','DJF':'DJF'}

npert = np.size(pert)
nens = np.size(ens)
plat = ''
diag = 'atmos'
diago = 'var2d'
var = ['t_surf','alb_sfc','ice_mask']

nvar = np.size(var)
varo = var
sim = []
simo = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
        simo.append(pert_dict[pert[i]]+enso[j])
nsim = np.size(sim)

zm = False
init = False
init3d = False
init3dp = False
timeo = '196901-200012'
yr1 = np.arange(1861,1862,1)
yr2 = np.arange(2000,2001,1)
nyr1 = np.size(yr1)
nyr = yr2[-1]-yr1[0]+1
yr = np.arange(yr1[0],yr2[-1],1)
yrstr = []
for yri in range(nyr1):
    yr1C = ('000'+str(yr1[yri]))[-4:]
    yr2C = ('000'+str(yr2[yri]))[-4:]
    yrC = yr1C+'01-'+yr2C+'12.'
    yrstr.append(yrC)
ind = range(nsim)
nfile = np.size(yrstr)

"""
check before this line
"""

#%%
#npert = 1
land_mask = 0
for vi in range(nvar):
    for i in ind:
        atmdir = indir+exper+sim[i]+'/'+plat+'pp/'+diag+'/'
        stafile = atmdir+diag+'.static.nc'
        fs = []
        fs.append(nc.netcdf_file(stafile,'r',mmap=True))
        bk = fs[-1].variables['bk'][:].astype(np.float64)
        pk = fs[-1].variables['pk'][:].astype(np.float64)
        lat = fs[-1].variables['lat'][:].astype(np.float64)
        lon = fs[-1].variables['lon'][:].astype(np.float64)
        phalf = fs[-1].variables['phalf'][:].astype(np.float64)
        zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
        if ('land_mask' in fs[-1].variables):
            land_mask = fs[-1].variables['land_mask'][:].astype(np.float64)
        fs[-1].close()
        nlat = np.size(lat)
        nlon = np.size(lon)
        nphalf = np.size(phalf)
        #%%
        filedir = atmdir+indir_sub
        files = []
        for fi in range(nfile):
            filename = filedir+diag+'.'+yrstr[fi]+var[vi]+'.nc'
            files.append(filename)
        ts = pp.ts_multi_files(files,var[vi],0)
        if zm:
            ts = np.nanmean(ts,-1)
        #%%
        if init and vi==0:
            outfile = outdir+'dim.'+simo[i]+'.npz'
            fio.save(outfile,lat=lat,lon=lon,phalf=phalf,land_mask=land_mask,year=yr)
        if init3d:
            tmp = nc.netcdf_file(files[-1],'r',mmap=True)
            pfull = tmp.variables['pfull'][:].astype(np.float64)
            outfile = outdir+'dim.'+simo[i]+'.npz'
            fio.save(outfile,pfull=pfull)
        if init3dp:
            tmp = nc.netcdf_file(files[-1],'r',mmap=True)
            level = tmp.variables['level'][:].astype(np.float64)
            outfile = outdir+'dim.'+simo[i]+'.npz'
            fio.save(outfile,level=level)
        for flagi in range(nflag):
            ts_flag = ts
            outdir_sub='ts/'+sub_dict[flag[flagi]]+'/'
            if (flag[flagi] != ''):
                ts_flag = pp.month_to_year(ts,flag[flagi])
            outfile = outdir+outdir_sub+diago+'.'+timeo+'.'+simo[i]+'.npz'
            fio.save(outfile,**{varo[vi]:ts_flag[108*12:,...]})
    print var[vi]
    
