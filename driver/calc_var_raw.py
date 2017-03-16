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
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3

indir = '/archive/Zhaoyi.Shen/fms/ulm_201510/Held-Suarez/'
outdir = '/home/z1s/research/BC-circulation/analysis/npz/idealized/ulm/'
exper = 'HSt42_'
pert = ['Ty90','l8_heat1x_Ty90']
perto = ['Ty90','l8_1x_Ty90']
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-intel-prod'
diag = 'atmos_daily'
diago = 'var3d'
var = 'temp'
varo = 'temp_zm'
npert = np.size(pert)
init = True
init3d = True
for i in range(npert):
    atmdir = indir+exper+pert[i]+plat+reg+'/history/'
    stafile = atmdir+'00000.atmos_average.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    phalf = fs[-1].variables['phalf'][:].astype(np.float64)
    zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    #%%
    filename = atmdir+'00000.atmos_daily.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    tmp = fs[-1].variables[var][500:,:,:,:].astype(np.float64) #t,p,lat,lon
    tmp_zm = np.mean(tmp,3)
    tmp_zm = np.mean(tmp_zm,0)
    if init:
        outfile = outdir+'dim.'+perto[i]+'.npz'
        fio.save(outfile,lat=lat,lon=lon,phalf=phalf)  
    if init3d:
        pfull = fs[-1].variables['pfull'][:].astype(np.float64)
        outfile = outdir+'dim.'+perto[i]+'.npz'
        fio.save(outfile,pfull=pfull)
    outfile = outdir+diago+'.'+perto[i]+'.npz'
    fio.save(outfile,**{varo:tmp_zm})
    
