# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:37:34 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib/')
import binfile_io as fio
import amgrid as grid
from scipy.io import netcdf as nc
import numpy as np
import calendar
import matplotlib.pyplot as plt

CP_AIR = 1004.64
LE = 2.500e6
GRAV = 9.80
RAIR = 287.04
RADIUS = 6371.0e3

indir = '/archive/yim/HS_michelle/'
outdir = '/home/z1s/research/HS_michelle/npz/'
exper = 'HSt42_with_clouds_'
pert = ['s4','s5','s6','s7','s8','s9']
perto = ['LH1lat15','LH0.5lat15','LH0lat15','LH1','LH0.5','LH0']
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-default-prod'
diag = 'atmos_daily'
diago = 'var3d'
npert = np.size(pert)
var1_name = 'vcomp' #if adjust, vcomp must be var1
var2_name = 'vcomp'
varo_name = 'vvTr_zm'
adj = True #T if vcomp, F if omega (need different adjustment)
sind = range(npert)
init = False
init3d = False
for si in sind:
    atmdir = indir+exper+pert[si]+plat+reg+'/history/'
    stafile = atmdir+'00000.atmos_average.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    phalf = fs[-1].variables['phalf'][:].astype(np.float64)
    #zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    nlev = np.size(phalf)
    #%%
    filename = atmdir+'00000.atmos_daily.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    var1 = fs[-1].variables[var1_name][500:,:,:,:].astype(np.float64) #t,p,lat,lon
    var2 = fs[-1].variables[var2_name][500:,:,:,:].astype(np.float64) #t,p,lat,lon
    var1_tr = var1-np.mean(var1,-1)[...,np.newaxis]    
    var2_tr = var2-np.mean(var2,-1)[...,np.newaxis]
    tr = var1_tr*var2_tr
    tr_zm = np.mean(tr,-1)
    tr_zm = np.mean(tr_zm,0)
    if init:
        outfile = outdir+'dim.'+perto[si]+'.npz'
        fio.save(outfile,lat=lat,lon=lon,phalf=phalf)  
    if init3d:
        pfull = fs[-1].variables['pfull'][:].astype(np.float64)
        outfile = outdir+'dim.'+perto[si]+'.npz'
        fio.save(outfile,pfull=pfull)
    outfile = outdir+'av/'+diago+'.'+perto[si]+'.npz'
    fio.save(outfile,**{varo_name:tr_zm})
    
#%%
"""
Tot1 = (MMC+St+Tr)
xsc = 2*np.pi*Rad*np.cos(lat/90*np.pi/2)
xsc.shape = [1,nlat]
clev = np.arange(-7,8,1)*1e15

plt.figure(figsize=[4*6/3,4])
plt.contourf(mon,lat,np.transpose(np.sum(St*dp1,1)/g*xsc),clev,cmap=plt.cm.RdYlBu_r,extend='both')
plt.colorbar()
#%%
clev = np.arange(0,1.1,0.1)*1e6
plt.figure(figsize=[4*6/3,4])
plt.contourf(mon,lat,np.transpose(np.sum(EKETr*dp1,1)/g),clev,cmap=plt.cm.hot,extend='max')
plt.colorbar()
"""