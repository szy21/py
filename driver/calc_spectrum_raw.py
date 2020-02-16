# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:37:34 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib/')
import binfile_io as fio
import spacetime_analysis as stan
import amgrid as grid
from scipy.io import netcdf as nc
import numpy as np
from numpy import fft
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
pert = ['s24','s25']
perto = ['s24','s25']
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-default-prod'
diag = 'atmos_daily'
diago = 'var3d'
npert = np.size(pert)
var_name = 'vcomp' #if adjust, vcomp must be var1
adj = True #T if vcomp, F if omega (need different adjustment)
sind = range(npert)
init = False
init3d = False
nfft_time = 60
int_time = 86400
L = np.zeros((npert,64))
for si in range(npert):
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
    var = fs[-1].variables[var_name][500:,:,:,:].astype(np.float64) #t,p,lat,lon
    ps = fs[-1].variables['ps'][500:,:,:].astype(np.float64)
    fs[-1].close()
    var_tr = var-np.mean(var,-1)[...,np.newaxis]
    power = np.abs(fft.fft(var_tr,axis=-1))**2
    phalf = grid.calcSigmaPres(ps,pk,bk)
    dp = (phalf[:,1:,:,:]-phalf[:,:-1,:,:])
    dp_m = np.mean(dp,-1)[...,np.newaxis]
    power_col = np.sum(power*dp,1)/np.sum(dp,1)
    power_m = np.mean(power_col,0)[:,:nlon/2]
    k = np.arange(0,nlon/2)
    k_m = np.sum(power_m*k,-1)/np.sum(power_m,-1)
    L[si,:] = 2*np.pi*RADIUS*np.cos(lat*np.pi/180.)/k_m
    if init:
        outfile = outdir+'dim.'+perto[si]+'.npz'
        fio.save(outfile,lat=lat,lon=lon,phalf=phalf)  
    if init3d:
        pfull = fs[-1].variables['pfull'][:].astype(np.float64)
        outfile = outdir+'dim.'+perto[si]+'.npz'
        fio.save(outfile,pfull=pfull)
    #outfile = outdir+'av/'+diago+'.'+perto[si]+'.npz'
    #fio.save(outfile,**{varo_name:tr_zm})
    
#%%
color = ['k','b','r','b','r']
ls = ['-','-','-','--','--']
plt.figure()
for si in range(npert):
    plt.plot(lat,L[si,:]/1.e6,color=color[si],ls=ls[si])
plt.xlim(-90,90)
plt.xticks(np.arange(-60,61,30))
plt.xlabel('Latitude')
plt.ylabel('Length (10^6 m)')
plt.legend(perto)
