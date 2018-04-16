# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:37:34 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import amgrid as grid
import postprocess as pp
from scipy.io import netcdf as nc
import numpy as np
import matplotlib.pyplot as plt

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3
p0 = 1.0e3

indir = '/archive/yim/HS_michelle/'
outdir = '/home/z1s/research/HS_michelle/npz/305K/'
exper = 'HSt42_with_clouds_'
pert = ['s22','s23']
perto = ['LH1evap30dv10','LH1evap23dv10']
sc = np.array((1.,1.))
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-default-prod'
diag = 'atmos_daily'
diago = 'var3d'
var = 'ucomp'
varo = 'ucomp_zm'
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
    pfull = fs[-1].variables['pfull'][:].astype(np.float64)
    #zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    #%%
    filename = atmdir+'00000.atmos_daily.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    
    temp = fs[-1].variables['temp'][500:,:,:,:].astype(np.float64) #t,p,lat,lon
    q = fs[-1].variables['mix_rat'][500:,:,:,:].astype(np.float64)
    pfull.shape = (1,20,1,1)
    theta = temp*(p0/pfull)**(Rair/Cp)
    thetae = theta*np.exp(sc[i]*Le*q/(Cp*temp))
    temp_zm = np.mean(np.mean(temp,3),0)
    theta_zm = np.mean(np.mean(theta,3),0)
    thetae_zm = np.mean(np.mean(thetae,3),0)
    pfull.shape = (20)
    
    tmp = fs[-1].variables[var][500:,:,:,:].astype(np.float64)
    tmp_zm = np.mean(np.mean(tmp,3),0)
    if init:
        outfile = outdir+'dim.'+perto[i]+'.npz'
        fio.save(outfile,lat=lat,lon=lon)  
    if init3d:
        pfull = fs[-1].variables['pfull'][:].astype(np.float64)
        outfile = outdir+'dim.'+perto[i]+'.npz'
        fio.save(outfile,pfull=pfull,phalf=phalf)
    outfile = outdir+'av/'+diago+'.'+perto[i]+'.npz'
    fio.save(outfile,**{varo:tmp_zm})
    fio.save(outfile,temp_zm=temp_zm,theta_zm=theta_zm,thetae_zm=thetae_zm)
    
