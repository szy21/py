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
import calendar
import matplotlib.pyplot as plt

CP_AIR = 1004.64
LE = 2.500e6
GRAV = 9.80
RAIR = 287.04
RADIUS = 6371.0e3

indir = '/archive/yim/HS_michelle/'
outdir = '/home/z1s/research/HS_michelle/npz/305K/'
exper = 'HSt42_with_clouds_'
pert = ['s24','s25']
perto = ['s24','s25']
lat_bound = 0
sc = np.array((1.,0.))
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-default-prod'
diag = 'atmos_daily'
diago = 'eflx2d'
npert = np.size(pert)
var1_name = 'vcomp' #if adjust, vcomp must be var1
var2_name = 'temp'
varo_name = 'vTtr_zm'
adj = True #T if vcomp, F if omega (need different adjustment)
sind = range(npert)
init = True
init3d = False
pind = 0
for si in sind:
    atmdir = indir+exper+pert[si]+plat+reg+'/history/'
    stafile = atmdir+'00000.atmos_average.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    pfull = fs[-1].variables['pfull'][pind:].astype(np.float64)
    #zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    nlev = np.size(pfull)
    #%%
    filename = atmdir+'00000.atmos_daily.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    vcomp = fs[-1].variables['vcomp'][500:,pind:,:,:].astype(np.float64) #t,p,lat,lon
    temp = fs[-1].variables['temp'][500:,pind:,:,:].astype(np.float64)
    height = fs[-1].variables['height'][500:,pind:,:,:].astype(np.float64)
    q = fs[-1].variables['mix_rat'][500:,pind:,:,:].astype(np.float64)
    ps = fs[-1].variables['ps'][500:,:,:].astype(np.float64)
    fs[-1].close()
    nt = np.shape(vcomp)[0]
    phalf = grid.calcSigmaPres(ps,pk,bk)
    dp = (phalf[:,1:,:,:]-phalf[:,:-1,:,:])[:,pind:,:,:]
    sc_arr = np.zeros(np.shape(vcomp))
    sc_arr[...] = sc[si]
    #sc_arr[:,np.where(pfull>=800),:,:] = 0
    sc_arr[:,:,np.where((lat>=-lat_bound) & (lat<=lat_bound)),:] = 0   
    MSE = CP_AIR*temp+GRAV*height+LE*q*sc_arr
    DSE = CP_AIR*temp+GRAV*height
    Lq = LE*q*sc_arr
    MSE = MSE
    MSE = MSE-np.mean(MSE)
    MSE_m = np.mean(MSE,-1)[...,np.newaxis]
    MSE_ed = MSE-MSE_m
    vcomp_m = np.mean(vcomp,-1)[...,np.newaxis]
    vcomp_ed = vcomp-vcomp_m
    dp_m = np.mean(dp,-1)[...,np.newaxis]
    dp_ed = dp-dp_m
    vMSE = np.sum(vcomp*MSE*dp,1)/GRAV #t,lat,lon
    vMSEcol_Tot = np.mean(np.mean(vMSE,-1),0)
    vMSE_m = np.sum(vcomp_m*MSE_m*dp_m,1)/GRAV #t,lat,lon
    vMSEcol_MMC = np.mean(np.mean(vMSE_m,-1),0)
    vMSE_ed = np.sum(vcomp_ed*MSE_ed*dp_m,1)/GRAV #t,lat,lon
    vMSEcol_Ed = np.mean(np.mean(vMSE_ed,-1),0)
    """
    vPos = vcomp_m.copy()
    vNeg = vcomp_m.copy()
    ind1 = np.where(vcomp_m<0)
    ind2 = np.where(vcomp_m>0)
    vPos[ind1] = 0
    vNeg[ind2] = 0
    ratio = np.sum(vPos*dp_m,1)/abs(np.sum(vNeg*dp_m,1))
    ratio1 = 1/ratio
    ind = np.where(np.isinf(ratio))
    ratio[ind] = 0
    ind = np.where(np.isinf(ratio1))
    ratio1[ind] = 0
    ratio.shape = (nt,1,nlat,1)
    ratio1.shape = (nt,1,nlat,1)
    vAdj = vPos*np.sqrt(ratio1)+vNeg*np.sqrt(ratio)
    """
    vMSE_ed1 = np.sum(vcomp_ed*MSE_m*dp_ed,1)/GRAV
    vMSEcol_Ed1 = np.mean(np.mean(vMSE_ed1,-1),0)
    vMSE_ed2 = np.sum(vcomp_m*MSE_ed*dp_ed,1)/GRAV
    vMSEcol_Ed2 = np.mean(np.mean(vMSE_ed2,-1),0)
    vMSE_ed3 = np.sum(vcomp_ed*MSE_ed*dp_ed,1)/GRAV
    vMSEcol_Ed3 = np.mean(np.mean(vMSE_ed3,-1),0)

    if init:
        outfile = outdir+'dim.'+perto[si]+'.npz'
        fio.save(outfile,lat=lat,lon=lon)  
    if init3d:
        pfull = fs[-1].variables['pfull'][:].astype(np.float64)
        outfile = outdir+'dim.'+perto[si]+'.npz'
        fio.save(outfile,pfull=pfull)
    outfile = outdir+'av/'+diago+'.'+perto[si]+'.npz'
    fio.save(outfile,\
             vMSETot_col_zm=vMSEcol_Tot,vMSEMMC_col_zm=vMSEcol_MMC,vMSEEd_col_zm=vMSEcol_Ed)
#%%
"""
plt.figure()
plt.contourf(lat,pfull,np.mean(np.mean(vcomp_m,-1),0),cmap=plt.cm.RdYlBu_r)
plt.gca().invert_yaxis()
plt.colorbar()
"""
#%%
plt.figure()
plt.plot(vMSEcol_Tot*2*np.pi*RADIUS*np.cos(lat/180.*np.pi))
plt.plot((vMSEcol_MMC+vMSEcol_Ed)*2*np.pi*RADIUS*np.cos(lat/180.*np.pi))
plt.plot(vMSEcol_MMC*2*np.pi*RADIUS*np.cos(lat/180.*np.pi),':')
plt.plot(vMSEcol_Ed*2*np.pi*RADIUS*np.cos(lat/180.*np.pi),':')
plt.plot((vMSEcol_Ed1+vMSEcol_Ed2+vMSEcol_Ed3)*2*np.pi*RADIUS*np.cos(lat/180.*np.pi),'--')
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