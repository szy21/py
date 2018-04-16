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
import matplotlib as mpl
fsize = 16
font = {'size'   : fsize}
mpl.rc('font', size=fsize)
mpl.rc('lines', linewidth=2)
mpl.rc('figure', figsize=[4.5*4./3,4.5])
mpl.rc('font', family='sans-serif')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['ps.useafm'] = True

CP_AIR = 1004.64
LE = 2.500e6
GRAV = 9.80
RAIR = 287.04
RADIUS = 6371.0e3

indir = '/archive/yim/HS_michelle/'
outdir = '/home/z1s/research/HS_michelle/npz/305K/'
exper = 'HSt42_with_clouds_'
pert = ['s31']
lat_bound = 0
sc = np.array((0.,1.,2.))
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-default-prod'
diag = 'atmos_daily'

npert = np.size(pert)
var1_name = 'vcomp' #if adjust, vcomp must be var1
var2_name = 'temp'
varo_name = 'vTtr_zm'
adj = True #T if vcomp, F if omega (need different adjustment)
sind = range(npert)
init = True
init3d = False
pind_top = 0
pind_bot = 3
for si in sind:
    atmdir = indir+exper+pert[si]+plat+reg+'/history/'
    stafile = atmdir+'00000.atmos_average.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    pfull = fs[-1].variables['pfull'][pind_top:pind_bot].astype(np.float64)
    #zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    nlev = np.size(pfull)
    #%%
    filename = atmdir+'00000.atmos_daily.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    vcomp = fs[-1].variables['vcomp'][500:,pind_top:pind_bot,:,:].astype(np.float64) #t,p,lat,lon
    temp = fs[-1].variables['temp'][500:,pind_top:pind_bot,:,:].astype(np.float64)
    height = fs[-1].variables['height'][500:,pind_top:pind_bot,:,:].astype(np.float64)
    q = fs[-1].variables['mix_rat'][500:,pind_top:pind_bot,:,:].astype(np.float64)
    ps = fs[-1].variables['ps'][500:,:,:].astype(np.float64)
    fs[-1].close()
    nt = np.shape(vcomp)[0]
    phalf = grid.calcSigmaPres(ps,pk,bk)
    dp = (phalf[:,1:,:,:]-phalf[:,:-1,:,:])[:,pind_top:pind_bot,:,:]
    sc_arr = np.zeros(np.shape(vcomp))
    sc_arr[...] = sc[si]
    #sc_arr[:,np.where(pfull>=800),:,:] = 0
    sc_arr[:,:,np.where((lat>=-lat_bound) & (lat<=lat_bound)),:] = 0   
    MSE = CP_AIR*temp+GRAV*height+LE*q*sc_arr
    DSE = CP_AIR*temp+GRAV*height
    Lq = LE*q*sc_arr
    MSE = MSE
    ylabel = 'MSE flux (PW)'
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
    
    vMSE_ed1 = np.sum(vcomp_ed*MSE_m*dp_ed,1)/GRAV
    vMSEcol_Ed1 = np.mean(np.mean(vMSE_ed1,-1),0)
    vMSE_ed2 = np.sum(vcomp_m*MSE_ed*dp_ed,1)/GRAV
    vMSEcol_Ed2 = np.mean(np.mean(vMSE_ed2,-1),0)
    vMSE_ed3 = np.sum(vcomp_ed*MSE_ed*dp_ed,1)/GRAV
    vMSEcol_Ed3 = np.mean(np.mean(vMSE_ed3,-1),0)
    if si==0:
        Tot = vMSEcol_Tot[np.newaxis,...]
        MMC = vMSEcol_MMC[np.newaxis,...]
        Ed = vMSEcol_Ed[np.newaxis,...]
    else:
        Tot = np.concatenate((Tot,vMSEcol_Tot[np.newaxis,...]),axis=0)
        MMC = np.concatenate((MMC,vMSEcol_MMC[np.newaxis,...]),axis=0)
        Ed = np.concatenate((Ed,vMSEcol_Ed[np.newaxis,...]),axis=0)
    
#%%
"""
plt.figure()
plt.contourf(lat,pfull,np.mean(np.mean(vcomp_m,-1),0),cmap=plt.cm.RdYlBu_r)
plt.gca().invert_yaxis()
plt.colorbar()
"""
sc = 1e-15
xsc = 2*np.pi*RADIUS*np.cos(lat/180.*np.pi)

x = lat
lattick = np.arange(-60,61,30)
xtick = lattick
xlim = [-90,90]
xticklabel = lattick
#%%
#ylabel = 'Zonal wind (m s$^{-1}$)'
plt.figure()
plt.plot(x,vMSEcol_Tot*xsc*sc,'r')
plt.plot(x,(vMSEcol_MMC+vMSEcol_Ed)*xsc*sc,'r--')
plt.plot(x,vMSEcol_MMC*xsc*sc,'k-')
plt.plot(x,vMSEcol_Ed*xsc*sc,'k:')
#plt.plot((vMSEcol_Ed1+vMSEcol_Ed2+vMSEcol_Ed3)*2*np.pi*RADIUS*np.cos(lat/180.*np.pi),'--')
"""
ind = range(npert)
color = ['k','b','r','b','r']
ls = ['-','-','-','--','--']
plt.figure()
for i in sind:
    plt.plot(x,Tot[i,:]*xsc*sc,color=color[i],ls='-')
plt.legend(pert,fontsize=fsize,loc=2)
for i in sind:
    plt.plot(x,MMC[i,:]*xsc*sc,color=color[i],ls='--')
for i in sind:
    plt.plot(x,Ed[i,:]*xsc*sc,color=color[i],ls=':')
 """   
#plt.gca().invert_yaxis()
#%%
plt.xticks(xtick)
plt.xlim(xlim)
plt.ylim([-4.5,4.5])
plt.xlabel('Latitude')
plt.ylabel(ylabel)

plt.tight_layout()