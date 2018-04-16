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

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3
p0 = 1.0e3

indir = '/archive/yim/HS_michelle/'
outdir = '/home/z1s/research/HS_michelle/npz/305K/'
exper = 'HSt42_with_clouds_'
pert = ['s31','s29','s30']
qsc = np.array((0.,1.,2.))
reg = '/1x0m1000d_32pe'
plat = '/gfdl.ncrc3-default-prod'
diag = 'atmos_daily'
diago = 'var3d'
var = 'ucomp'
sc = 1
clev = np.arange(260,361,15)
clev1 = np.arange(-18,19,2)/sc
clabel = 'theta'
npert = np.size(pert)
init = True
init3d = True
for i in range(npert):
    atmdir = indir+exper+pert[i]+plat+reg+'/history/'
    stafile = atmdir+'00000.atmos_average.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    #bk = fs[-1].variables['bk'][:].astype(np.float64)
    #pk = fs[-1].variables['pk'][:].astype(np.float64)
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
    thetae = theta*np.exp(qsc[i]*Le*q/(Cp*temp))
    temp_zm = np.mean(np.mean(temp,3),0)
    theta_zm = np.mean(np.mean(theta,3),0)
    thetae_zm = np.mean(np.mean(thetae,3),0)
    pfull.shape = (20)
    
    tmp = fs[-1].variables[var][500:,:,:,:].astype(np.float64)
    tmp_zm = np.mean(np.mean(tmp,3),0)
    tmp_zm = theta_zm
    if i==0:
        data = tmp_zm[np.newaxis,...]
    else:
        data = np.concatenate((data,tmp_zm[np.newaxis,...]),axis=0)
    if init3d:
        pfull = fs[-1].variables['pfull'][:].astype(np.float64)

latS = 0
latN = 90
plev = np.arange(1,25,1)
lattick = np.arange(-90,91,30)
ptick = np.array([0,0.2,0.4,0.6,0.8,1.0])*1000

cmap = plt.cm.RdYlBu_r
clev = clev[np.where(clev!=0)]
x = lat
y = pfull
xtick = lattick
xlim = [-90,90]
ytick = ptick
data = data*sc
#fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True)
#%%
for i in range(1,npert):
    #plt.subplot(3,1,i)
    plt.figure()
    cs = plt.contour(x,y,data[0,:,:],\
                     clev,\
                     linewidths=1,colors='k')
    plt.clabel(cs,fontsize=16,fmt='%1.0f',inline=1)
    cs = plt.contour(x,y,data[0,:,:],\
                     [0],\
                     linewidths=2,colors='k')
    diff = np.squeeze(data[i,:,:]-data[0,:,:])
    plt.contourf(x,y,diff,\
                 clev1,\
                 cmap=cmap,extend='both')
    cb = plt.colorbar(orientation='horizontal',pad=0.2)
    #cb.formatter.set_powerlimits((0,0))
    #cb.ax.yaxis.set_offset_position('right')
    cb.update_ticks()
    cb.set_label(clabel)
    plt.gca().invert_yaxis()
    
    plt.xticks(xtick)
    plt.xlim(xlim)
    plt.xlabel('Latitude')    
    plt.yticks(ytick)
    #plt.yscale('log')
    plt.ylabel('Pressure (hPa)')
    plt.tight_layout()
#%%
color = ['k','b','r','b','r']
ls = ['-','-','-','--','--']
plt.figure()
for i in range(npert):
    plt.plot(x,data[i,6,:],color=color[i],ls=ls[i])
plt.legend(pert,loc='best',ncol=3)
plt.xticks(xtick)
plt.xlim(xlim)
plt.xlabel('Latitude')
plt.ylabel('Zonal wind at 325 hPa (m/s)')
plt.tight_layout()   
