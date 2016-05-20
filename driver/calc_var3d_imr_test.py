# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:37:34 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/PythonScripts')
import binfile_io as fio
import amgrid as grid
from scipy.io import netcdf as nc
import numpy as np
import calendar
import matplotlib.pyplot as plt
import matplotlib as mpl

Cp = 1004.64
Le = 2.500e6
g = 9.80
Rair = 287.04
Rad = 6371.0e3

basedir = '/archive/Spencer.Clark/imr_skc/'
outdir = '/home/z1s/research/nonlinearity/analysis/npz/'
exper = ''
pert = ['control','2xCO2']
perto = ['ctrl','CO2']
plat = '/gfdl.ncrc3-default-repro/'
var = 'ucomp'
varo = 'ucomp'
npert = np.size(pert)
#npert = 1
data = np.zeros([npert,30,64])
for i in range(npert):
    fs = []
    filedir = basedir+exper+pert[i]+plat+'history/'
    yrC = '00040101'
    #tmpZon = np.zeros([nmon,nlev-1,nlat])
    #phalfZon = np.zeros([nmon,nlev,nlat])
    filename = filedir+yrC+'.atmos_month.nc'
    fs.append(nc.netcdf_file(filename,'r',mmap=True))
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    pfull = fs[-1].variables['pfull'][:].astype(np.float64)
    tmp = fs[-1].variables[var][:].astype(np.float64) #t,p,lat,lon
    #tmp[np.where(tmp<-999)] = np.nan
    tmp = np.mean(tmp,3)
    data[i,:,:] = np.mean(tmp,0)
    fs[-1].close()
    
lattick = np.arange(-90,91,30)
ptick = np.array([0,0.2,0.4,0.6,0.8,1.0])*1000
sc = 1    
cmap = plt.cm.RdYlBu_r
clev = np.arange(-40,41,10)
clev = clev[np.where(clev!=0)]
#clev = np.array([-30,-20,10,10,20,30])
#clev = np.array([-20,-10,-1,0,1,10,20])
clev1 = np.arange(-3,3.1,0.5)/sc
#clev1 = np.array([-2,-1,-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,0,\
#                 0.01,0.02,0.05,0.1,0.2,0.5,1,2])*1e-5
norm = mpl.colors.BoundaryNorm(clev1,cmap.N)
x = lat
y = pfull
xtick = lattick
ytick = ptick
#clabel = 'Temperature (K)'
clabel = 'Zonal wind (m s$^{-1}$)'
for i in [1]:
    #plt.subplot(3,1,i)
    plt.figure()
    cs = plt.contour(x,y,data[0,:,:]*sc,\
                     clev,\
                     linewidths=1,colors='k')
    plt.clabel(cs,fontsize=16,fmt='%1.0f',inline=1)
    cs = plt.contour(x,y,data[0,:,:]*sc,\
                     [0],\
                     linewidths=2,colors='k')
    diff = data[i,:,:]-data[0,:,:]
    plt.contourf(x,y,diff,\
                 clev1,norm=norm,\
                 cmap=cmap,extend='both')
    cb = plt.colorbar(orientation='horizontal',pad=0.2)
    cb.formatter.set_powerlimits((0,0))
    #cb.ax.yaxis.set_offset_position('right')
    cb.update_ticks()
    cb.set_label(clabel)
    plt.gca().invert_yaxis()
    plt.xticks(xtick)
    plt.xlabel('Latitude')    
    plt.yticks(ytick)
    plt.ylabel('Pressure (hPa)')
    plt.tight_layout()
    
