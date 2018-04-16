# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/PythonScripts')
import numpy as np
from scipy import stats as st
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
g = 9.80
Rad = 6371.0e3
outdir = '/home/z1s/research/nonlinear/npz/SM2/asym/'
outdir_sub = 'ts/DJF/'
pert = ['ctrl','m0.25latc25lonc100260']
npert = np.size(pert)
#npert = 4
var = 'temp_zm'
clabel = 'temp (K)'
#clabel = 'Zonal wind (m s$^{-1}$)'
sc = 1
clev = np.arange(-30,31,10)
clev1 = np.arange(-1,1.1,0.1)/sc
diag = 'var3dp'
time = '0761-0860.'
filename = outdir+'dim.'+pert[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
pfull = npz['level']
npfull = np.size(pfull)
tmp = np.zeros([npert,npfull,nlat])
for i in range(npert):
    filename = outdir+outdir_sub+diag+'.'+time+pert[i]+'.npz'
    npz = np.load(filename)
    #tmp[i,:,:,:] = npz[var] #pert,yr,p,lat
    #tmp[i,:,:,:] = 0.5*(npz[var][:,1:,:]+npz[var][:,:-1,:])
    tmp[i,:,:] = np.mean(npz[var][20:,...],0)
    #theta = npz['theta_zm']
    #tmp[i,:,:][16:,:] = theta[16:,:]
    #tmp[i,:,:][:,21:42] = theta[:,21:42]

ind = range(1,npert)
#ind=[1]
xsc = 2*np.pi*Rad*np.cos(lat/90*np.pi/2)
xsc.shape = [1,1,nlat]
#tmpm = tmpm*xsc*100/g
#u300 = ucomp[:,:,6,:]
#%%
latS = 0
latN = 90
plev = np.arange(1,25,1)
montick = np.arange(1,13,1)
lattick = np.arange(-90,91,30)
ptick = np.array([0,0.2,0.4,0.6,0.8,1.0])*1000

data = tmp[0,:,:]
cmap = plt.cm.RdYlBu_r
clev = clev[np.where(clev!=0)]
#clev = np.array([-30,-20,10,10,20,30])
#clev = np.array([-20,-10,-1,0,1,10,20])

#clev1 = np.array([-2,-1,-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,0,\
#                 0.01,0.02,0.05,0.1,0.2,0.5,1,2])*1e-5
norm = mpl.colors.BoundaryNorm(clev1,cmap.N)
x = lat
y = pfull
xtick = lattick
xlim = [-90,90]
ytick = ptick

"""
plt.figure()
plt.contourf(x,y,data*sc,\
             clev,\
             cmap=cmap,extend='both')
plt.gca().invert_yaxis()
plt.colorbar()
plt.xticks(xtick)
plt.yticks(ytick)
plt.tight_layout()
"""
#fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True)
for i in []:
    #plt.subplot(3,1,i)
    plt.figure()
    cs = plt.contour(x,y,tmp[i,:,:],\
                     clev,\
                     linewidths=1,colors='k')
    plt.clabel(cs,fontsize=16,fmt='%1.0f',inline=1)
    cs = plt.contour(x,y,tmp[i,:,:],\
                     [0],\
                     linewidths=2,colors='k')
    plt.gca().invert_yaxis()
    plt.xticks(xtick)
    plt.xlim(xlim)
    plt.xlabel('Latitude')    
    plt.yticks(ytick)
    #plt.yscale('log')
    plt.ylabel('Pressure (hPa)')
    plt.title(clabel+'   '+pert[i])
    plt.tight_layout()
for i in range(1,npert):
    #plt.subplot(3,1,i)
    plt.figure()
    cs = plt.contour(x,y,data*sc,\
                     clev,\
                     linewidths=1,colors='k')
    plt.clabel(cs,fontsize=16,fmt='%1.0f',inline=1)
    cs = plt.contour(x,y,data*sc,\
                     [0],\
                     linewidths=2,colors='k')
    if (i<npert):    
        diff = np.squeeze(tmp[i,:,:]-tmp[0,:,:])
    if (i==npert):
        diff = np.squeeze(tmp[1,:,:]+tmp[2,:,:]-2*tmp[0,:,:])
    #diff[sigxy1] = np.nan
    plt.contourf(x,y,diff,\
                 clev1,norm=norm,\
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
    plt.plot(x,tmp[i,-1,:],color=color[i],ls=ls[i])
plt.legend(['LH0','LH0.5','LH1'],loc='best',ncol=3)
plt.xticks(xtick)
plt.xlim(xlim)
plt.xlabel('Latitude')
plt.ylabel('Zonal wind at 325 hPa (m/s)')
plt.tight_layout()
