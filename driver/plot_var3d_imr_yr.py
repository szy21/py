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

basedir = '/home/z1s/research/nonlinearity/analysis/npz/'
pert = ['ctrl','CO2','m2c40w30','CO2+m2c40w30']
npert = np.size(pert)
npert = 4
var = 'temp'
diag = 'var3d'
filename = basedir+'dim.'+pert[0]+'_sigma.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
pfull = npz['pfull']
npfull = np.size(pfull)
tmp = np.zeros([npert,9,npfull,nlat])
for i in range(npert):
    filename = basedir+'1yr/'+diag+'.'+pert[i]+'_sigma.npz'
    npz = np.load(filename)
    #tmp[i,:,:,:] = npz[var] #pert,yr,p,lat
    #tmp[i,:,:,:] = 0.5*(npz[var][:,1:,:]+npz[var][:,:-1,:])
    tmp[i,:,:,:] = npz[var]

tmpm = np.mean(tmp[:,6:,:,:],1)
tmpm = tmp[:,7,:,:]
ind = range(1,npert)
ind = [1]
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

data = tmpm[0,:,:]
sc = 1
cmap = plt.cm.RdYlBu_r
clev = np.arange(-40,41,10)
clev = clev[np.where(clev!=0)]
#clev = np.array([-30,-20,10,10,20,30])
#clev = np.array([-20,-10,-1,0,1,10,20])
clev1 = np.arange(-2,2.1,0.25)/sc
#clev1 = np.array([-2,-1,-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,0,\
#                 0.01,0.02,0.05,0.1,0.2,0.5,1,2])*1e-5
norm = mpl.colors.BoundaryNorm(clev1,cmap.N)
x = lat
y = pfull
xtick = lattick
xlim = [-60,60]
ytick = ptick
#clabel = 'Temperature (K)'
clabel = 'Zonal wind (m s$^{-1}$)'
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
 
#fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True)
"""
for i in [2]:
    #plt.subplot(3,1,i)
    plt.figure()
    cs = plt.contour(x,y,data*sc,\
                     clev,\
                     linewidths=1,colors='k')
    plt.clabel(cs,fontsize=16,fmt='%1.0f',inline=1)
    cs = plt.contour(x,y,data*sc,\
                     [0],\
                     linewidths=2,colors='k')
    if (i<4):    
        diff = np.squeeze(tmpm[i,:,:]-tmpm[0,:,:])
    if (i==4):
        diff = np.squeeze(tmpm[1,:,:]+tmpm[2,:,:]-2*tmpm[0,:,:])
    #diff[sigxy1] = np.nan
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
    plt.xlim(xlim)
    plt.xlabel('Latitude')    
    plt.yticks(ytick)
    #plt.yscale('log')
    plt.ylabel('Pressure (hPa)')
    plt.tight_layout()

#%%

data = tmpm[:,15,:]
#data = np.transpose(data)
cmap = plt.cm.RdYlBu_r
clev = np.arange(0,31,10)
clev1 = np.arange(-3.5,3.6,0.5)
x = lat
xtick = lattick
xlabel = 'zonal wind (m s$^{-1}$)'
ind = [1,2,3,4]
leg = ['CO2','solar','both','add']
#plt.figure()
plt.figure()
for i in ind:
    if (i<4):
        diff = (data[i,:]-data[0,:])*sc
    if (i==4):
        diff = (data[1,:]+data[2,:]-2*data[0,:])*sc
    plt.plot(x,diff)
    
#plt.gca().invert_yaxis()
plt.legend(leg,ncol=2,fontsize=fsize)
plt.xticks(xtick)
plt.xlim(xlim)
plt.ylim([-3,3])
plt.xlabel(xlabel)
plt.tight_layout()
#%%
"""
#%%
plt.figure()
plt.contourf(x,y,tmp[0,:,:]-np.fliplr(tmp[0,:,:]))
plt.ylim([0,1000])
plt.gca().invert_yaxis()
plt.colorbar()
#%%
"""