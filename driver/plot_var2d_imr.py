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

basedir = '/home/z1s/research/nonlinearity/analysis/npz/ts/'
pert = ['ctrl','CO2',\
        'm2c25w30','m2c30w30','m2c35w30','m2c40w30','m2c45w30','m2c50w30',\
        'CO2+m2c35w30','CO2+m2c40w30','CO2+m2c45w30',\
        'm3c35w30','m3c40w30']
#pert = ['ctrl','CO2','m2c35w30','CO2+m2c35w30']
npert = np.size(pert)
#npert = 4
var = 'tsfc'
diag = 'var2d'
filename = basedir+'dim.'+pert[0]+'_sigma.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
ntime = 72
nyr = 6
tmp = np.zeros([npert,ntime,nlat])
tsfc = np.zeros([npert,ntime,nlat])
rain = np.zeros([npert,ntime,nlat])
ind = [0,1,2,3,4,5,6,7]
for i in ind:
    filename = basedir+diag+'.'+pert[i]+'_sigma.npz'
    npz = np.load(filename)
    tsfc[i,:,:] = npz['tsfc']
    rain[i,:,:] = npz['rain_ls']+npz['rain_cv']
    #tmp[i,:,:] = npz[var]

weight = np.cos(lat*np.pi/180.)
weight1 = weight/np.sum(weight)
weight1.shape = [1,1,nlat]
tmpglb = np.sum(tmp*weight1,2)
tsfcglb = np.sum(tsfc*weight1,2)
rainglb = np.sum(rain*weight1,2)
tsfcglb1 = np.zeros([npert,nyr])
rainglb1 = np.zeros([npert,nyr])
for i in range(nyr):
    tsfcglb1[:,i] = np.mean(tsfcglb[:,i*12:(i+1)*12],1)
    rainglb1[:,i] = np.mean(rainglb[:,i*12:(i+1)*12],1)
sc = 1
z = np.zeros([npert,2])
for i in [2,6,7]:
    plt.plot((tsfcglb[i,:]-tsfcglb[0,:])*sc)
plt.figure()   
for i in [1,7]:
    x = tsfcglb1[i,:]-tsfcglb1[0,:]
    y = rainglb1[i,:]-rainglb1[0,:]
    plt.plot(x,y,'.')
    z[i,:] = np.polyfit(x,y,1)

#%%
"""
lattick = np.arange(-60,61,30)
x = np.sin(lat/180.*np.pi)
xtick = np.sin(lattick/180.*np.pi)
xticklabel = lattick

lattick = np.arange(-90,91,30)
x = lat
xtick = lattick
xticklabel = lattick

xlim = [-90,90]
ylabel = 'Temperature (K)'
#ylabel = 'Zonal wind (m s$^{-1}$)'
ind = range(1,npert+1)

leg = []
#for i in range(npert):
#    leg.append(pert[i][3:5])

plt.figure()
plt.plot(x,tmp[0,:]*sc)
plt.xticks(xtick,xticklabel)
plt.xlabel('Latitude')
plt.ylabel(ylabel)
plt.tight_layout()

leg = ['CO2','c35','both','add']
plt.figure()
for i in ind:
    if (i<npert):    
        diff = np.squeeze(tmp[i,:]-tmp[0,:])
    if (i==npert):
        diff = np.squeeze(tmp[1,:]+tmp[2,:]-2*tmp[0,:])
    #diff[sigxy1] = np.nan
    plt.plot(x,diff*sc)
plt.xticks(xtick,xticklabel)
#plt.xlim(xlim)
#plt.ylim([-1.2,1.2])
plt.xlabel('Latitude') 
plt.ylabel(ylabel)
plt.legend(leg[:],fontsize=fsize,ncol=2,loc='best',\
           handletextpad=0,labelspacing=0,columnspacing=0.5)
plt.tight_layout()
"""
#%%

