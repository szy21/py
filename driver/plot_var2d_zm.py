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
outdir = '/home/z1s/research/nonlinear/npz/SM2/'
outdir_sub = 'ts/annual/'
pert = ['ctrl','m2c00','2xCO2','2xCO2+m2c00']
npert = np.size(pert)
#npert = 4
var = 'temp_zm'
diag = 'var3d'
time = '0761-0860'
leg = ['c00','CO2','both','add']
filename = outdir+'dim.'+pert[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
pfull = npz['pfull']
npfull = np.size(pfull)
#yr = npz['year']
#nyr = np.size(yr)
tmp = np.zeros([npert+1,100,nlat])
tmp_mean = np.zeros([npert+1,nlat])
tmp_var = np.zeros([npert+1,nlat])
for i in range(npert+1):
    #tmp[i,:,:,:] = npz[var] #pert,yr,p,lat
    #tmp[i,:,:,:] = 0.5*(npz[var][:,1:,:]+npz[var][:,:-1,:])
    if (i<npert):
        filename = outdir+outdir_sub+diag+'.'+time+'.'+pert[i]+'.npz'
        npz = np.load(filename)
        tmp[i,:,:] = npz[var][:,-1,:]
    if (i==npert):
        tmp[i,:,:] = tmp[1,:,:]+tmp[2,:,:]-tmp[0,:,:]
    tmp_mean[i,:] = np.mean(tmp[i,20:,:],0)
    tmp_var[i,:] = np.var(tmp[i,20:,:],0)
tmp_ano_mean = np.zeros([npert+1,nlat])
tmp_ano_std = np.zeros([npert+1,nlat])
tmp_ano_se = np.zeros([npert+1,nlat])
for i in range(npert+1):
    if (i<npert):
        diff = (tmp[i,:,:]-tmp[0,:,:])
    if (i==npert):
        diff = (tmp[1,:,:]+tmp[2,:,:]-2*tmp[0,:,:])
    tmp_ano_mean[i,:] = np.mean(diff[20:,:],0)
    tmp_ano_std[i,:] = np.std(diff[20:,:],0)
    tmp_ano_se[i,:] = np.sqrt(tmp_var[0,:]/80+tmp_var[i,:]/80)
    
#%%
#data = np.transpose(data)
sc = 1
x = lat
lattick = np.arange(-60,61,30)
xtick = lattick
xlim = [-90,90]
xticklabel = lattick
ylabel = 'Temperature (K)'
#ylabel = 'Zonal wind (m s$^{-1}$)'

ind = range(1,npert+1)
plt.figure()
for i in ind:
    plt.errorbar(x,tmp_ano_mean[i,:],tmp_ano_se[i,:])
    
#plt.gca().invert_yaxis()
plt.xticks(xtick)
plt.xlim(xlim)
plt.ylim([-3,11])
plt.xlabel('Latitude')
plt.ylabel(ylabel)
plt.legend(leg,ncol=2,fontsize=fsize,loc='best',\
           handletextpad=0,labelspacing=0,columnspacing=0.5)
plt.tight_layout()

#%%