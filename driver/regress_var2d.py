# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import amgrid as grid
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

CP_AIR = 1004.64
GRAV = 9.80
RAD = 6371.0e3
LE = 2.5e6

outdir = '/home/z1s/research/nonlinear/npz/SM2/'
outdir_sub = 'ts/annual/'
pert = ['ctrl','2xCO2','m6c35']
npert = np.size(pert)
var1 = 't_ref_zm'
var2 = 'netrad_toa_zm'
diag = 'var2d'
time = '0761-0860'
filename = outdir+'dim.'+pert[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
ntime = 100
tmp = np.zeros([npert,ntime,nlat])
t_ref = np.zeros([npert,ntime,nlat])
precip = np.zeros([npert,ntime,nlat])
netrad_toa = np.zeros([npert,ntime,nlat])
ind = range(npert)
for i in ind:
    filename = outdir+outdir_sub+diag+'.'+time+'.'+pert[i]+'.npz'
    npz = np.load(filename)
    t_ref[i,:,:] = npz[var1]
    netrad_toa[i,:,:] = npz[var2]

area = grid.calcGridArea(lat,1)
area.shape = [1,1,nlat]

t_ref_gm = np.sum(t_ref*area,2)/np.sum(area)
precip_gm = np.sum(precip*area,2)/np.sum(area)
netrad_toa_gm = np.sum(netrad_toa*area,2)/np.sum(area)
#tmp_gm = np.sum(tmp*weight1,2)

sc = 1
z = np.zeros([npert,2])

plt.figure()   
for i in range(1,npert):
    x = (t_ref_gm[i,:]-t_ref_gm[0,:])[:20]
    y = (netrad_toa_gm[i,:]-netrad_toa_gm[0,:])[:20]
    plt.scatter(x,y)
    z = np.polyfit(x,y,1)
    print z
    plt.plot(x,z[0]*x+z[1])

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

