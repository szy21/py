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

RADIUS = 6371.0e3

outdir = '/home/z1s/research/HS_michelle/npz/305K/'
outdir_sub = 'av/'
pert = ['s24','s25']
npert = np.size(pert)
#npert = 4
var = 'vLqTot_col_zm'
ylabel = 'Moisture flux (PW)'
diag = 'eflx2d'
time = ''
legend = ['s24','s25']
filename = outdir+'dim.'+pert[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
#pfull = npz['pfull']
#npfull = np.size(pfull)
#yr = npz['year']
#nyr = np.size(yr)
tmp = np.zeros((npert,nlat))
for i in range(npert):
    filename = outdir+outdir_sub+diag+'.'+time+pert[i]+'.npz'
    npz = np.load(filename)
    tmp[i,:] = npz[var]    
#%%
#data = np.transpose(data)
sc = 1e-15
xsc = 2*np.pi*RADIUS*np.cos(lat/180.*np.pi)
x = lat
lattick = np.arange(-60,61,30)
xtick = lattick
xlim = [-90,90]
xticklabel = lattick

#ylabel = 'Zonal wind (m s$^{-1}$)'

ind = range(npert)
color = ['k','b','r','b','r']
ls = ['-','-','-','--','--']
plt.figure()
for i in ind:
    plt.plot(x,tmp[i,:]*xsc*sc,color=color[i],ls=ls[i])
    
#plt.gca().invert_yaxis()
plt.xticks(xtick)
plt.xlim(xlim)
#plt.ylim([-3,11])
plt.xlabel('Latitude')
plt.ylabel(ylabel)
plt.legend(legend[:3],fontsize=fsize,loc='best')
plt.tight_layout()

#%%