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
outdir = '/home/z1s/research/BC-circulation/analysis/npz/idealized/ulm/'
outdir_sub = ''
pert = ['ctrl','Ty90',\
        'l8_1x_Ty60','l8_1x_Ty90']
npert = np.size(pert)
#npert = 4
var = 'temp_zm'
diag = 'var3d'
filename = outdir+'dim.'+pert[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
pfull = npz['pfull']
npfull = np.size(pfull)
tmp = np.zeros([npert,npfull,nlat])
for i in range(npert):
    filename = outdir+outdir_sub+diag+'.'+pert[i]+'.npz'
    npz = np.load(filename)
    #tmp[i,:,:,:] = npz[var] #pert,yr,p,lat
    #tmp[i,:,:,:] = 0.5*(npz[var][:,1:,:]+npz[var][:,:-1,:])
    tmp[i,:,:] = npz[var]

yind = np.concatenate((np.arange(10,21,1),np.arange(43,54,1)))   
weight = np.cos(lat[yind]/90.*np.pi/2)
weight1 = weight/np.sum(weight)
weight1.shape = [1,1,np.size(yind)]
tmp_rm = np.sum(tmp[:,:,yind]*weight1,2)

xsc = 2*np.pi*Rad*np.cos(lat/90*np.pi/2)
xsc.shape = [1,1,nlat]
#%%
latS = 0
latN = 90
plev = np.arange(1,25,1)
montick = np.arange(1,13,1)
lattick = np.arange(-90,91,30)
ptick = np.array([0,0.2,0.4,0.6,0.8,1.0])*1000

sc = 1
x = lat
y = pfull
xtick = lattick
xlim = [-90,90]
ytick = ptick
xlabel = 'Temperature (K)'
ylabel = 'Pressure (hPa)'
#clabel = 'Zonal wind (m s$^{-1}$)'
ind = range(2,npert)
#fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True)
color = ['k','r','g','b','k','r','g','b','r','g']
legend = np.array(['ctrl','rot2x','ctrl','rot2x'])
data = tmp_rm
data = 0.5*(data[:,:-1]+data[:,1:])
y = pfull[:]
y = 0.5*(y[:-1]+y[1:])
ylim = [100,1000]
yticks = [200,400,600,800,1000]
xticks = [0,4,8,12,16]
"""
plt.figure(figsize=[4.5,4.5])
for i in range(npert):
    plt.plot(data[i,:],y,color=color[i])
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.legend(legend[0:npert],loc='best',fontsize=fsize)
plt.xlabel(xlabel)#,labelpad=20)
plt.ylabel(ylabel)
plt.ylim(ylim)
plt.xlim([280,380])
plt.yticks(yticks)
plt.gca().invert_yaxis()
plt.tight_layout()
"""
plt.figure(figsize=[4.5,4.5])
for i in ind:
    plt.plot(data[i,:]-data[i-2,:],y[:],color=color[i])
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(legend[ind],loc='best',fontsize=fsize)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim(ylim)
#plt.xlim([-3e-4,3e-4])
plt.xlim([-2,10])
plt.yticks(yticks)
#plt.xticks(xticks)
plt.gca().invert_yaxis()
plt.tight_layout()

