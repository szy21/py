# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s

need to fix this!
"""

import sys
sys.path.append('/home/z1s/py/lib/')
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
"""
basedir = '/home/z1s/research/nonlinearity/npz/imr/ts/'
pert = ['ctrl','CO2',\
        'm2c25w30','m2c30w30','m2c35w30','m2c40w30','m2c45w30','m2c50w30',\
        'CO2+m2c35w30','CO2+m2c40w30','CO2+m2c45w30',\
        'm3c35w30','m3c40w30']
"""
model = ['AM2.1n','AM3n']
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz/AM3n/'
flag = 'JJA'
flag_dict = {'annual':'annual','MAM':'MAM','JJA':'JJA','SON':'SON','DJF':'DJF'}
outdir_sub='ts/'+flag_dict[flag]+'/'
NH_dict = {'annual':False,'MAM':True,'JJA':True,'SON':True,'DJF':True}
NH = NH_dict[flag]
pert = ['SST','all','WMGG','aero']
xval_dict = {'annual':0, 'MAM':0.5, 'JJA':1.0, 'SON':1.5, 'DJF':2.0}
xval = xval_dict[flag]
npert = np.size(pert)
fcolor=['black','red','blue','green']
ens = ['_A1','_A2','_A3','_A4','_A5']
#ens= ['ext','ext2','ext3']
nens = np.size(ens)
sim = []
yr = np.arange(1870,2016,1)
yr_ref = [1961,1990]
yr_trd = np.arange(1973,2016,1)
trd_start = yr_trd[0]-yr[0]
nyr = np.size(yr)
nyr_trd = np.size(yr_trd)
time = '1870-2015'
model_start = yr[0]-np.int(time[:4])
xticks = np.arange(1975,2016,10)
ylim = [-1.5,1.5]
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
var = 't_ref'
diag = 'var2d'
filename = outdir+'dim.'+sim[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
lon = npz['lon']
nlon = np.size(lon)
land_mask = npz['land_mask']
area = grid.calcGridArea(lat,lon)
#land_mask[np.where(land_mask<1)] = 0
if NH:
    land_mask[np.where(lat<0),:] = 0
#land_mask[:,:] = 1
area.shape = [1,nlat,nlon]
land_mask.shape = [1,nlat,nlon]
data = np.zeros([npert,nens,nyr])
for i in range(npert):
    for j in range(nens):
        filename = outdir+outdir_sub+diag+'.'+time+'.'+pert[i]+ens[j]+'.npz'
        npz = np.load(filename)
        tmp = npz[var][model_start:model_start+nyr,...]
        tmp = tmp*area*land_mask
        tmp = np.sum(tmp,-1)
        tmp = np.sum(tmp,-1)
        tmp = tmp/np.sum(area*land_mask)
        data[i,j,:] = tmp
#%%
yrind = np.where((yr>=yr_ref[0]) & (yr<=yr_ref[1]))
sc = 1
ref = np.mean(data[:,:,yrind],-1)
data_ano = data-ref
data_em = np.mean(data,1)
ref_em = np.mean(data_em[:,yrind],-1)
data_em_ano = data_em-ref_em
z = np.zeros((nens,2))
data_ano_max = np.zeros((npert,nyr))
data_ano_min = np.zeros((npert,nyr))
for i in range(npert):
    for k in range(nyr):
        data_ano_max[i,k] = np.max(data_ano[i,:,k])
        data_ano_min[i,k] = np.min(data_ano[i,:,k])
z = np.zeros((npert,2))
ind = range(npert)
x = yr_trd

#plt.figure()
for i in range(npert):
    for j in range(nens):
        y = data_ano[i,j,trd_start:trd_start+nyr_trd]
        z[i,:] = np.polyfit(x,y,1)
        #z[i,:] = st.theilslopes(y,x) #theil:return 4 stats
        plt.scatter(xval+i/10.,z[i,0],color=fcolor[i])
        print z[i,0]
        
plt.ylabel('Trend (K per yr)')

plt.tight_layout()
#plt.ylim(ylim)
#%%


