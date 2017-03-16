# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib/')
import amgrid as grid
import postprocess as pp
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
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
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz/perfect_model/'
outdir_sub='ts/monthly/'
diff = True
region = 'global'
latlim_dict = {'EuNAf':[[25,65],[25,65]],
               'NAm':[[15,55]],
               'EAsSAs':[[15,50],[5,35]],
               'EAs':[[15,50]],
               'global':[[-90,90]],
               'tropics':[[-30,30]]}
lonlim_dict = {'EuNAf':[[0,50],[350,360]],
               'NAm':[[235,300]],
               'EAsSAs':[[95,160],[50,95]],
               'EAs':[[95,160]],
               'global':[[0,360]],
               'tropics':[[0,360]]}
latlim = latlim_dict[region]
lonlim = lonlim_dict[region]
nlim = np.shape(latlim)[0]
pert = ['CM2','AM2']
npert = np.size(pert)
fcolor=['black','red','blue','green']
ens = ['']
#ens= ['ext','ext2','ext3']
nens = np.size(ens)
sim = []
time = '196901-200012'
ntime = 384
interval = 12
yr = np.arange(1969,2001,1)
nyr = np.size(yr)
xticks = np.arange(1970,2001,5)
ylim = [280,285]
ind = [1]
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
var = 't_surf'
diag = 'var2d'
filename = outdir+'dim.'+sim[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
lon = npz['lon']
nlon = np.size(lon)
land_mask = np.zeros((ntime,nlat,nlon))
for i in range(ntime):
    land_mask[i,...] = 1-npz['land_mask']
land_mask[np.where(land_mask<1)] = 0
area = grid.calcGridArea(lat,lon)
#land_mask[:,:,:] = 1
region_mask = np.zeros((ntime,nlat,nlon))
for i in range(nlim):
    mask = grid.regionMask(lat,lon,latlim[i],lonlim[i])
    for ti in range(ntime):
        region_mask[ti,...][np.where(mask==1)] = land_mask[ti,...][np.where(mask==1)]
land_mask = region_mask.copy()
area.shape = [1,nlat,nlon]
#land_mask.shape = [1,nlat,nlon]
data = np.zeros([npert,nens,ntime])
for i in range(npert):
    for j in range(nens):
        filename = outdir+outdir_sub+diag+'.'+time+'.'+pert[i]+ens[j]+'.npz'
        npz = np.load(filename)
        ice_mask = 1-npz['ice_mask']
        #ice_mask[:,:,:] = 1
        land_mask[np.where(ice_mask<1)] = 0
for i in range(npert):
    for j in range(nens):
        filename = outdir+outdir_sub+diag+'.'+time+'.'+pert[i]+ens[j]+'.npz'
        npz = np.load(filename)
        tmp = npz[var]
        tmp = tmp*area*land_mask
        tmp = np.sum(tmp,-1)
        tmp = np.sum(tmp,-1)
        tmp = tmp/np.sum(np.sum(area*land_mask,-1),-1)
        data[i,j,:] = tmp
#%%
if diff:
    data = data-data[0,:,:]
data_em = np.mean(data,1)

data_mean = np.zeros((2,ntime-interval+1))
x = np.linspace(yr[0],yr[-1],ntime)
x_mean = np.zeros(ntime-interval+1)
for i in range(ntime-interval+1):
        data_mean[:,i] = np.mean(data_em[:,i:i+interval],1)
        x_mean[i] = np.mean(x[i:i+interval])
z = np.zeros((npert,2))


"""
m = Basemap(projection='mill',\
            fix_aspect=True,\
            llcrnrlat=0,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
plt.figure()
m.drawcoastlines()
latt,lont,mask_tr = pp.grid_for_map(lat,lon,land_mask)
lons,lats = np.meshgrid(lont,latt)
x1,y1 = m(lons,lats)
m.contourf(x1,y1,mask_tr[0,:,:],corner_mask=False)

"""
plt.figure()
for i in ind:
    y = data_mean[i,:]
    x = x_mean
    plt.plot(x,y,color=fcolor[0])
    z[i,:] = np.polyfit(x,y,1)
    #plt.plot(x,z[i,0]*x+z[i,1],'--',color=fcolor[i])
    plt.ylabel('Temperature difference (K)')
    plt.xticks(xticks)
    plt.xlim(yr[0],yr[-1]) 
    #plt.ylim(ylim)
    plt.tight_layout()
#plt.legend(['coupled','uncoupled'],loc='best',fontsize=fsize)
print z
print np.mean(y)
#%%


