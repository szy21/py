# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import postprocess as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
fsize = 14
font = {'size'   : fsize}
mpl.rc('font', size=fsize)
mpl.rc('lines', linewidth=2)
mpl.rc('figure', figsize=[4.5*4./3,4.5])

CP_AIR = 1004.64
GRAV = 9.80
LE = 2.500e6
"""
basedir = '/home/z1s/research/hydro/npz/p1l/'
pert = ['2000','4xCO2']
"""
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz/perfect_model/'
outdir_sub = 'ts/monthly/'
pert = ['CM2','AM2']
npert = np.size(pert)
ens = ['']
nens = np.size(ens)
sim = []
yr = np.arange(1870,1975,1)
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
var = 't_surf'
sc = 1
diag = 'var2d'
time = '196901-200012'
filename = outdir+'dim.'+sim[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
lon = npz['lon']
nlon = np.size(lon)
land_mask = npz['land_mask']
nf = 2
ntime = 384
tmp = np.zeros([nf,ntime,nlat,nlon])
for i in range(nf):
    filename = outdir+outdir_sub+diag+'.'+time+'.'+sim[i]+'.npz'
    npz = np.load(filename)
    tmp[i,:,:,:] = npz[var]
    ice_mask = npz['ice_mask']
    tmp[i,...][np.where(ice_mask>0)] = np.nan
    #tmp[i,:,:] = npz['shflx']+npz['precip']*LE
#%%
"""
lattick = np.arange(-90,91,30)
x = lat
xtick = lattick
plt.figure()
for i in range(1,nsim):
    plt.plot(x,tmp[i,:]-tmp[0,:])
plt.legend(['c25','c30'],fontsize=fsize,loc='best')
plt.ylabel('swdn_toa (W m$^{-2}$)')
plt.xticks(lattick)
plt.ylim([-30,5])
plt.tight_layout()
#%%

tmpzon = np.mean(tmp,2)
weight = np.cos(lat*np.pi/180.)
weight1 = weight/np.sum(weight)
weight1.shape = [1,nlat]
tmpglb = np.sum(tmpzon*weight1,1)
"""
#%%
tmp = tmp-tmp[0,:,:,:]
tmp = np.min(tmp,1)
tmp[:,np.where(land_mask>0)[0],np.where(land_mask>0)[1]] = np.nan
latt,lont,tmpt = pp.grid_for_map(lat,lon,tmp)

data = tmpt*sc
clabel = "Temperature difference (K)"
m = Basemap(projection='mill',\
            fix_aspect=True,\
            llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
cmap = plt.cm.RdYlBu_r
clev = np.arange(-1,1.1,0.2)
lons,lats = np.meshgrid(lont,latt)
x,y = m(lons,lats)

plt.figure()
m.drawcoastlines()
m.contourf(x,y,data[1,:,:],clev,cmap=cmap,extend='both')
cb = m.colorbar(location='bottom',size='5%',pad='8%')
cb.set_label(clabel)
plt.tight_layout()
"""
clev1 = np.arange(-2,2.1,0.2)
cmap = plt.cm.RdYlBu_r
for i in [1,2]:
    plt.figure()
    m.drawcoastlines()
    x,y = m(lons,lats)
    #cs = m.contour(x,y,data[0,:,:],clev,linewidths=1,colors='k')
    #plt.clabel(cs,fontsize=12,fmt='%1.0f',inline=1)
    m.contourf(x,y,data[i,:,:]-data[0,:,:],clev1,cmap=cmap,extend='both')
    cb = m.colorbar(location='bottom',size='5%',pad='8%')
    cb.set_label(clabel)
    plt.tight_layout()
"""
#%%
