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
mpl.rc('figure', figsize=[4*4./3,4])

CP_AIR = 1004.64
GRAV = 9.80
LE = 2.500e6
"""
basedir = '/home/z1s/research/hydro/npz/p1l/'
pert = ['2000','4xCO2']
"""
outdir = '/home/Zhaoyi.Shen/research/landprecip/npz/'
outdir_sub = 'ts/JJA/'
pert = ['UW','UW+2K']
npert = np.size(pert)
ens = ['']
nens = np.size(ens)
sim = []
yr = np.arange(1870,1975,1)
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
var1 = 'uq_bottom'
var2 = 'vq_bottom'
sc = 1e3
diag = 'qflx2d'
time = '1983-2012'
filename = outdir+'dim.'+sim[0]+'.npz'
npz = np.load(filename)
lat = npz['lat']
nlat = np.size(lat)
lon = npz['lon']
nlon = np.size(lon)
tmp = np.zeros((nsim,2,nlat,nlon))
for i in range(nsim):
    filename = outdir+outdir_sub+diag+'.'+time+'.'+sim[i]+'.npz'
    npz = np.load(filename)
    tmp[i,0,:,:] = np.mean(npz[var1],0)-tmp[0,0,:,:]
    tmp[i,1,:,:] = np.mean(npz[var2],0)-tmp[0,1,:,:]
    #tmp[i,:,:] = npz['shflx']+npz['precip']*LE
speed = np.sqrt(tmp[:,0,:,:]**2+tmp[:,1,:,:]**2)
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

latt,lont,tmpt = pp.grid_for_map(lat,lon,tmp)
latt,lont,speedt = pp.grid_for_map(lat,lon,speed)

tmpt = tmpt*sc
clabel = "t_surf (K)"
m = Basemap(projection='mill',\
            fix_aspect=True,\
            llcrnrlat=20,urcrnrlat=55,llcrnrlon=60,urcrnrlon=140)
cmap = plt.cm.RdYlBu_r
clev = np.arange(-0.1,0.11,0.01)
lons,lats = np.meshgrid(lont,latt)
x,y = m(lons,lats)
yy = np.arange(0,len(latt),1)
xx = np.arange(0,len(lont),1)
points = np.meshgrid(yy,xx)

plt.figure()
m.drawcoastlines()
m.drawcountries()
m.contourf(x,y,speedt[1,:,:],clev,cmap=cmap,extend='both')
cb = m.colorbar(location='bottom',size='5%',pad='8%')
Q = m.quiver(x[points],y[points],tmpt[1,0,:,:][points],tmpt[1,1,:,:][points],scale=500)
qk = plt.quiverkey(Q,0.8,1.05,20,'20 m s$^{-1}$ g kg$^{-1}$')
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
