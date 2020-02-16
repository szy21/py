# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import amgrid as grid
import postprocess as pp
import numpy as np
from scipy.io import netcdf as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
fsize = 14
font = {'size'   : fsize}
mpl.rc('font', size=fsize)
mpl.rc('lines', linewidth=2)
mpl.rc('figure', figsize=[6*4./3,6])
mpl.rcParams['contour.corner_mask'] = False

region = ['EuNAf','NAm','EAs','SAs']
#region = ['tsEU','tsEA','tsNA']
#region = ['largeEU','largeEA']
nregion = np.size(region)

land = True

filename = '/home/z1s/research/climate/atmos_level.static.nc'
fs = nc.netcdf_file(filename,'r',mmap=True)
lat = fs.variables['lat'][:]
nlat = np.size(lat)
lon = fs.variables['lon'][:]
nlon = np.size(lon)
land_mask1 = fs.variables['land_mask'][:]
fs.close()
land_mask = land_mask1.copy()
if not land:
    land_mask[:,:] = 1
region_mask = -np.ones((nlat,nlon))
region_mask_obs = -np.ones((nlat,nlon))

latt,lont,land_mask_obs = pp.grid_for_map(lat,lon,land_mask)


for regi in range(nregion):
    (latlim,lonlim) = grid.get_region_lim(region[regi],lonmax=360)
    nlim = np.shape(latlim)[0]
    for i in range(nlim):
        mask = grid.regionMask(lat,lon,latlim[i],lonlim[i])
        region_mask[np.where(mask==1)] = regi+1
    (latlim,lonlim) = grid.get_region_lim(region[regi],lonmax=180)
    nlim = np.shape(latlim)[0]
    for i in range(nlim):
        mask = grid.regionMask(latt,lont,latlim[i],lonlim[i])
        region_mask_obs[np.where(mask==1)] = regi+1
region_mask[np.where(land_mask<0.3)] = -1
region_mask_obs[np.where(land_mask_obs<0.3)] = -1


m = Basemap(projection='mill',\
            fix_aspect=True,\
            llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
#m = Basemap(projection='npstere',boundinglat=90,lon_0=-90,round=True)

latt,lont,region_mask_tr = pp.grid_for_map(lat,lon,region_mask)
lons,lats = np.meshgrid(lont,latt)
x,y = m(lons,lats)
cmap = plt.cm.Accent
clev = np.arange(0,nregion+1,1)
plt.figure()
m.drawcoastlines()
m.drawcountries()
m.drawparallels(np.arange(-80,81,20))
m.contourf(x,y,region_mask_obs,clev,cmap=cmap)
cb = m.colorbar()
cb.set_ticklabels(region)
cb.update_ticks()
plt.tight_layout()
"""
clev1 = np.arange(-1.6,1.7,0.2)
cmap = plt.cm.RdYlBu
clabel = "precipitation change (mm day$^{-1}$)"
lat1s = [0,-10,-5]
lat2s = [0,10,5]
lon1s = [0,90,105]
lon2s = [0,150,135]
for i in [1,2]:
    plt.figure()
    m.drawcoastlines()
    x,y = m(lons,lats)
    cs = m.contour(x,y,data[0,:,:],clev,linewidths=1,colors='k')
    plt.clabel(cs,fontsize=12,fmt='%1.0f',inline=1)
    m.contourf(x,y,data[i,:,:]-data[0,:,:],clev1,cmap=cmap,extend='both')
    cb = m.colorbar(location='bottom',size='5%',pad='8%')
    cb.set_label(clabel)
    lonss = [lon1s[i],lon1s[i],lon2s[i],lon2s[i],lon1s[i]]
    latss = [lat1s[i],lat2s[i],lat2s[i],lat1s[i],lat1s[i]]
#    m.drawgreatcircle(lon1,lat1,lon1,lat2,linewidth=2,color='m')
    x,y = m(lonss,latss) # forgot this line 
    m.plot(x,y,'-',markersize=10,linewidth=2,color='m')
    plt.tight_layout()
"""
#%%