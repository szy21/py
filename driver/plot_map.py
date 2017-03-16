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

latlim_dict = {'EU':[[36,45],[36,45],[45,66],[45,66],[66,81]],
    'NM':[[76,83],[60,76],[29,60],[25,29]],
    'EA':[[22,28],[28,47],[47,50],[19,22]],
    'SA':[[25,29],[-11,38],[-11,28],[-11,22]],
    'ME':[[36,38],[25,38],[12,38],[29,38]],  
    'RU':[[38,45],[38,81],[47,81],[50,81]],
    'AF':[[-35,36],[-35,36],[-35,25],[-35,12]],
    'AU':[[-47,-11]],
    'SM':[[-56,29],[-56,25]],
    'global':[[-90,90]]}
lonlim_dict = {'EU':[[336,360],[0,24],[336,360],[0,49],[0,49]],
    'NM':[[191,287],[191,298],[191,307],[263,307]],
    'EA':[[97,145],[80,145],[87,145],[106,145]],
    'SA':[[63,70],[70,80],[80,97],[97,150]],
    'ME':[[24,32],[32,39],[39,63],[63,70]],
    'RU':[[24,49],[49,80],[80,87],[87,191]],
    'AF':[[343,360],[0,32],[32,39],[39,51]],
    'AU':[[113,179]],
    'SM':[[245,259],[259,326]],
    'global':[[0,360]]}

region = ['EU','NM','EA','SA','ME','RU','AF','AU','SM']
#region = ['AU']
nregion = np.size(region)

filename = '/home/z1s/research/climate/atmos_level.static.nc'
fs = nc.netcdf_file(filename,'r',mmap=True)
lat = fs.variables['lat'][:]
nlat = np.size(lat)
lon = fs.variables['lon'][:]
nlon = np.size(lon)
land_mask = fs.variables['land_mask'][:]
fs.close()
region_mask = np.zeros((nlat,nlon))
region_mask_obs = np.zeros((nlat,nlon))

latt,lont,land_mask_obs = pp.grid_for_map(lat,lon,land_mask)


for regi in range(nregion):
    (latlim,lonlim) = grid.get_region(region[regi],lonmax=360)
    nlim = np.shape(latlim)[0]
    for i in range(nlim):
        mask = grid.regionMask(lat,lon,latlim[i],lonlim[i])
        region_mask[np.where(mask==1)] = regi+1
    (latlim,lonlim) = grid.get_region(region[regi],lonmax=180)
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
m.contourf(x,y,region_mask_tr,clev,cmap=cmap)
m.colorbar()
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