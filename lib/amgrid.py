# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:43:35 2015

@author: z1s
"""
import numpy as np

Rad = 6371.0e3

def calcSigmaPres(ps,pk,bk):
    """
calculate pressure in model grid box
p = pk+bk*ps
---INPUT---
ps: surface pressure (3d)
pk,bk: parameter in sigma coordinate (1d, size: model vertical level + 1)
---OUTPUT---
phalf: pressure in model grid box (4d)
    """
    nphalf = np.size(pk) # vertical level + 1
    nlon = ps.shape[2]
    nlat = ps.shape[1]
    ntime = ps.shape[0]
    #preshalf = zeros(nlon,nlat,nphalf,ntime);
    #ind = np.where(bk==0)
    psc = ps.copy()
    pkc = pk.copy()
    bkc = bk.copy()
    psc.shape = (ntime,1,nlat,nlon)
    pkc.shape = (1,nphalf,1,1)
    bkc.shape = (1,nphalf,1,1)
    phalf = pkc+bkc*psc
    return phalf

def interp(pInterp,pfull,arr):
    """
interpolate
---INPUT---
pInterp: interpolate pressure
pfull: original pressure
arr(time,pfull,lat,lon): array in original pressure
---OUTPUT---
arr1(time,pInterp,lat,lon): array in interpolate pressure
    """
    nt = arr.shape[0]
    nlat = arr.shape[2]
    nlon = arr.shape[3]
    npInterp = np.size(pInterp)
    arr1 = np.zeros([nt,npInterp,nlat,nlon])    
    for ti in range(nt):
        for lati in range(nlat):
            for loni in range(nlon):
                xp = pfull[ti,:,lati,loni]
                fp = arr[ti,:,lati,loni]
                arr1[ti,:,lati,loni] = np.interp(pInterp,xp,fp,\
                                                 left=np.nan,right=np.nan)
    return arr1
    
def calcGridArea_square(lat,lon):
#calculate surface area in model grid
#---INPUT---
#lat: latitude
#lon: longitude
#---OUTPUT---
#area: surface area (lat,lon)
    nlat = np.size(lat)
    lat1 = np.zeros(nlat+2)
    lat1[0] = -90
    lat1[-1] = 90
    lat1[1:-1] = lat
    dlat = lat1[1:]-lat1[:-1]
    dlat[0] = dlat[0]+0.5*(lat1[1]-lat1[0])
    dlat[-1] = dlat[-1]+0.5*(lat1[-1]-lat1[-2])
    dlat1 = 0.5*(dlat[1:] + dlat[:-1])
    dlat1.shape = [nlat,1]
    nlon = np.size(lon)
    lon1 = np.zeros(nlon+2)
    lon1[0] = 0
    lon1[-1] = 360
    lon1[1:-1] = lon
    dlon = lon1[1:]-lon1[:-1]
    dlon[0] = dlon[0]+0.5*(lon1[1]-lon1[0])
    dlon[-1] = dlon[-1]+0.5*(lon1[-1]-lon1[-2])
    dlon1 = 0.5*(dlon[1:] + dlon[:-1])
    dlon1.shape = [1,nlon]
    xsc = 2*np.pi*Rad/360*np.cos(lat/90*np.pi/2)
    ysc = 2*np.pi*Rad/360
    xsc.shape = [nlat,1]
    area = dlat1*dlon1*xsc*ysc
    return area
   
def calcGridArea(lat,lon):
    """
calculate surface area in model grid
---INPUT---
lat: latitude
lon: longitude
---OUTPUT---
area(lat,lon): surface are
    """
    nlat = np.size(lat)
    nlon = np.size(lon)
    dp = np.pi/(nlat-1)
    dl = 2*np.pi/nlon
    sine = np.zeros(nlat)
    sine[0] = -1
    for i in range(1,nlat):
        sine[i] = np.sin(-0.5*np.pi+(i-0.5)*np.pi/(nlat-1))
    weight = np.zeros(nlat)
    for i in range(nlat-1):
        weight[i] = sine[i+1]-sine[i]
    weight[nlat-1] = 1-sine[nlat-1]
    cosp = weight/dp
    area = np.zeros([nlat,nlon])
    for i in range(nlat):
        area[i,:] = Rad**2*dl*dp*cosp[i]
    return area

def regionMask(lat,lon,latlim,lonlim):
    """
calculate regional mask
---INPUT---
lat: latitude
lon: longtitude
latlim:
lonlim:
---OUTPUT---
mask(lat,lon): 0 or 1
    """
    nlat = np.size(lat)
    nlon = np.size(lon)
    mask = np.zeros((nlat,nlon))
    for i in range(nlat):
        for j in range(nlon):
            if lat[i]>=latlim[0] and lat[i]<=latlim[1] \
            and lon[j]>=lonlim[0] and lon[j]<=lonlim[1]:
                mask[i,j] = 1
    return mask

def get_region_lim(region,lonmax=360):
    latlim_dict_360 = {
    'coveu':[[35,70],[35,70]],
    'covas':[[10,50]],
	'coveas':[[20,50]],
	'covsas':[[5,35]],
    'covnam':[[30,50]],
    'simEU':[[30,60],[30,60]],
    'simEA':[[5,35]],
    'EU':[[36,66],[38,81]],
    'NM':[[76,83],[60,76],[29,60],[25,29]],
    'SEUS':[[24,40]],
    'EUS':[[30,47]],
    'EA':[[22,28],[28,47],[47,50],[19,22]],
    'SA':[[25,29],[-11,38],[-11,28],[-11,22]],
    'ME':[[36,38],[25,38],[12,38],[29,38]],  
    'RU':[[38,81],[47,81],[50,81],[50,81]],
    'AF':[[-35,36],[-35,36],[-35,25],[-35,12]],
    'NAF':[[25,36],[25,36]],
    'AU':[[-47,-11]],
    'SM':[[-56,29],[-56,25]],
    'glb':[[-90,90],[-90,90]],
    'NH':[[0,90],[0,90]],
    'SH':[[-90,0],[-90,0]],
    'tropics':[[-30,30],[-30,30]],
    'midlat':[[-60,-30],[-60,-30],[30,60],[30,60]],
    'highlat':[[-90,-60],[-90,-60],[60,90],[60,90]]
	}
    lonlim_dict_360 = {
    'coveu':[[0,40],[350,360]],
    'covas':[[70,140]],
	'coveas':[[95,135]],
	'covsas':[[70,90]],
    'covnam':[[235,300]],
    'simEU':[[0,50],[350,360]],
    'simEA':[[70,120]],
    'EU':[[336,360],[0,49]],
    'NM':[[191,287],[191,298],[191,307],[263,307]],
    'EA':[[97,145],[80,145],[87,145],[106,145]],
    'SA':[[63,70],[70,80],[80,97],[97,150]],
    'ME':[[24,32],[32,39],[39,63],[63,70]],
    'RU':[[49,80],[80,87],[87,180],[180,191]],
    'AF':[[343,360],[0,32],[32,39],[39,51]],
    'NAF':[[343,360],[0,32]],
    'AU':[[113,179]],
    'SM':[[245,259],[259,326]],
    'glb':[[0,180],[180,360]],
    'NH':[[0,180],[180,360]],
    'SH':[[0,180],[180,360]],
    'tropics':[[0,180],[180,360]],
    'midlat':[[0,180],[180,360],[0,180],[180,360]],
    'highlat':[[0,180],[180,360],[0,180],[180,360]]
	}
    latlim_dict = latlim_dict_360.copy()
    lonlim_dict = lonlim_dict_360.copy()
    for key in lonlim_dict:
        a = []
        for i in lonlim_dict[key]:
            if (max(i)>180):
                a.append(list(np.subtract(i,360)))
            else:
                a.append(i)
        lonlim_dict[key] = a
    if (lonmax==360):
       latlim = latlim_dict_360[region]
       lonlim = lonlim_dict_360[region]
    else:
       latlim = latlim_dict[region]
       lonlim = lonlim_dict[region]
    return latlim,lonlim
