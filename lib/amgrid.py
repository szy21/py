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
    latlim_dict_360 = {'EuNAf':[[25,65],[25,65]],
    'NAm':[[15,55]],
    'EAsSAs':[[15,50],[5,35]],
    'EAs':[[15,50]],
    'SAs':[[5,35]],
    'tsEU':[[35,60],[35,60]],
    'tsEA':[[15,50]],     
    'tsNA':[[15,55]],
    'EU':[[36,66],[38,81]],
    'simEU':[[30,60],[30,60]],
    'simEU1':[[35,65],[35,65]],
    'largeEU':[[30,70],[30,70]],
    'trendEU':[[25,70],[25,70]],                
    'NM':[[76,83],[60,76],[29,60],[25,29]],
    'simNM':[[45,65]],
    'SEUS':[[24,40]],
    'EUS':[[30,47]],
    'EA':[[22,28],[28,47],[47,50],[19,22]],
    'SA':[[25,29],[-11,38],[-11,28],[-11,22]],
    'simEA':[[5,35]],
    'largeEA':[[10,30]],
    'trendEA':[[5,35]],
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
    'highlat':[[-90,-60],[-90,-60],[60,90],[60,90]]}
    lonlim_dict_360 = {'EuNAf':[[0,50],[350,360]],
    'NAm':[[235,300]],
    'EAsSAs':[[95,160],[50,95]],
    'EAs':[[95,160]],
    'SAs':[[50,95]],
    'tsEU':[[0,50],[350,360]],
    'tsEA':[[75,150]],
    'tsNA':[[235,300]],
    'EU':[[336,360],[0,49]],
    'simEU':[[0,50],[350,360]],
    'simEU1':[[0,50],[350,360]],
    'largeEU':[[0,65],[350,360]],
    'trendEU':[[0,50],[340,460]],
    'NM':[[191,287],[191,298],[191,307],[263,307]],
    'simNM':[[250,300]],
    'SEUS':[[254,280]],
    'EUS':[[260,280]],
    'EA':[[97,145],[80,145],[87,145],[106,145]],
    'SA':[[63,70],[70,80],[80,97],[97,150]],
    'simEA':[[70,120]],
    'largeEA':[[65,125]],
    'trendEA':[[65,125]],
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
    'highlat':[[0,180],[180,360],[0,180],[180,360]]}
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