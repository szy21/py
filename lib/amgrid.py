# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:43:35 2015

@author: z1s
"""
import numpy as np

Rad = 6371.0e3


def calcSigmaPres(ps,pk,bk):
#calculate pressure in model grid box
#p = pk+bk*ps
#---INPUT---
#ps: surface pressure (3d)
#pk,bk: parameter in sigma coordinate (1d, size: model vertical level + 1)
#---OUTPUT---
#phalf: pressure in model grid box (4d)
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
#interpolate
#---INPUT---
#pInterp: interpolate pressure
#pfull: original pressure
#arr: 3D array (time,lat,lon) in original pressure
#---OUTPUT---
#arr1: 3D array (time,lat,lon) in interpolate pressure
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
"""    
def calcGridArea(lat,lon):
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
"""    
def calcGridArea(lat,lon):
#calculate surface area in model grid
#---INPUT---
#lat: latitude
#lon: longitude
#---OUTPUT---
#area: surface area (lat,lon)
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