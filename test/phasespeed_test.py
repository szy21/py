# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:25:52 2016

@author: Zhaoyi.Shen
"""

import sys
sys.path.append('/home/z1s/py/lib/')
import spacetime_analysis as stan
import numpy as np
from scipy.io import netcdf as nc
from matplotlib import pyplot as plt

RADIUS = 6371.0e3

basedir = '/archive/Zhaoyi.Shen/ulm_201510/'
outdir = '/home/z1s/research/nonlinearity/analysis/npz/'
exper = 'imr_t42_'
pert = ['control']
plat = '/gfdl.ncrc3-default-prod/'
diag = 'atmos_level'
var = 'ucomp'
npert = np.size(pert)
#npert = 1
data = np.zeros([npert,64])
fs = []
for i in range(npert):
    atmdir = basedir+exper+pert[i]+plat+'pp/'+diag+'/'
    stafile = atmdir+diag+'.static.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    phalf = fs[-1].variables['phalf'][:].astype(np.float64)
    zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    nphalf = np.size(phalf)
    #%%
    filedir = atmdir+'ts/daily/1yr/'
    yr = np.arange(4,5,1)
    nyr = np.size(yr)
    data = np.zeros([nyr,nphalf-1,nlat])
    #tmpZon = np.zeros([nmon,nlev-1,nlat])
    #phalfZon = np.zeros([nmon,nlev,nlat])    
    nmon=12
    for yri in range(nyr):
        yrC = '000'+str(yr[yri])+'0101-'+'000'+str(yr[yri])+'1231.'

        dayfile = filedir+diag+'.'+yrC+'ucomp.nc'
        fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
        ucomp = fs[-1].variables['ucomp'][:90,15,:,:].astype(np.float64)
        fs[-1].close()
        dayfile = filedir+diag+'.'+yrC+'vcomp.nc'
        fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
        vcomp = fs[-1].variables['vcomp'][:90,15,:,:].astype(np.float64)
        fs[-1].close()
        uTr = ucomp-np.mean(ucomp,2)[:,:,np.newaxis]
        vTr = vcomp-np.mean(vcomp,2)[:,:,np.newaxis]
        uTr = uTr[::2,:,:]
        vTr = vTr[::2,:,:]

    #%%
    ncp = 50
    int_time = 86400*2
    nwave = 2
    nfft_time = 30
    pcp_all = np.zeros([nlat,ncp])
    cp_all = np.zeros([nlat,ncp])
    p_all = np.zeros([nlat,46,nlon/2+1])
    for lati in range(nlat):
        
        p_all[lati,:,:],omega_all = \
        stan.calc_cospectrum(uTr[:,lati,:],vTr[:,lati,:])
        
        pcp_all[lati,:],cp_all[lati,:] = \
        stan.calc_phasespeed_cospectrum(uTr[:,lati,:],vTr[:,lati,:],
                                        ncp=ncp,nwave=nwave,
                                        int_time=int_time,nfft_time=nfft_time)
    
    #%%
    sf = stan.normalize(np.shape(uTr)[0],np.shape(uTr)[-1],
                        ncp,nwave,int_time,nfft_time)
    """
    p_allm = np.mean(p_all[12:15,:,:],0)/1e3
    plt.figure(figsize=[4.5,4.5])
    cs = plt.contour(omega_all,np.arange(1,nlat,1)[:10],np.transpose(p_allm[:,:10]),10,colors='k')
    plt.clabel(cs,fontsize=14,fmt='%1.2f',inline=1)
    """
    #%%
    plt.figure(figsize=[4.5,4.5])
    cs = plt.contour(cp_all[0,:],lat,pcp_all*sf,10,colors='k')
    plt.clabel(cs,fontsize=14,fmt='%1.2f',inline=1)
    um = np.mean(ucomp,2)
    um = np.mean(um,0)
    um = um/np.cos(lat/180.*np.pi)
    plt.plot(um,lat)
    plt.ylim([-70,70])