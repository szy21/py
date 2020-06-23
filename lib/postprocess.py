# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:46:11 2016

@author: Zhaoyi.Shen
"""
from scipy.io import netcdf as nc
from netCDF4 import Dataset
import numpy as np
import calendar

def av_multi_files(files, var, dim):
    nfile = np.size(files)
    fs = []
    fs.append(nc.netcdf_file(files[0],'r',mmap=True))
    tmp = fs[-1].variables[var][:].astype(np.float64)
    fs[-1].close()
    tmp = np.mean(tmp,dim)
    av = np.zeros(tmp.shape+(nfile,))
    for fi in range(nfile):
        fs.append(nc.netcdf_file(files[fi],'r',mmap=True))
        tmp = fs[-1].variables[var][:].astype(np.float64)
        fs[-1].close()
        tmp = np.mean(tmp,dim)
        av[...,fi] = tmp
    av = np.mean(av,-1)
    return av
    
def ts_multi_files(files, var_name, dim, f=None):
    if f is None:
        f = lambda x: x
    nfile = np.size(files)
    fs = []
    #fs.append(nc.netcdf_file(files[0],'r',mmap=True))
    fs.append(Dataset(files[0],'r',mmap=True))
    var = fs[-1].variables[var_name]
    tmp = f(var[:].astype(np.float64))
    """
    if hasattr(var,'_FillValue'):
        tmp[np.where(tmp==var._FillValue)] = np.nan
    if hasattr(var,'scale_factor'):
        tmp = tmp*var.scale_factor
    """
    fs[-1].close()
    ts = tmp
    for fi in range(1,nfile):
        #fs.append(nc.netcdf_file(files[fi],'r',mmap=True))
        fs.append(Dataset(files[fi],'r',mmap=True))
        var = fs[-1].variables[var_name]
        tmp = f(var[:].astype(np.float64))
        """
        if hasattr(var,'_FillValue'):
            tmp[np.where(tmp==var._FillValue)] = np.nan
        if hasattr(var,'scale_factor'):
            tmp = tmp*var.scale_factor
        """
        fs[-1].close()
        ts = np.concatenate((ts,tmp),axis=dim)
    if  isinstance(ts,np.ma.core.masked_array):
        fillvalue = tmp.fill_value
        ts = np.array(ts)
        ts[np.where(ts==fillvalue)] = np.nan
    return ts

def month_to_year(arr1,year,flag,weighted=True):
    t_dict = {'annual':[0,12],\
        'MAM':[2,5],'JJA':[5,8],'SON':[8,11],'DJF':[-1,2],'JAS':[6,9],\
        'MJJASO':[4,10],'JJASON':[5,11],'JJAS':[5,9]}
    weight_dict = {'annual':np.array((31,28,31,30,31,30,31,31,30,31,30,31))/365.,\
        'MAM':np.array((31,30,31))/92.,\
        'JJA':np.array((30,31,31))/92.,\
        'SON':np.array((30,31,30))/91.,\
        'DJF':np.array((31,31,28))/90.,\
        'JAS':np.array((31,31,30))/92.,\
        'MJJASO':np.array((31,30,31,31,30,31))/184.,\
        'JJASON':np.array((30,31,31,30,31,30))/183.,\
        'JJAS':np.array((30,31,31,30))/122.}
    weight_leap_dict = weight_dict.copy()
    weight_leap_dict['annual'] = np.array((31,29,31,30,31,30,31,31,30,31,30,31))/366.
    weight_leap_dict['DJF'] = np.array((31,31,29))/91.
    t = t_dict[flag]
    nyr = np.shape(arr1)[0]/12
    arr2 = np.zeros((nyr,)+arr1.shape[1:])
    for yri in range(nyr):
        is_leap = calendar.isleap(year[yri])
        if is_leap:
            weight = weight_leap_dict[flag].copy()
        else:
            weight = weight_dict[flag].copy()
        t1 = yri*12+t[0]        
        t2 = yri*12+t[1]
        if (t1<0):
            t1 = 0
            if is_leap:
                weight = np.array((31,29))/60.
            else:
                weight = np.array((31,28))/59.
        weight.shape = weight.shape+(1,)*(arr1.ndim-1)
        if not weighted:
            arr2[yri,...] = np.mean(arr1[t1:t2,...],0)
        else:
            arr2[yri,...] = np.sum(arr1[t1:t2,...]*weight,0)
    return arr2
    
def day_to_year(arr1,year,flag):
    nyr = np.size(year)
    arr2 = np.zeros((nyr,)+arr1.shape[1:])
    nleap = 0
    for yri in range(nyr):
        is_leap = calendar.isleap(year[yri])
        if is_leap:
            days = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
        else:
            days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        t_dict = {'annual':[0,np.sum(days)],
                  'MAM':[np.sum(days[:2]),np.sum(days[:5])],
                  'JJA':[np.sum(days[:5]),np.sum(days[:8])],
                  'SON':[np.sum(days[:8]),np.sum(days[:11])],
                  'DJF':[-31,np.sum(days[:2])]}
        t = t_dict[flag]
        t1 = yri*365+nleap+t[0]
        if t1<0:
            t1 = 0
        t2 = yri*365+nleap+t[1]
        arr2[yri,...] = np.mean(arr1[t1:t2,...],0)
        if is_leap:
            nleap = nleap+1
    return arr2
    
def grid_for_map(lat,lon,arr1):
    nlon = np.size(lon)
    nlat = np.size(lat)
    lonmid = int(nlon/2)
    lont = np.zeros(nlon)
    lont[:lonmid] = lon[lonmid:]-360
    lont[lonmid:nlon] = lon[:lonmid]
    latt = np.zeros(nlat)
    latt[:nlat] = lat
    arr2 = np.zeros(arr1.shape[:-2]+(arr1.shape[-2],arr1.shape[-1]))
    arr2[...,:nlat,:lonmid] = arr1[...,:,lonmid:]
    arr2[...,:nlat,lonmid:nlon] = arr1[...,:,:lonmid]
    """
    arr2[...,0,1:lonmid+1] = arr1[...,0,lonmid:]
    arr2[...,0,lonmid+1:nlon+1] = arr1[...,0,:lonmid]
    arr2[...,nlat+1,1:lonmid+1] = arr1[...,nlat-1,lonmid:]
    arr2[...,nlat+1,lonmid+1:nlon+1] = arr1[...,nlat-1,:lonmid]
    arr2[...,:,0] = arr2[...,:,1]
    arr2[...,:,nlon+1] = arr2[...,:,1]
    """
    return latt,lont,arr2
    
        
