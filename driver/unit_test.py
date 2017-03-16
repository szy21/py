# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:44:40 2017

@author: Zhaoyi.Shen
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import postprocess as pp
import numpy as np
from scipy.io import netcdf as nc
import calendar

arr1 = np.arange(1,25,1)
yr = np.array((1999,2000))
ans_key = {'annual':np.array((2382/365.,6776/366.)),\
    'MAM':np.array((368/92.,1472/92.)),\
    'JJA':np.array((645/92.,1749/92.)),\
    'SON':np.array((910/91.,2002/91.)),\
    'DJF':np.array((87/59.,1181/91.))}
for flag in ['MAM','JJA','SON','DJF']:
    arr2 = pp.month_to_year(arr1,yr,flag)
    ans = ans_key[flag]
    if np.max(np.abs(arr2-ans)) < 1e-8:
        print flag+':Correct!'
    else:
        print flag+':',arr2,ans

fdir = '/archive/Zhaoyi.Shen/fms/ulm/AM2/AM2_control_1990/gfdl.ncrc3-default-prod-openmp/pp/atmos_level/'
fname = fdir+'ts/annual/16yr/atmos_level.1983-1998.netrad_toa.nc'
f = nc.netcdf_file(fname,'r')
ann = f.variables['netrad_toa'][:]
fname = fdir+'ts/monthly/16yr/atmos_level.198301-199812.netrad_toa.nc'
f = nc.netcdf_file(fname,'r')
tmp = f.variables['netrad_toa'][:]
mon = pp.month_to_year(tmp,np.ones(16),'annual')
diff = np.abs(mon-ann)
if np.max(diff) < 1e-5:
    print 'Correct!'
else:
    print diff.max(), np.unravel_index(diff.argmax(),diff.shape)

fdir = '/archive/Zhaoyi.Shen/fms/ulm/AM3/c48L48_am3p11_allforc_A1/gfdl.ncrc3-intel-prod-openmp/pp/atmos_level/'
fname = fdir+'ts/annual/5yr/atmos_level.2010-2014.t_ref.nc'
f = nc.netcdf_file(fname,'r')
ann = f.variables['t_ref'][:].astype(np.float64)
fname = fdir+'ts/monthly/5yr/atmos_level.201001-201412.t_ref.nc'
f = nc.netcdf_file(fname,'r')
tmp = f.variables['t_ref'][:].astype(np.float64)
mon = pp.month_to_year(tmp,np.arange(2010,2015,1),'annual')
diff = np.abs(mon-ann)
if np.max(diff) < 1e-5:
    print 'Correct!'
else:
    print diff.max(), np.unravel_index(diff.argmax(),diff.shape)