# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:49:37 2016

@author: Zhaoyi.Shen
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/z1s/PythonScripts')
import postprocess as pp

fdir = '/archive/Zhaoyi.Shen/fms/ulm/AM2/AM2_control_1990/gfdl.ncrc3-default-prod-openmp/pp/atmos_level/'
f = fdir+'ts/annual/16yr/atmos_level.1983-1998.netrad_toa.nc'
ncf = nc.netcdf_file(f,'r',mmap=True)
ann = ncf.variables['netrad_toa'][0,0,0].astype(np.float64)
f = fdir+'ts/monthly/16yr/atmos_level.198301-199812.netrad_toa.nc'
ncf = nc.netcdf_file(f,'r',mmap=True)
tmp = ncf.variables['netrad_toa'][:12,0,0].astype(np.float64)
mon = pp.month_to_year(tmp,np.ones(1),'annual')