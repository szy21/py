# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:06:25 2016

@author: Zhaoyi.Shen
"""

#need to fix!!

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
from scipy.io import netcdf as nc
from netCDF4 import Dataset
import numpy as np

infile='/archive/Zhaoyi.Shen/fms/input/SST/hadisst_sst.data.nc'
outfile='/archive/Zhaoyi.Shen/fms/input/SST/hadisst_sst_am2.data.nc'
f_in = Dataset(infile,'r')
dim_in = f_in.dimensions
var_in = f_in.variables
nlat = dim_in['lat'].size
nlon = dim_in['lon'].size
ntime = dim_in['time'].size
f_in.close()

root_grp = Dataset(outfile,'w')
root_grp.createDimension('time',None)
root_grp.createDimension('lat',nlat)
root_grp.createDimension('lon',nlon)
root_grp.createDimension('idim',ntime)
root_grp.createDimension('nv',2)
root_grp.close()
