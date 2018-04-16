# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:59:39 2016

@author: Zhaoyi.Shen
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import postprocess as pp
from scipy.io import netcdf as nc
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt

model_in = 'AM3n'
model_out = 'AM4_CMIP5'
indir = '/archive/Zhaoyi.Shen/home/research/climate/npz/'+model_in+'/'
indir_sub = 'ts/annual/'
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz/'+model_out+'/'
outdir_sub = indir_sub
pert = ['SST','aero']
npert = np.size(pert)
ens = ['_A1']
nens = np.size(ens)
sim = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
var = 'SO2_emis_cmip2'
diag = 'emis2d'
time1 = '1870-2015'
time2 = '1870-2014'
for si in range(nsim):
    grid_inf = indir+'dim.'+sim[0]+'.npz'
    npz = np.load(grid_inf)
    lat_in = npz['lat']
    lon_in = npz['lon']
    filename = indir+indir_sub+diag+'.'+time1+'.'+sim[si]+'.npz'
    npz = np.load(filename)
    tmp_in = npz[var]
    ntime = np.shape(tmp_in)[0]
    grid_outf = outdir+'dim.'+sim[0]+'.npz'
    npz = np.load(grid_outf)
    lat_out = npz['lat']
    nlat = np.size(lat_out)
    lon_out = npz['lon']
    nlon = np.size(lon_out)
    (lonm_out,latm_out) = np.meshgrid(lon_out,lat_out)
    tmp_out = np.zeros((ntime-1,nlat,nlon))
    for ti in range(ntime-1):
        (lonm_in,latm_in) = np.meshgrid(lon_in,lat_in)
        """
        tmp_rbf = interp.Rbf(latm_in,lonm_in,tmpt[ti,:,:],function='linear',smooth=0)
        tmp_out[ti,:,:] = tmp_rbf(latm_out,lonm_out)
        """
        tmp_out[ti,:,:] = \
        interp.griddata(np.array([latm_in.ravel(),lonm_in.ravel()]).T,\
        tmp_in[ti,:,:].ravel(),(latm_out,lonm_out),method='nearest')
    outfile = outdir+outdir_sub+diag+'.'+time2+'.'+sim[si]+'.npz'
    fio.save(outfile,**{var:tmp_out})
    print sim[si]