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

model = 'CM4'
obs = 'BEST_LO'
indir = '/archive/Zhaoyi.Shen/home/research/climate/npz/'+model+'/'
indir_sub = 'ts/DJF/'
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz_interp/'+obs+'/'+model+'/'
outdir_sub = indir_sub
grid_outf = '/archive/Zhaoyi.Shen/home/research/climate/npz_interp/'+\
    obs+'/'+'dim.'+obs+'.npz'
npz = np.load(grid_outf)
lat_out = npz['lat']
nlat = np.size(lat_out)
lon_out = npz['lon']
nlon = np.size(lon_out)
(lonm_out,latm_out) = np.meshgrid(lon_out,lat_out)
pert = ['all']
npert = np.size(pert)
ens = ['_A1']
nens = np.size(ens)
sim = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
var = 't_ref'
diag = 'var2d'
time = '1850-2014'
for si in range(nsim):
    grid_inf = indir+'dim.'+sim[0]+'.npz'
    npz = np.load(grid_inf)
    lat_in = npz['lat']
    lon_in = npz['lon']
    land_mask_in = npz['land_mask']
    (latt,lont,land_maskt) = pp.grid_for_map(lat_in,lon_in,land_mask_in)
    (lonm_in,latm_in) = np.meshgrid(lont,latt)
    land_mask_out = interp.griddata(np.array([latm_in.ravel(),lonm_in.ravel()]).T,\
        land_maskt.ravel(),(latm_out,lonm_out),method='linear')
    outfile = outdir+'dim.'+sim[si]+'.npz'
    fio.save(outfile,land_mask=land_mask_out)
    filename = indir+indir_sub+diag+'.'+time+'.'+sim[si]+'.npz'
    npz = np.load(filename)
    tmp_in = npz[var]
    latt,lont,tmpt = pp.grid_for_map(lat_in,lon_in,tmp_in)
    ntime = np.shape(tmpt)[0]
    tmp_out = np.zeros((ntime,nlat,nlon))
    for ti in range(ntime):
        (lonm_in,latm_in) = np.meshgrid(lont,latt)
        """
        tmp_rbf = interp.Rbf(latm_in,lonm_in,tmpt[ti,:,:],function='linear',smooth=0)
        tmp_out[ti,:,:] = tmp_rbf(latm_out,lonm_out)
        """
        tmp_out[ti,:,:] = \
            interp.griddata(np.array([latm_in.ravel(),lonm_in.ravel()]).T,\
            tmpt[ti,:,:].ravel(),(latm_out,lonm_out),method='linear')
    outfile = outdir+outdir_sub+diag+'.'+time+'.'+sim[si]+'.npz'
    fio.save(outfile,**{var:tmp_out})
    print sim[si]