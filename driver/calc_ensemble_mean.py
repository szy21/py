# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:06:24 2016

@author: Zhaoyi.Shen
"""
import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import postprocess as pp
from scipy.io import netcdf as nc
import numpy as np

indir = '/archive/Zhaoyi.Shen/home/research/climate/npz/AM4n/'
indir_sub = 'ts/SON/'
outdir = indir
outdir_sub = indir_sub
pert = ['aero']
npert = np.size(pert)
ens = ['_A1','_A2','_A3','_A4','_A5']
nens = np.size(ens)
diag = 'var2d'

var = ['t_ref']
#var = ['salt_col','dust_col']
"""
var = ['swdn_sfc','swup_sfc','swdn_toa','swup_toa',\
       'swdn_sfc_clr','swup_sfc_clr','swdn_toa_clr','swup_toa_clr',\
       'lwdn_sfc','lwdn_sfc_clr','lwup_sfc','lwup_sfc_clr','olr','olr_clr',\
       'evap','shflx','netrad_toa','netrad_toa_clr']
"""
nvar = np.size(var)
time = '1870-2015'
for vi in range(nvar):
    for i in range(npert):
        filename = indir+indir_sub+diag+'.'+time+'.'+pert[i]+ens[0]+'.npz'
        npz = np.load(filename)
        tmp = npz[var[vi]]
        av = np.zeros(tmp.shape+(nens,))
        for j in range(nens):
            filename = indir+indir_sub+diag+'.'+time+'.'+pert[i]+ens[j]+'.npz'
            npz = np.load(filename)
            av[...,j] = npz[var[vi]]
        av = np.mean(av,-1)
        
        outfile = outdir+outdir_sub+diag+'.'+time+'.'+pert[i]+'_mean'+'.npz'
        fio.save(outfile,**{var[vi]:av})
    print var[vi]