# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:01:42 2015

@author: Zhaoyi.Shen
"""

import sys
sys.path.append('/home/z1s/PythonScripts')
sys.path.append('/home/z1s/py/lib')
import numpy as np
import binfile_io as fio

Rad = 6371.0e3
#basedir = '/archive/Zhaoyi.Shen/home/research/climate/npz/new/AM3/ts/JJA/'
basedir = '/home/z1s/research/landprecip/npz/ts/JJA/'
pert = ['UW','UW+2K']
npert = np.size(pert)
ens = ['']
npert = np.size(pert)
nens = np.size(ens)
sim = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
#npert = 4
diag = 'var2d'
time = '1983-2012.'
tmp = np.zeros([12,24,90])
for si in range(nsim):
    filename = basedir+diag+'.'+time+sim[si]+'.npz'
    #outfile = basedir+'dim.'+pert[si]+'_sigma.npz'
    npz = np.load(filename)
    #tmp = npz['z500_2-6d_std']
    #fio.save(filename,**{var[vi]:tmp})
    #fio.save(filename,z500_std=tmp)
    fio.delete(filename,'z500_std_o4')