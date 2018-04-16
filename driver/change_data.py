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

RADIUS = 6371.0e3
basedir = '/archive/Zhaoyi.Shen/home/research/climate/npz/AM4/ts/JJASON/'
#basedir = '/home/z1s/research/nonlinear/npz/SM2/sym/'
pert = ['SSTvol']
npert = np.size(pert)
ens = ['_A1']
#ens = ['mean']
npert = np.size(pert)
nens = np.size(ens)
sim = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
#npert = 4
diag = 'var2d.'
time = '1870-2014.'
#tmp = np.zeros([12,24,90])
var = ['aer_ex_c_vs','aer_ab_c_vs']

#varo = ['z500','pv500']
nvar = np.size(var)
for si in range(nsim):
    filename = basedir+diag+time+sim[si]+'.npz'
    outfile = basedir+'dim.'+sim[si]+'.npz'
    npz = np.load(filename)
    #level = npz['level']
        #fio.save(outfile,**{var[vi]:tmp})
    #fio.save(outfile,level=level)
    for vi in range(nvar):
        fio.delete(filename,var[vi])
        #fio.delete(filename,'netrad_toa_cld','netrad_toa_clr')
    print sim[si]