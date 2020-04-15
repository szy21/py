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
LE = 2.500e6

#basedir = '/home/z1s/research/nonlinear/npz/SM2/sym/ts/JJA/'
basedir = '/archive/Zhaoyi.Shen/home/research/climate/npz/AM4n/ts/JJAS/'
pert = ['SST_','1860aero_']
"""
pert = ['ctrl_warsaw','2xCO2_warsaw',\
'm2c20w100','m2.5c20w100','m3c20w100','m3.5c20w100',\
'm4c20w100','m4.5c20w100','m5c20w100','m5.5c20w100',\
'2xCO2+m2c20w100','2xCO2+m2.5c20w100',\
'2xCO2+m3c20w100','2xCO2+m3.5c20w100',\
'2xCO2+m4c20w100','2xCO2+m4.5c20w100',\
'2xCO2+m5c20w100','2xCO2+m5.5c20w100']
"""
npert = np.size(pert)
#ens = ['']
ens = ['A1','A2','A3','A4','A5']
npert = np.size(pert)
nens = np.size(ens)
sim = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
nsim = np.size(sim)
#npert = 4
diag = 'rad2d.'
time = '1870-2015.'
#tmp = np.zeros([12,24,90])
var = ['swup_toa','swdn_toa','olr']
#varo = ['z500','pv500']
nvar = np.size(var)
for si in range(nsim):       
    filename = basedir+diag+time+sim[si]+'.npz'    
    npz = np.load(filename)
    netrad_toa= npz['swdn_toa']-npz['swup_toa']-npz['olr']
    fio.save(filename,netrad_toa=netrad_toa)
    
    for vi in range(nvar):
        filename = basedir+diag+time+sim[si]+'.npz'
        npz = np.load(filename)
        #tmp = npz[var[vi]]
        #fio.save(outfile,**{var[vi]:tmp})
        if var[vi] in npz.keys():
            fio.delete(filename,var[vi])

    #fio.delete(filename,'pv500')
    print sim[si]