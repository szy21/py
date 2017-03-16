# -*- coding: utf-8 -*-
"""
Created on Mon May 2 09:37:34 2016

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import postprocess as pp
from scipy.io import netcdf as nc
import numpy as np
import calendar
import matplotlib.pyplot as plt

CP_AIR = 1004.64
LE = 2.500e6
GRAV = 9.80
RAIR = 287.04
RADIUS = 6371.0e3
"""
indir = '/archive/Zhaoyi.Shen/fms/siena/AM2.1/'
indir_sub = 'ts/monthly/146yr/'
outdir = '/archive/Zhaoyi.Shen/home/research/climate/npz/AM2.1n/'
exper = 'AM2.1RC3_'
pert_dict = {'1860_':'SST_','allforc_':'all_',\
             'WMGGOnly_':'WMGG_','aeroOnly_':'aero_',\
             '1860aero_':'1860aero_'}
pert = ['1860_','allforc_','aeroOnly_','1860aero_']
ens = ['A1','A2','A3','A4','A5']
enso = ens
"""
indir = '/archive/Zhaoyi.Shen/fms/ulm_201505_c3/SM2/'
indir_sub = 'ts/monthly/20yr/'
outdir = '/home/Zhaoyi.Shen/research/nonlinear/npz/SM2/'
exper = 'SM2_'
pert_dict = {'control_1990':'ctrl','2xCO2':'2xCO2',\
    'm2c50w30':'m2c50','m2c45w30':'m2c45','m2c35w30':'m2c35','m2c25w30':'m2c25',\
    'm2c15w30':'m2c15','m2c05w30':'m2c05','m2c00w30':'m2c00',\
    'm2c35w60':'m2c35w60','m6c35w30':'m6c35',\
    'm6c25w30':'m6c25','m1c25w30':'m1c25','m4c25w30':'m4c25',\
    'm0.5c15w30':'m0.5c15',\
    '2xCO2+m2c50w30':'2xCO2+m2c50','2xCO2+m2c45w30':'2xCO2+m2c45',\
    '2xCO2+m2c35w30':'2xCO2+m2c35','2xCO2+m2c25w30':'2xCO2+m2c25',\
    '2xCO2+m2c15w30':'2xCO2+m2c15','2xCO2+m2c05w30':'2xCO2+m2c05',\
    '2xCO2+m2c00w30':'2xCO2+m2c00',\
    '2xCO2+m6c35w30':'2xCO2+m6c35',\
    '2xCO2+m0.5c15w30':'2xCO2+m0.5c15',\
    '2xCO2+m8c15w30':'2xCO2+m8c15','2xCO2+m10c15w30':'2xCO2+m10c15'}
pert = [
    #'control_1990', '2xCO2',\
    #'m2c00w30','m2c05w30','m2c15w30','m2c25w30','m2c35w30','m2c45w30',\
    #'m2c50w30','m6c25w30','m6c35w30','m2c35w60',\
    #'2xCO2+m2c00w30','2xCO2+m2c05w30','2xCO2+m2c15w30','2xCO2+m2c25w30',\
    #'2xCO2+m2c35w30','2xCO2+m2c45w30','2xCO2+m2c50w30','2xCO2+m6c35w30',\
    #'m0.5c15w30','2xCO2+m0.5c15w30',\
    'm1c25w30','m4c25w30',\
    #'2xCO2+m10c15w30',\
    #'2xCO2+m8c15w30'
    ]
ens = ['']
enso = ens
"""
indir = '/archive/s1h/am2/'
indir_sub = 'ts/monthly/30yr/'
outdir = '/home/Zhaoyi.Shen/research/landprecip/npz/'
exper = 'am2clim_reyoi'
pert_dict = {'':'RAS','_uw_lofactor0.5':'UW'}
pert = ['','_uw_lofactor0.5']
ens = ['','+2K']
enso = ens
"""
flag = ['annual','DJF','JJA']
nflag = np.size(flag)
sub_dict = {'annual':'annual','MAM':'MAM','JJA':'JJA','SON':'SON','DJF':'DJF'}

npert = np.size(pert)
nens = np.size(ens)
plat = 'gfdl.ncrc3-intel-prod-openmp/'
diag = 'atmos_level'
diago = 'var2d'
var = ['t_ref']
"""
var = ['t_ref','LWP','precip','low_cld_amt','mid_cld_amt','high_cld_amt','tot_cld_amt']

var = ['swdn_sfc','swup_sfc','swdn_toa','swup_toa',\
       'swdn_sfc_clr','swup_sfc_clr','swdn_toa_clr','swup_toa_clr',\
       'lwdn_sfc','lwdn_sfc_clr','lwup_sfc','lwup_sfc_clr','olr','olr_clr',\
       'evap','shflx','netrad_toa','netrad_toa_clr']
"""
nvar = np.size(var)
varo = ['t_ref_zm']
sim = []
simo = []
for i in range(npert):
    for j in range(nens):
        sim.append(pert[i]+ens[j])
        simo.append(pert_dict[pert[i]]+enso[j])
nsim = np.size(sim)

zm = True
init = True
init3d = False
init3dp = False
timeo = '0761-0860'
yr1 = np.arange(761,842,20)
yr2 = np.arange(780,861,20)
yr_ts = np.ones(100)
nyr1 = np.size(yr1)
nyr = yr2[-1]-yr1[0]+1
yr = np.arange(yr1[0],yr2[-1],1)
yrstr = []
for yri in range(nyr1):
    yr1C = ('000'+str(yr1[yri]))[-4:]
    yr2C = ('000'+str(yr2[yri]))[-4:]
    yrC = yr1C+'01-'+yr2C+'12.'
    yrstr.append(yrC)
ind = range(nsim)
nfile = np.size(yrstr)

"""
check before this line
"""

#%%
#npert = 1
land_mask = 0
for vi in range(nvar):
    for i in ind:
        atmdir = indir+exper+sim[i]+'/'+plat+'pp/'+diag+'/'
        stafile = atmdir+diag+'.static.nc'
        fs = []
        fs.append(nc.netcdf_file(stafile,'r',mmap=True))
        bk = fs[-1].variables['bk'][:].astype(np.float64)
        pk = fs[-1].variables['pk'][:].astype(np.float64)
        lat = fs[-1].variables['lat'][:].astype(np.float64)
        lon = fs[-1].variables['lon'][:].astype(np.float64)
        phalf = fs[-1].variables['phalf'][:].astype(np.float64)
        zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
        if ('land_mask' in fs[-1].variables):
            land_mask = fs[-1].variables['land_mask'][:].astype(np.float64)
        fs[-1].close()
        nlat = np.size(lat)
        nlon = np.size(lon)
        nphalf = np.size(phalf)
        #%%
        filedir = atmdir+indir_sub
        files = []
        for fi in range(nfile):
            filename = filedir+diag+'.'+yrstr[fi]+var[vi]+'.nc'
            files.append(filename)
        ts = pp.ts_multi_files(files,var[vi],0)
        if zm:
            ts = np.nanmean(ts,-1)
        #%%
        if init and vi==0:
            outfile = outdir+'dim.'+simo[i]+'.npz'
            fio.save(outfile,lat=lat,lon=lon,phalf=phalf,land_mask=land_mask,year=yr)
        if init3d:
            tmp = nc.netcdf_file(files[-1],'r',mmap=True)
            pfull = tmp.variables['pfull'][:].astype(np.float64)
            outfile = outdir+'dim.'+simo[i]+'.npz'
            fio.save(outfile,pfull=pfull)
        if init3dp:
            tmp = nc.netcdf_file(files[-1],'r',mmap=True)
            level = tmp.variables['level'][:].astype(np.float64)
            outfile = outdir+'dim.'+simo[i]+'.npz'
            fio.save(outfile,level=level)
        for flagi in range(nflag):
            if (flag[flagi] != ''):
                ts_flag = pp.month_to_year(ts,yr_ts,flag[flagi])
                outdir_sub='ts/'+sub_dict[flag[flagi]]+'/'
            outfile = outdir+outdir_sub+diago+'.'+timeo+'.'+simo[i]+'.npz'
            fio.save(outfile,**{varo[vi]:ts_flag})
        print sim[i]
    print var[vi]
    
