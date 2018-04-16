
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:37:34 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib')
import binfile_io as fio
import amgrid as grid
import postprocess as pp
from scipy.io import netcdf as nc
from netCDF4 import Dataset
import numpy as np
import calendar
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

CP_AIR = 1004.64
CV_AIR = 717.6
LE = 2.500e6
LF = 3.34e5
GRAV = 9.80
RDGAS = 287.04
RADIUS = 6371.0e3

indir = '/archive/Ming.Zhao/awg/verona/'
outdir = '/archive/Zhaoyi.Shen/home/research/landprecip/npz/'
exper = 'c96L32_am4g10r8_2010climo_mse3_rerun'
pert = ['']
perto = ['AM4']
npert = np.size(pert)
plat = '/gfdl.ncrc3-intel-prod-openmp/'
flag = ['annual']
nflag = np.size(flag)
diag = 'atmos_level'
diago = 'eflx2d'
season = 'annual'
sub_dict = {'annual':'annual','MAM':'MAM','JJA':'JJA','SON':'SON','DJF':'DJF'}
var = 'precip'
outdir_sub='ts/daily/'
varo = 'precip'
timeo = '1983-2012'
yr1 = np.arange(17,18,2)
yr2 = np.arange(21,22,2)
#yr_ts = np.arange(1870,2015,1)

nyr1 = np.size(yr1)
nyr = yr2[-1]-yr1[0]+1
yr = np.arange(yr1[0],yr2[-1]+1,1)
yr_ts = np.ones(nyr)
yrstr = []
for yri in range(nyr1):
    yr1C = ('000'+str(yr1[yri]))[-4:]
    yr2C = ('000'+str(yr2[yri]))[-4:]
    yrC = yr1C+'0101-'+yr2C+'1231.'
    yrstr.append(yrC)
nfile = np.size(yrstr)
init = True
sind = range(npert)
#%%
for i in range(npert):
    fs = []
    stafile1 = '/archive/Ming.Zhao/awg/verona/c96L32_am4g10r8/gfdl.ncrc3-intel-prod-openmp/pp/atmos_level/atmos_level.static.nc'    
    fs.append(nc.netcdf_file(stafile1,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    plev = fs[-1].variables['phalf'][:].astype(np.float64)
    fs[-1].close()
    
    atmdir = indir+exper+pert[i]+plat+'pp/'+diag+'/'
    stafile = atmdir+diag+'.static.nc'
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    
    zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    if ('land_mask' in fs[-1].variables):
            land_mask = fs[-1].variables['land_mask'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    nlev = np.size(plev)
    #%%
    daydir = atmdir+'ts/daily/5yr/'
    #timeo = '1983-2012'
    mon = np.arange(1,13,1)
    nmon = np.size(mon)
    net_atm = np.zeros((nyr*nmon,nlat,nlon))
    vMSETot = np.zeros((nyr*nmon,nlat,nlon))
    vMSEMC = np.zeros((nyr*nmon,nlat,nlon))
    vMSETr = np.zeros((nyr*nmon,nlat,nlon))
    uMSETot = np.zeros((nyr*nmon,nlat,nlon))
    uMSEMC = np.zeros((nyr*nmon,nlat,nlon))
    uMSETr = np.zeros((nyr*nmon,nlat,nlon))
    wMSEbot = np.zeros((nyr*nmon,nlat,nlon))
    wMSEtop = np.zeros((nyr*nmon,nlat,nlon))
    dMSE = np.zeros((nyr*nmon,nlat,nlon))
    dene = np.zeros((nyr*nmon,nlat,nlon))
    vMC = np.zeros((nyr*nmon,nlev-1,nlat,nlon))
    uMC = np.zeros((nyr*nmon,nlev-1,nlat,nlon))
    wMC = np.zeros((nyr*nmon,nlev-1,nlat,nlon))
    MSEMC = np.zeros((nyr*nmon,nlev-1,nlat,nlon))
    dpMC = np.zeros((nyr*nmon,nlev-1,nlat,nlon))
    pfullMC = np.zeros((nyr*nmon,nlev-1,nlat,nlon))
    first = np.zeros(60)
    last = np.zeros(60)
    first[0] = 0
    last[0] = 30
    nyr1 = nyr
    ptop = 0
    #%%
    for yri in range(nfile):
        days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        if calendar.isleap(yr1[yri]):
            days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        for moni in range(1,12):
            first[moni] = last[moni-1]
            last[moni] = first[moni]+days[moni]
        for moni in range(12,24):
            first[moni] = last[moni-1]
            last[moni] = first[moni]+days[moni-12]
        for moni in range(24,36):
            first[moni] = last[moni-1]
            last[moni] = first[moni]+days[moni-24]
        for moni in range(36,48):
            first[moni] = last[moni-1]
            last[moni] = first[moni]+days[moni-36]
        for moni in range(48,60):
            first[moni] = last[moni-1]
            last[moni] = first[moni]+days[moni-48]
        yrC = yrstr[yri]
        #yrC = str(yr[yri])+'010100-'+str(yr[yri])+'123123.'
        for moni in range(60):        
            # read data
            mind = range(int(first[moni]),int(last[moni]+1))
            dayfile = daydir+diag+'.'+yrC+'vcomp.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            vcomp = fs[-1].variables['vcomp'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'ucomp.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            ucomp = fs[-1].variables['ucomp'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'omega.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            omega = fs[-1].variables['omega'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'temp.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            temp = fs[-1].variables['temp'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()  
            dayfile = daydir+diag+'.'+yrC+'z_full.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            hght = fs[-1].variables['z_full'][mind,:,:,:].astype(np.float64)
            #hght = 0.5*(hght[:,1:,:,:]+hght[:,:-1,:,:])[:,ptop:,:,:]
            fs[-1].close()    
            dayfile = daydir+diag+'.'+yrC+'sphum.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            sphum = fs[-1].variables['sphum'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'ice_wat.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            ice_wat = fs[-1].variables['ice_wat'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'ps.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            ps = fs[-1].variables['ps'][mind,:,:].astype(np.float64)
            fs[-1].close()
            
            dayfile = daydir+diag+'.'+yrC+'olr.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            olr = fs[-1].variables['olr'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'swup_toa.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            swup_toa = fs[-1].variables['swup_toa'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'swdn_toa.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            swdn_toa = fs[-1].variables['swdn_toa'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'swup_sfc.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            swup_sfc = fs[-1].variables['swup_sfc'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'swdn_sfc.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            swdn_sfc = fs[-1].variables['swdn_sfc'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'lwup_sfc.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            lwup_sfc = fs[-1].variables['lwup_sfc'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'lwdn_sfc.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            lwdn_sfc = fs[-1].variables['lwdn_sfc'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'evap.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            evap = fs[-1].variables['evap'][mind,:,:].astype(np.float64)
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'shflx.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            shflx = fs[-1].variables['shflx'][mind,:,:].astype(np.float64)
            fs[-1].close()
            # net atm radiative and turbulent flux
            net = swdn_toa-swup_toa-olr+swup_sfc+lwup_sfc-swdn_sfc-lwdn_sfc+shflx+LE*evap
            net_atm[yri*12+moni,:,:] = np.mean(net,0)
            
            nt = vcomp.shape[0]
            # calculate pressure
            phalf = grid.calcSigmaPres(ps,pk,bk)
            pfull = 0.5*(phalf[:,1:,:,:]+phalf[:,:-1,:,:])
            dp = (phalf[:,1:,:,:]-phalf[:,:-1,:,:])[:,ptop:,:,:]            
            #%%
            # calculate MSE (subtract mean)
            MSE = CP_AIR*temp+GRAV*hght+LE*sphum-LF*ice_wat
            ene = CV_AIR*temp+GRAV*hght+LE*sphum-LF*ice_wat
            MSEm = np.sum(MSE*dp,1)/np.sum(dp,1)
            MSEm = np.mean(MSEm,0)
            area = grid.calcGridArea(lat,lon)
            MSEm = np.sum(MSEm*area)/np.sum(area)
            # MSEm = np.mean(MSE[:,:,:,:])
            MSE = MSE-MSEm
            # calculate dp
            dpMon = np.mean(dp,0)
            dpMC[yri*12+moni,:,:,:] = dpMon
            pfullMC[yri*12+moni,:,:,:] = np.mean(pfull,0)
            MSECol = np.sum(MSE*dpMon,1)/GRAV
            dMSE[yri*12+moni,:,:] = (MSECol[-1,...]-MSECol[0,...])/(np.size(mind)*86400)
            eneCol = np.sum(MSE*dpMon,1)/GRAV
            dene[yri*12+moni,:,:] = (eneCol[-1,...]-eneCol[0,...])/(np.size(mind)*86400)
            vMSE = np.sum(vcomp*MSE*dpMon,1)/GRAV
            vMSETot[yri*12+moni,:,:] = np.mean(vMSE,0)
            vMon = np.mean(vcomp,0)
            MSEMon = np.mean(MSE,0)
            vMC[yri*12+moni,:,:,:] = vMon
            MSEMC[yri*12+moni,:,:,:] = MSEMon
            vTr = vcomp-vMon[np.newaxis,...] 
            MSETr = MSE-MSEMon[np.newaxis,...]
            vMSE = np.sum(vTr*MSETr*dpMon,1)/GRAV
            vMSETr[yri*12+moni,:,:] = np.mean(vMSE,0)
            vMSE = np.sum(MSEMon*vMon*dpMon,0)/GRAV
            vMSEMC[yri*12+moni,:,:] = vMSE
            uMSE = np.sum(ucomp*MSE*dpMon,1)/GRAV
            uMSETot[yri*12+moni,:,:] = np.mean(uMSE,0)
            uMon = np.mean(ucomp,0)
            uMC[yri*12+moni,:,:,:] = uMon
            uTr = ucomp-uMon[np.newaxis,...] 
            uMSE = np.sum(uTr*MSETr*dpMon,1)/GRAV
            uMSETr[yri*12+moni,:,:] = np.mean(uMSE,0)
            uMSE = np.sum(MSEMon*uMon*dpMon,0)/GRAV
            uMSEMC[yri*12+moni,:,:] = uMSE
            wMC[yri*12+moni,:,:,:] = np.mean(omega,0)
            wMSEbot[yri*12+moni,:,:] = (np.mean(omega*MSE,0)[-1,...])/GRAV
            wMSEtop[yri*12+moni,:,:] = (np.mean(omega*MSE,0)[0,...])/GRAV
    #%%
    #mind = [6,7,8]
    """
    if init:
        outfile = outdir+'dim.'+pert_dict[pert[i]]+'.npz'
        fio.save(outfile,lat=lat,lon=lon,phalf=phalf,land_mask=land_mask,year=yr)
    """
    for flagi in range(nflag):
        vhTot = pp.month_to_year(vMSETot,yr_ts,flag[flagi])#*20650/1e19
        vhMC = pp.month_to_year(vMSEMC,yr_ts,flag[flagi])
        vhTr = pp.month_to_year(vMSETr,yr_ts,flag[flagi])
        uhTot = pp.month_to_year(uMSETot,yr_ts,flag[flagi])
        uhMC = pp.month_to_year(uMSEMC,yr_ts,flag[flagi])
        uhTr = pp.month_to_year(uMSETr,yr_ts,flag[flagi])
        whbot = pp.month_to_year(wMSEbot,yr_ts,flag[flagi])
        whtop = pp.month_to_year(wMSEtop,yr_ts,flag[flagi])
        net = pp.month_to_year(net_atm,yr_ts,flag[flagi])
        de = pp.month_to_year(dene,yr_ts,flag[flagi])
        """
        outdir_sub='ts/'+sub_dict[flag[flagi]]+'/'
        outfile = outdir+outdir_sub+diago+'.'+timeo+'.'+pert_dict[pert[i]]+'.npz'
        fio.save(outfile,vMSETot_col=vhTot,vMSEMC_col=vhMC,vMSETr_col=vhTr,\
                 uMSETot_col=uhTot,uMSEMC_col=uhMC,uMSETr_col=uhTr,\
                 wMSEbot=whbot,wMSEtop=whtop,net_atm=net,dene_col=de)
        """
    vm = pp.month_to_year(vMC,yr_ts,flag[flagi])
    um = pp.month_to_year(uMC,yr_ts,flag[flagi])
    wm = pp.month_to_year(wMC,yr_ts,flag[flagi])
    hm = pp.month_to_year(MSEMC,yr_ts,flag[flagi])
    dpm = pp.month_to_year(dpMC,yr_ts,flag[flagi])
    pfullm = pp.month_to_year(pfullMC,yr_ts,flag[flagi])
#%%
vhTot1 = vhMC+vhTr
uhTot1 = uhMC+uhTr
xsc = 2*np.pi*RADIUS*np.cos(lat*np.pi/180.)/360.
ysc = np.pi*RADIUS/180.
sc = 1e-15*360
area = grid.calcGridArea(lat,lon)
glb = np.sum(net*area[np.newaxis,...])/np.sum(area)
plt.plot(lat,np.mean(np.mean(vhTot,0),-1)*xsc*sc,'r')
#plt.plot(lat,np.mean(np.mean(vhTot1,0),-1)*xsc*sc,'k')
plt.plot(lat,np.mean(np.mean(vhMC,0),-1)*xsc*sc,'k--')
plt.plot(lat,np.mean(np.mean(vhTr,0),-1)*xsc*sc,'k:')
plt.plot(lat,np.cumsum(np.mean(np.mean(net-glb,0),-1)*xsc*sc*ysc*2.5),'r--')

#%%
nmon = nyr
latr = lat*np.pi/180.
vhTotr = vhTot*np.cos(latr[np.newaxis,:,np.newaxis])
vhMCr = vhMC*np.cos(latr[np.newaxis,:,np.newaxis])
vhTrr = vhTr*np.cos(latr[np.newaxis,:,np.newaxis])

dvhTotdy = np.zeros((nmon,nlat,nlon))
duhTotdx = np.zeros((nmon,nlat,nlon))
dvhMCdy = np.zeros((nmon,nlat,nlon))
duhMCdx = np.zeros((nmon,nlat,nlon))
dvhTrdy = np.zeros((nmon,nlat,nlon))
duhTrdx = np.zeros((nmon,nlat,nlon))
dhmdy = np.zeros((nmon,nlev-1,nlat,nlon))
dhmdx = np.zeros((nmon,nlev-1,nlat,nlon))
duhTot = np.zeros((nmon,nlat,nlon))
duhMC = np.zeros((nmon,nlat,nlon))
duhTr = np.zeros((nmon,nlat,nlon))
dhmx = np.zeros((nmon,nlev-1,nlat,nlon))
dhmdp = np.zeros((nmon,nlev-1,nlat,nlon))
dy = (lat[2:]-lat[:-2])*ysc
dy.shape = (1,nlat-2,1)
dyr = latr[2:]-latr[:-2]
dyr.shape = (1,nlat-2,1)
dx = np.zeros((nlat,nlon))
for i in range(nlat):
   dx[i,1:-1] = (lon[2:]-lon[:-2])*xsc[i]
   dx[i,0] = dx[i,1]
   dx[i,-1] = dx[i,1]
dx.shape = (1,nlat,nlon)
dvhTot = vhTotr[:,2:,:]-vhTotr[:,:-2,:]
dvhTotdy[:,1:-1,:] = dvhTot/dy/np.cos(latr[np.newaxis,1:-1,np.newaxis])
#dvhTotdy[:,1:-1,:] = dvhTot/dyr/(Rad*np.cos(latr[np.newaxis,1:-1,np.newaxis]))
duhTot[:,:,1:-1] = uhTot[:,:,2:]-uhTot[:,:,:-2]
duhTot[:,:,0] = uhTot[:,:,1]-uhTot[:,:,-1]
duhTot[:,:,-1] = uhTot[:,:,0]-uhTot[:,:,-2]
duhTotdx = duhTot/dx
Totdiv = dvhTotdy+duhTotdx
vertdiv = whbot-whtop
dvhMC = vhMCr[:,2:,:]-vhMCr[:,:-2,:]
dvhMCdy[:,1:-1,:] = dvhMC/dy/np.cos(latr[np.newaxis,1:-1,np.newaxis])
duhMC[:,:,1:-1] = uhMC[:,:,2:]-uhMC[:,:,:-2]
duhMC[:,:,0] = uhMC[:,:,1]-uhMC[:,:,-1]
duhMC[:,:,-1] = uhMC[:,:,0]-uhMC[:,:,-2]
duhMCdx = duhMC/dx
MCdiv = dvhMCdy+duhMCdx
dvhTr = vhTrr[:,2:,:]-vhTrr[:,:-2,:]
dvhTrdy[:,1:-1,:] = dvhTr/dy/np.cos(latr[np.newaxis,1:-1,np.newaxis])
duhTr[:,:,1:-1] = uhTr[:,:,2:]-uhTr[:,:,:-2]
duhTr[:,:,0] = uhTr[:,:,1]-uhTr[:,:,-1]
duhTr[:,:,-1] = uhTr[:,:,0]-uhTr[:,:,-2]
duhTrdx = duhTr/dx
Trdiv = dvhTrdy+duhTrdx

dhmy = hm[:,:,2:,:]-hm[:,:,:-2,:]
dhmdy[:,:,1:-1,:] = dhmy/dy
dhmx[:,:,:,1:-1] = hm[:,:,:,2:]-hm[:,:,:,:-2]
dhmx[:,:,:,0] = hm[:,:,:,1]-hm[:,:,:,-1]
dhmx[:,:,:,-1] = hm[:,:,:,0]-hm[:,:,:,-2]
dhmdx = dhmx/dx
vdh = np.sum(vm*dhmdy*dpm,1)/GRAV
udh = np.sum(um*dhmdx*dpm,1)/GRAV
uvdh = vdh+udh 
#%%           
dhmp = hm[:,2:,:,:]-hm[:,:-2,:,:]
dp = pfullm[:,2:,:,:]-pfullm[:,:-2,:,:]
dhmdp[:,1:-1,:,:] = dhmp/dp
#dhmdp[:,0,:,:] = dhmdp[:,1,:,:]
#dhmdp[:,-1,:,:] = dhmdp[:,-2,:,:]
wdh = np.sum(wm*dhmdp*dpm,1)/GRAV

region_mask = np.zeros((nlat,nlon))
latlim = [[-18,-5]]
lonlim = [[294,320]]
latlim = [[10,20],[10,20]]
lonlim = [[0,40],[342,360]]
nlim = np.shape(latlim)[0]
for i in range(nlim):
    mask = grid.regionMask(lat,lon,latlim[i],lonlim[i])
    region_mask[np.where(mask==1)] = 1
area.shape = ((1,nlat,nlon))
region_mask.shape = ((1,nlat,nlon))
net_rm = net*area*region_mask
net_rm = np.sum(np.sum(net_rm,-1),-1)/np.sum(area*region_mask)
Totdiv_rm = Totdiv*area*region_mask
Totdiv_rm = np.sum(np.sum(Totdiv_rm,-1),-1)/np.sum(area*region_mask)
vertdiv_rm = vertdiv*area*region_mask
vertdiv_rm = np.sum(np.sum(vertdiv_rm,-1),-1)/np.sum(area*region_mask)
MCdiv_rm = MCdiv*area*region_mask
MCdiv_rm = np.sum(np.sum(MCdiv_rm,-1),-1)/np.sum(area*region_mask)
Trdiv_rm = Trdiv*area*region_mask
Trdiv_rm = np.sum(np.sum(Trdiv_rm,-1),-1)/np.sum(area*region_mask)
uvdh_rm = uvdh*area*region_mask
uvdh_rm = np.sum(np.sum(uvdh_rm,-1),-1)/np.sum(area*region_mask)
wdh_rm = wdh*area*region_mask
wdh_rm = np.sum(np.sum(wdh_rm,-1),-1)/np.sum(area*region_mask)
de_rm = de*area*region_mask
de_rm = np.sum(np.sum(de_rm,-1),-1)/np.sum(area*region_mask)
#%%
clev = np.arange(-190,191,20)
m = Basemap(projection='mill',\
            fix_aspect=True,\
            llcrnrlat=-35,urcrnrlat=5,llcrnrlon=-120,urcrnrlon=-20)
latt,lont,nett = pp.grid_for_map(lat,lon,net)
latt,lont,Totdivt = pp.grid_for_map(lat,lon,Totdiv)
latt,lont,det = pp.grid_for_map(lat,lon,de)
latt,lont,MCdivt = pp.grid_for_map(lat,lon,MCdiv)
latt,lont,Trdivt = pp.grid_for_map(lat,lon,Trdiv)
latt,lont,uvdht = pp.grid_for_map(lat,lon,uvdh)
latt,lont,wdht = pp.grid_for_map(lat,lon,wdh)
latt,lont,whbott = pp.grid_for_map(lat,lon,whbot)
latt,lont,whtopt = pp.grid_for_map(lat,lon,whtop)
lons,lats = np.meshgrid(lont,latt)
x,y = m(lons,lats)
cmap = plt.cm.RdYlBu
#%%
plt.figure()
m.drawcoastlines()
m.drawcountries()
m.contourf(x,y,np.mean(nett,0),clev,cmap=cmap,extend='both')
cb = m.colorbar(location='bottom',size='5%',pad='8%')
plt.tight_layout()
plt.figure()
m.drawcoastlines()
m.drawcountries()
m.contourf(x,y,np.mean(Totdivt+det,0),clev,cmap=cmap,extend='both')
cb = m.colorbar(location='bottom',size='5%',pad='8%')
plt.tight_layout()