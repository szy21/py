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
import numpy as np
import calendar
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

Cp = 1004.64
LE = 2.500e6
LF = 3.34e5
g = 9.80
Rair = 287.04
Rad = 6371.0e3
basedir = '/archive/s1h/am2/'
exper = 'am2clim_'
pert = ['reyoi']
npert = np.size(pert)
plat = '/gfdl.ncrc2-default-prod/'
diag = 'atmos_inst'

for i in range(npert):
    atmdir = basedir+exper+pert[i]+plat+'pp/'+diag+'/'
    stafile = atmdir+diag+'.static.nc'
    fs = []
    fs.append(nc.netcdf_file(stafile,'r',mmap=True))
    bk = fs[-1].variables['bk'][:].astype(np.float64)
    pk = fs[-1].variables['pk'][:].astype(np.float64)
    lat = fs[-1].variables['lat'][:].astype(np.float64)
    lon = fs[-1].variables['lon'][:].astype(np.float64)
    plev = np.array((1000,925,850,700,600,500,400,300,250,200,150,100,70,50,30,20,10))*100.
    dplev = plev[:-1]-plev[1:]
    dplev = np.concatenate((dplev,[plev[-1]]),axis=0)
    nlev = np.size(dplev)+1
    zsurf = fs[-1].variables['zsurf'][:].astype(np.float64)
    fs[-1].close()
    nlat = np.size(lat)
    nlon = np.size(lon)
    #%%
    daydir = atmdir+'ts/3hr/1yr/'
    yr = np.arange(2000,2001,1)
    #yr = np.arange(2008,2014,1)
    nyr = np.size(yr)
    mon = np.arange(1,13,1)
    nmon = np.size(mon)
    netatm = np.zeros((nyr,nmon,nlat,nlon))
    vMSETot = np.zeros((nyr,nmon,nlat,nlon))
    vMSEMMC = np.zeros((nyr,nmon,nlat,nlon))
    vMSETr = np.zeros((nyr,nmon,nlat,nlon))
    uMSETot = np.zeros((nyr,nmon,nlat,nlon))
    uMSEMMC = np.zeros((nyr,nmon,nlat,nlon))
    uMSETr = np.zeros((nyr,nmon,nlat,nlon))
    wMSEbot = np.zeros((nyr,nmon,nlat,nlon))
    wMSEtop = np.zeros((nyr,nmon,nlat,nlon))
    dMSE = np.zeros((nyr,nmon,nlat,nlon))
    vMMC = np.zeros((nyr,nmon,nlev-1,nlat,nlon))
    uMMC = np.zeros((nyr,nmon,nlev-1,nlat,nlon))
    wMMC = np.zeros((nyr,nmon,nlev-1,nlat,nlon))
    MSEMMC = np.zeros((nyr,nmon,nlev-1,nlat,nlon))
    dpMMC = np.zeros((nyr,nmon,nlev-1,nlat,nlon))
    pfullMMC = np.zeros((nyr,nmon,nlev-1,nlat,nlon))
    first = np.zeros(12)
    last = np.zeros(12)
    first[0] = 0
    last[0] = 30*8
    nyr1 = nyr
    ptop = 0
    #%%
    for yri in range(nyr1):
        if calendar.isleap(yr[yri]):
            days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])*8
        else:
            days = np.array([31,28,31,30,31,30,31,31,30,31,30,31])*8
        for moni in range(1,12):
            first[moni] = last[moni-1]+1
            last[moni] = first[moni]+days[moni]-1
        yrC = str(yr[yri])+'010100-'+str(yr[yri])+'123123.'
        for moni in [6,7,8]:
            mind = range(int(first[moni]),int(last[moni]+1))
            dayfile = daydir+diag+'.'+yrC+'vcomp.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            vcomp = fs[-1].variables['vcomp'][mind,ptop:,:,:].astype(np.float64)
            vcomp[np.where(vcomp<-999)] = np.nan
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'ucomp.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            ucomp = fs[-1].variables['ucomp'][mind,ptop:,:,:].astype(np.float64)
            ucomp[np.where(ucomp<-999)] = np.nan
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'omega.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            omega = fs[-1].variables['omega'][mind,ptop:,:,:].astype(np.float64)
            omega[np.where(omega<-999)] = np.nan
            fs[-1].close()
            dayfile = daydir+diag+'.'+yrC+'temp.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            temp = fs[-1].variables['temp'][mind,ptop:,:,:].astype(np.float64)
            fs[-1].close()  
            dayfile = daydir+diag+'.'+yrC+'hght.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            hght = fs[-1].variables['hght'][mind,:,:,:].astype(np.float64)
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
            #radiation
            dayfile = daydir+diag+'.'+yrC+'netrad_toa.nc'
            fs.append(nc.netcdf_file(dayfile,'r',mmap=True))
            netrad_toa = fs[-1].variables['netrad_toa'][mind,:,:].astype(np.float64)
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
            #start calculation
            net = netrad_toa+swup_sfc+lwup_sfc-swdn_sfc-lwdn_sfc+shflx+LE*evap
            netatm[yri,moni,:,:] = np.mean(net,0)
            nt = vcomp.shape[0]
            
            #%%
            dplev.shape = [1,nlev-1,1,1]         
            MSE = Cp*temp+g*hght+LE*sphum-LF*ice_wat
            MSE[np.where(MSE<0)] = np.nan
            MSEm = np.nanmean(MSE*dplev,1)*(nlev-1)/np.sum(dplev)
            MSEm = np.mean(MSEm,0)
            area = grid.calcGridArea(lat,lon)
            MSEm = np.nansum(MSEm*area)/np.sum(area)
            #MSEm = np.nanmean(MSE[:,:,:,:])
            MSE = MSE-MSEm           
            MSEcol = np.nansum(MSE*dplev,1)/g
            dMSE[yri,moni,:,:] = (MSEcol[-1,...]-MSEcol[0,...])/(np.size(mind)*3*3600)
            vMSE = vcomp*MSE #time,p,lat,lon
            vMSE = np.nansum(vMSE*dplev,1)/g
            vMSETot[yri,moni,:,:] = np.mean(vMSE,0)
            vMon = np.nanmean(vcomp,0)
            MSEMon = np.nanmean(MSE,0)
            vMMC[yri,moni,:,:,:] = vMon
            MSEMMC[yri,moni,:,:,:] = MSEMon
            vTr = vcomp-vMon[np.newaxis,...] 
            MSETr = MSE-MSEMon[np.newaxis,...]
            vMSE = np.nansum(vTr*MSETr*dplev,1)/g
            vMSETr[yri,moni,:,:] = np.mean(vMSE,0)
            vMSE = np.nansum(MSEMon*vMon*dplev[0,...],0)/g
            vMSEMMC[yri,moni,:,:] = vMSE  
            uMSE = ucomp*MSE #time,p,lat,lon
            uMSE = np.nansum(uMSE*dplev,1)/g
            uMSETot[yri,moni,:,:] = np.mean(uMSE,0)
            uMon = np.nanmean(ucomp,0)
            uMMC[yri,moni,:,:,:] = uMon
            uTr = ucomp-uMon[np.newaxis,...] 
            uMSE = np.nansum(uTr*MSETr*dplev,1)/g
            uMSETr[yri,moni,:,:] = np.mean(uMSE,0)
            uMSE = np.nansum(MSEMon*uMon*dplev[0,...],0)/g
            uMSEMMC[yri,moni,:,:] = uMSE
            wMMC[yri,moni,:,:,:] = np.nanmean(omega,0)
            wMSEbot[yri,moni,:,:] = (np.nanmean(omega*MSE,0)[-1,...])/g
            wMSEtop[yri,moni,:,:] = (np.nanmean(omega*MSE,0)[0,...])/g
    #%%
    mind = [6,7,8]
    vhTot = np.mean(vMSETot[:nyr1,mind,:,:],0)#*20650/1e19
    vhMMC = np.mean(vMSEMMC[:nyr1,mind,:,:],0)
    vhTr = np.mean(vMSETr[:nyr1,mind,:,:],0)
    uhTot = np.mean(uMSETot[:nyr1,mind,:,:],0)
    uhMMC = np.mean(uMSEMMC[:nyr1,mind,:,:],0)
    uhTr = np.mean(uMSETr[:nyr1,mind,:,:],0)
    whbot = np.mean(wMSEbot[:nyr1,mind,:,:],0)
    whtop = np.mean(wMSEtop[:nyr1,mind,:,:],0)
    net = np.mean(netatm[:nyr1,mind,:,:],0)
    dh = np.mean(dMSE[:nyr1,mind,:,:],0)
    vMC = np.mean(vMMC[:nyr1,mind,...],0)
    uMC = np.mean(uMMC[:nyr1,mind,...],0)
    wMC = np.mean(wMMC[:nyr1,mind,...],0)
    hMC = np.mean(MSEMMC[:nyr1,mind,...],0)
    dpMC = np.mean(dpMMC[:nyr1,mind,...],0)
    pfullMC = np.mean(pfullMMC[:nyr1,mind,...],0)
#%%
vhTot1 = (vhMMC+vhTr)
uhTot1 = uhMMC+uhTr
xsc = 2*np.pi*Rad*np.cos(lat*np.pi/180.)/360.
ysc = np.pi*Rad/180.
sc = 1e-15
plt.figure()
plt.plot(lat,np.mean(np.mean(vhTot,0),-1)*xsc*sc*360,'r')
#plt.plot(lat,np.mean(np.mean(vhTot1,0),-1)*xsc*sc,'k')
#plt.plot(lat,np.mean(np.mean(vhMMC,0),-1)*xsc*sc*360,'k--')
plt.plot(lat,np.mean(np.mean(vhTr,0),-1)*xsc*sc*360,'k:')

#%%
nmon = np.size(mind)
latr = lat*np.pi/180.
vhTotr = vhTot*np.cos(latr[np.newaxis,:,np.newaxis])
vhMMCr = vhMMC*np.cos(latr[np.newaxis,:,np.newaxis])
vhTrr = vhTr*np.cos(latr[np.newaxis,:,np.newaxis])

dvhTotdy = np.zeros((nmon,nlat,nlon))
duhTotdx = np.zeros((nmon,nlat,nlon))
dvhMMCdy = np.zeros((nmon,nlat,nlon))
duhMMCdx = np.zeros((nmon,nlat,nlon))
dvhTrdy = np.zeros((nmon,nlat,nlon))
duhTrdx = np.zeros((nmon,nlat,nlon))
dhMCdy = np.zeros((nmon,nlev-1,nlat,nlon))
dhMCdx = np.zeros((nmon,nlev-1,nlat,nlon))
duhTot = np.zeros((nmon,nlat,nlon))
duhMMC = np.zeros((nmon,nlat,nlon))
duhTr = np.zeros((nmon,nlat,nlon))
dhMCx = np.zeros((nmon,nlev-1,nlat,nlon))
dhMCdp = np.zeros((nmon,nlev-1,nlat,nlon))
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
dvhMMC = vhMMCr[:,2:,:]-vhMMCr[:,:-2,:]
dvhMMCdy[:,1:-1,:] = dvhMMC/dy/np.cos(latr[np.newaxis,1:-1,np.newaxis])
duhMMC[:,:,1:-1] = uhMMC[:,:,2:]-uhMMC[:,:,:-2]
duhMMC[:,:,0] = uhMMC[:,:,1]-uhMMC[:,:,-1]
duhMMC[:,:,-1] = uhMMC[:,:,0]-uhMMC[:,:,-2]
duhMMCdx = duhMMC/dx
MMCdiv = dvhMMCdy+duhMMCdx
dvhTr = vhTrr[:,2:,:]-vhTrr[:,:-2,:]
dvhTrdy[:,1:-1,:] = dvhTr/dy/np.cos(latr[np.newaxis,1:-1,np.newaxis])
duhTr[:,:,1:-1] = uhTr[:,:,2:]-uhTr[:,:,:-2]
duhTr[:,:,0] = uhTr[:,:,1]-uhTr[:,:,-1]
duhTr[:,:,-1] = uhTr[:,:,0]-uhTr[:,:,-2]
duhTrdx = duhTr/dx
Trdiv = dvhTrdy+duhTrdx

dhMCy = hMC[:,:,2:,:]-hMC[:,:,:-2,:]
dhMCdy[:,:,1:-1,:] = dhMCy/dy
dhMCx[:,:,:,1:-1] = hMC[:,:,:,2:]-hMC[:,:,:,:-2]
dhMCx[:,:,:,0] = hMC[:,:,:,1]-hMC[:,:,:,-1]
dhMCx[:,:,:,-1] = hMC[:,:,:,0]-hMC[:,:,:,-2]
dhMCdx = dhMCx/dx
vdh = np.sum(vMC*dhMCdy*dpMC,1)/g
udh = np.sum(uMC*dhMCdx*dpMC,1)/g
uvdh = vdh+udh 
#%%           
#dhMCp = hMC[:,4:,:,:]-8*hMC[:,3:-1,:,:]+8*hMC[:,1:-3,:,:]-hMC[:,:-4,:,:]
#dp = (pfullMC[:,4:,:,:]-pfullMC[:,:-4,:,:])*-3
dhMCp = hMC[:,2:,:,:]-hMC[:,:-2,:,:]
dp = pfullMC[:,2:,:,:]-pfullMC[:,:-2,:,:]
dhMCdp[:,1:-1,:,:] = dhMCp/dp
wdh = np.sum(wMC[:,:,:,:]*dhMCdp[:,:,:,:]*dpMC[:,:,:,:],1)/g
area = grid.calcGridArea(lat,lon)
region_mask = np.zeros((nlat,nlon))
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
MMCdiv_rm = MMCdiv*area*region_mask
MMCdiv_rm = np.sum(np.sum(MMCdiv_rm,-1),-1)/np.sum(area*region_mask)
Trdiv_rm = Trdiv*area*region_mask
Trdiv_rm = np.sum(np.sum(Trdiv_rm,-1),-1)/np.sum(area*region_mask)
uvdh_rm = uvdh*area*region_mask
uvdh_rm = np.sum(np.sum(uvdh_rm,-1),-1)/np.sum(area*region_mask)
wdh_rm = wdh*area*region_mask
wdh_rm = np.sum(np.sum(wdh_rm,-1),-1)/np.sum(area*region_mask)
dh_rm = dh*area*region_mask
dh_rm = np.sum(np.sum(dh_rm,-1),-1)/np.sum(area*region_mask)
#%%
clev = np.arange(-190,191,20)
m = Basemap(projection='mill',\
            fix_aspect=True,\
            llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
latt,lont,nett = pp.grid_for_map(lat,lon,net)

latt,lont,Totdivt = pp.grid_for_map(lat,lon,Totdiv)
latt,lont,MMCdivt = pp.grid_for_map(lat,lon,MMCdiv)
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
m.contourf(x,y,np.mean(-nett,0),clev,cmap=cmap,extend='both')
cb = m.colorbar(location='bottom',size='5%',pad='8%')
plt.tight_layout()
plt.figure()
m.drawcoastlines()
m.drawcountries()
m.contourf(x,y,np.mean(Totdivt,0),clev,cmap=cmap,extend='both')
cb = m.colorbar(location='bottom',size='5%',pad='8%')
plt.tight_layout()