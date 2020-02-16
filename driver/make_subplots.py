# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:00:50 2015

@author: z1s
"""

import sys
sys.path.append('/home/z1s/py/lib/')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
fsize = 14
font = {'size'   : fsize}
mpl.rc('font', size=fsize)
mpl.rc('lines', linewidth=2)
mpl.rc('figure', figsize=[8*3.5/3.,8])
mpl.rc('font', family='sans-serif')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['contour.corner_mask'] = False

figsize = [9,9*3.25/3]
left = 0.1
right = 0.9
bottom = 0.14
top = 0.98
hspace = 0.6
wspace = 0.1

fig,axes = plt.subplots(nrows=4,ncols=3,gridspec_kw = {'height_ratios':[2.6,1,1,1]},figsize=figsize)
axes[0,0].axis('off')
axes[0,2].axis('off')
cax_obs = fig.add_axes([0.32,0.76,0.36,0.016])
cax = fig.add_axes([0.32,0.1,0.36,0.016])

plt.tight_layout() 
fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,hspace=hspace,wspace=wspace)

