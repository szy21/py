# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:22:19 2016

@author: Zhaoyi.Shen
"""

import numpy as np

def swobs_col(filename):
    npz = np.load(filename)
    swup_toa = npz['swup_toa']
    swdn_toa = npz['swdn_toa']
    swup_sfc = npz['swup_sfc']
    swdn_sfc = npz['swdn_sfc']
    return swdn_toa-swdn_sfc+swup_sfc-swup_toa
    
def swobs_col_clr(filename):
    npz = np.load(filename)
    swup_toa = npz['swup_toa_clr']
    swdn_toa = npz['swdn_toa_clr']
    swup_sfc = npz['swup_sfc_clr']
    swdn_sfc = npz['swdn_sfc_clr']
    return swdn_toa-swdn_sfc+swup_sfc-swup_toa
    
def lwobs_col(filename):
    npz = np.load(filename)
    olr = npz['olr']
    lwup_sfc = npz['lwup_sfc']
    lwdn_sfc = npz['lwdn_sfc']
    return -lwdn_sfc+lwup_sfc-olr

def lwobs_col_clr(filename):
    npz = np.load(filename)
    olr = npz['olr_clr']
    lwup_sfc = npz['lwup_sfc_clr']
    lwdn_sfc = npz['lwdn_sfc_clr']
    return -lwdn_sfc+lwup_sfc-olr
    
    