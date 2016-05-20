# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 20:50:22 2015

@author: z1s
"""
import numpy as np
import os

def save(filename,append=True,**kwargs):
    if (os.path.isfile(filename) and append):
        npz = np.load(filename)
        npzdict = dict(npz)
        newdict = dict(npzdict.items()+kwargs.items())
        np.savez(filename,**newdict)
    else:
        np.savez(filename, **kwargs)
        
def delete(filename, *args):
    npz = np.load(filename)
    npzdict = dict(npz)
    for var in args:
        npzdict.pop(var)
    np.savez(filename,**npzdict)