# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:41:51 2015

@author: z1s
"""
import matplotlib.pyplot as plt
import numpy as np
def plotyy(xx,arr1,arr2,**kwargs):
    figsize = [4*16./9,4];
    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    fig,ax1 = plt.subplots(figsize=figsize)
    c1 = 'b';
    c2 = 'g';
    y1 = 'arr1';
    y2 = 'arr2';
    if 'color1' in kwargs.keys():
        c1 = kwargs['color1']
    if 'color2' in kwargs.keys():
        c2 = kwargs['color2']
    ax1.plot(xx,arr1,c1)
    #if 'xlim' in kwargs.keys():
    #    ax1.set_xlim(kwargs['xlim'])
    
    if 'ytick1' in kwargs.keys():
        if 'yticklabel1' in kwargs.keys():
            plt.yticks(kwargs['ytick1'],kwargs['yticklabel1'])
        else:
            plt.yticks(kwargs['ytick1'])
    if 'ylim1' in kwargs.keys():
        ax1.set_ylim(kwargs['ylim1'])
    ax2 = ax1.twinx()
    ax2.plot(xx,arr2,c2) 
    if 'ytick2' in kwargs.keys():
        if 'yticklabel2' in kwargs.keys():
            plt.yticks(kwargs['ytick2'],kwargs['yticklabel2'])
        else:
            plt.yticks(kwargs['ytick2'])
    if 'ylim2' in kwargs.keys():
        ax2.set_ylim(kwargs['ylim2'])
    if 'xtick' in kwargs.keys():
        if 'xticklabel' in kwargs.keys():
            plt.xticks(kwargs['xtick'],kwargs['xticklabel'])
        else:
            ax2.set_xticks(kwargs['xtick'])
    if 'xlim' in kwargs.keys():
        ax2.set_xlim(kwargs['xlim'])
        #ax2.set_xlim(ax1.get_xlim())
    if 'xlabel' in kwargs.keys():
        ax1.set_xlabel(kwargs['xlabel'])
    if 'ylabel1' in kwargs.keys():
        y1 = kwargs['ylabel1']
        ax1.set_ylabel(y1,color=c1)
    if 'ylabel2' in kwargs.keys():
        y2 = kwargs['ylabel2']
        ax2.set_ylabel(y2,color=c2)
    if ('sci' in kwargs.keys()):
        if (kwargs['sci'][0]):
            ax1.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        if (kwargs['sci'][1]):
            ax2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.tight_layout()
    if ('corr' in kwargs.keys()) and (kwargs['corr']):
        print(y1,y2,np.corrcoef(arr1,arr2)[0,1],np.corrcoef(arr1,arr2)[0,1]**2)