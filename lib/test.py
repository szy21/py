# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:49:37 2016

@author: Zhaoyi.Shen
"""

import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
y1 = [1,2,3,4,5]
y2 = [5,4,3,2,1]

fig,ax1 = plt.subplots()
ax1.plot(x,y1)
#ax1.set_xlim([2,4])
ax2 = ax1.twinx()
ax2.plot(x,y2)
ax2.set_xlim([2,4])