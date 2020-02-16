# Adapted for numpy/ma/cdms2 by convertcdms.py
# LanczosFilter.py

"""Multivariate Lanczos filter

"""
# Copyright (C) 2000, Jon Saenz, Juan Zubillaga and Jesus Fernandez
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# 
# Jon Saenz, 20000305
# Juan Zubillaga, 20000324
# Jon Saenz, 20000616
# Jesus Fernandez, 20000724 
#
# For notation and mathematical details, see Duchon, 1979, Journal of Applied 
# Meteorology, Lanczos filtering in one and two dimensions, volume 18, 
# pages=1016-1022

import sys
sys.path.append('/home/z1s/py/lib/')

import numpy
import LinearFilter

import math

class LanczosFilter(LinearFilter.LinearFilter):
  """'LinearFilter' derived class for Lanczos filtering

  For notation and mathematical details, see Duchon, 1979, Journal of Applied 
  Meteorology, *Lanczos filtering in one and two dimensions*, volume 18, 
  pages 1016-1022.
  """
  def __init__(self,filtertype='bp',fc1=0.0,fc2=0.5,n=None):
    """Constructor for the class 'LanczosFilter'

    Initialize the variables to hold the needed data
    The filter works as follows, first, it gets the
    coefficients to be used to weight each record performing a
    recursive filtering on the impulse response function
    Next, filter the records by means of a running buffer which is
    weighted using the weights... and that's all

    Optional Arguments:

      'filtertype' -- String identifying the filter type. 'lp' for 
                      low-pass filter, 'hp' for high-pass filter and
                      'bp' for band-pass filter.

      'fc1' -- First cutoff frequency (in inverse time steps).
               Defaults to 0.0

      'fc2' -- Second cutoff frequency (in inverse time steps).
               Defaults to 0.5

      'n' -- Number of points for the filter. This number of points will
             be missed at the beginning and at the end of the raw record.
             Defaults to a number a 30% higher than that recomended for
             secure band-pass filtering by Duchon (1979).

    NOTE: With the default arguments the filter does NOT filter at all.
    It performs a band-pass filtering for all frequencies.
    
    NOTE 2: If the first and second cutoff frequencies are different 
    band-pass filtering is assumed even though 'lp' or 'hp' filtertype were 
    selected.
    """
    if fc1 != fc2:
      self.filtertype='bp'
    else:
      self.filtertype=filtertype
    # Accounts for the three types of filter
    self.fc1=fc1
    self.fc2=fc2
    if self.filtertype=='lp':
      self.fc1=0.0
    elif self.filtertype=='hp':
      self.fc2=0.0
    # Length of the buffer and the coefficients of the filter
    if n!=None:
       self.length=2*n+1
    else:
       # 30% higher than recommended for band-pass
       self.length=2*int(1.3*(1.3/abs(self.fc2-self.fc1)))+1
    # Get the filter coefficients
    self.coefs=self.getcoefs()
    # Position in the buffer to store the input datafield
    self.target=0
    # Output record id
    self.record=0


  def _place(self,k,n,pm1):
    return n+k*pm1
 
  def getcoefs(self):
    "Filter coefficients"
    n=self.length/2
    thepi=math.acos(-1.)
    ocoefs=numpy.zeros(self.length,numpy.float64)
    # This is pretty singular...
    # sinc(0)=1!!; sin(2fx)/x = 2f!!
    k=0
    ocoefs[self._place(k,n,1)]=2*(self.fc2-self.fc1)*1.
    # Slight modification for the high pass
    if self.filtertype=='hp':
      ocoefs[self._place(k,n,1)]=1+ocoefs[self._place(k,n,1)]
    for ik in xrange(n):
      k=ik+1
      sigma=math.sin(thepi*k/n)*n/thepi/k
      firstfactor=(math.sin(2*thepi*self.fc2*k)-
                   math.sin(2*thepi*self.fc1*k))/thepi/k
      ocoefs[self._place(k,n,1)]=firstfactor*sigma
      ocoefs[self._place(k,n,-1)]=firstfactor*sigma
    return ocoefs


  # Now, filter real data and compare with calibration results.
  # Both high pass and low-pass versions
  