# LinearFilter.py

"""Generic iteration for a linear filter

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
# Jesus Fernandez, 20000724

import numpy as Numeric

class LinearFilter:
  """Parent class to implementing linear filtering

  It cannot be instanciated. It is intended to be the mother class
  of another getting the coefficients of the filter.

  Any class derived from 'LinearFilter' _MUST_ define an attribute
  'coefs' storing the coefficients of the filter and an attribute
  'length' with the number of coeffients.
  """
  def reset(self):
    """Reset the linear filter

    After the reset() the filter can be used to start computing with
    a data set of a different shape
    """
    self.target=0
    self.multicoefs=None
    self.buffer=None

  def getfiltered(self,ifield,tcode=Numeric.float64):
    """Get a filtered record out of a raw one

    This method returns 'None' while not enough records have been passed to
    the filter. 'npoints' records are missed in the filtering process so
    the first valid (not 'None') value returned corresponds to the record
    'npoints' in the original field and is returned when 2*'npoints'+1 records
    have been passed to this function.

    Argument:

      'ifield' -- Piece of data to be filtered. It can be a 
                  multimensional array.

    Optional argument:

      'tcode' -- Numeric typecode for the internal cumputations
    """
    #######################################
    # The first time, initialize the buffer
    #######################################
    if self.target==0:
      # Initialize the input buffer
      self.buffer=Numeric.zeros((self.length,)+ifield.shape,tcode)
      # And create a Numeric array for the
      # coefficients with the needed shape
      self.multicoefs=Numeric.zeros(self.buffer.shape,Numeric.float64)
      for irec in xrange(self.length):
        self.multicoefs[irec]=self.multicoefs[irec]+self.coefs[irec]
    #########################################################
    # If the buffer needs being filled, do it and return None
    #########################################################
    if self.target<self.length:
      self.buffer[self.target]=Numeric.array(ifield,tcode)
      ################################################
      # The buffer is filled for the first time, 
      # so fill it, DON'T update self.target, 
      # it is not needed anymore, return a valid value
      ################################################
      if self.target==self.length-1:
        self.buffer[self.target]=Numeric.array(ifield,tcode)
        self.target=self.target+1
        return self.__thevalues()
      else:
        ############################################
        # It is empty (yet), point to the next point
        # and return None
        ############################################
        self.target=self.target+1
        return None
    self.buffer=Numeric.concatenate((self.buffer[1:],[ifield]))
    return self.__thevalues()

  # This returns the values taking into account that the buffer is
  # already complete but it can have LOTS of dimensions, so, be
  # extremely careful with this function.... if you touch it
  def __thevalues(self):
    return Numeric.add.reduce(self.multicoefs*self.buffer)
