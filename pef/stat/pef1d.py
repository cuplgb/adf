import math, numpy as np
from opt.linopt.opr8tr import operator
import pef.stat.conv1d as cnvop
from utils.ptyprint import create_inttag
import matplotlib.pyplot as plt

class pef1d(operator):
  """ PEFs in 1D """

  def __init__(self,n,nlag,lags=None,aux=None,verb=False):
    """
    pef1d constructor

    Parameters
      n    - length of data array
      nlag - number of lags for the filter (filter coefficients)
      lags - input int lag array (optional)
      aux  - input data array that will form the operator D 
    """
    self.__n = n
    # Compute lags for filter
    self.__nlag = nlag
    if(lags):
      self.lags = lags
      if(nlag != lags.shape[0]):
        raise Exception("nlag must be the same as the length of lags array")
    else:
      self.lags = np.arange(nlag,dtype='int32')
    # Set the auxiliary image
    if(aux is not None):
      self.__aux = aux

  def plotfilters(self,flt=None,show=True,**kwargs):
    """ Plots the filter shape for QC """
    # Make data array
    dtmp = np.zeros(self.__n)
    # Make random filters
    if(flt is not None):
      ftmp = flt
    else:
      ftmp = np.random.rand(self.__nlag)
    if(flt is not None):
      dtmp[0:self.__nlag] = ftmp[:]
    else:
      # Put a one at the block beginning
      dtmp[0] = 1.0
      # Fill with random numbers
      dtmp[0:+self.__nlag] = ftmp[:]
    # Plot the filters
    fig = plt.figure(figsize=(kwargs.get("wbox",14),kwargs.get("hbox",7)))
    ax = fig.gca()
    ax.stem(dtmp)
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
    ax.set_ylabel('Coefficient',fontsize=kwargs.get('labelsize',18))
    ax.tick_params(labelsize=kwargs.get('labelsize',18))
    if(show):
      plt.show()

  def set_aux(self,aux):
    """ Sets the auxilliary image """
    self.__aux = aux

  def create_data(self):
    """ Creates the data vector for the PEF estimation """
    # Create a temporary filter
    tflt = np.zeros(self.__nlag,dtype='float32')
    tflt[0] = 1.0
    # Create the data
    dat = np.zeros(self.__n,dtype='float32')
    self.forward(False,tflt,dat)

    return -dat

  def getdims(self):
    """ Returns the dimensions of the PEF and mask operator """
    # PEF dims
    pdims = {}
    pdims['ncols'] = self.__nlag; pdims['nrows'] = self.__n
    # Mask dims
    kdims = {}
    kdims['ncols'] = self.__nlag; kdims['nrows'] = self.__nlag

    return [kdims,pdims]

  def forward(self,add,flt,dat):
    """
    Applies the operator D (constructed from the aux array) that 
    will be convolved with the filter coefficients

    Parameters
      add - whether to add to the output [True/False]
      flt - the filter coefficients to be estimated
      dat - the result of the application of the data operator D to the PEF
    """
    # Check data size
    if(self.__n != dat.shape[0]):
      raise Exception("data shape (%d) must match n passed to constructor(%d)"%(dat.shape[0],self.__n))
    # Check filter size
    if(self.__nlag != flt.shape[0]):
      raise Exception("number of filter lags (%d) must match nlag passed to constructor (%d)"%(flt.shape[0],self.__nlag))

    if(not add):
      dat[:] = 0.0

    cnvop.conv1df_fwd(self.__nlag ,self.lags,  # Lags
                      self.__n, self.__aux,    # Data operator
                      flt, dat) 

  def adjoint(self,add,flt,dat):
    """
    Correlates the data operator D with the dat array to give an estimate
    of the filter coefficients

    Parameters
      add - whether to add to the output filter coefficients [True/False]
      flt - the output filter coefficients
      dat - the input data to be correlated with D
    """
    # Check data size
    if(self.__n != dat.shape[0]):
      raise Exception("data shape (%d) must match n passed to constructor(%d)"%(dat.shape[0],self.__n))
    # Check filter size
    if(self.__nlag != flt.shape[0]):
      raise Exception("number of filter lags (%d) must match nlag passed to constructor (%d)"%(flt.shape[0],self.__nlag))

    if(not add):
      flt[:] = 0.0

    cnvop.conv1df_adj(self.__nlag ,self.lags,  # Lags
                      self.__n, self.__aux,    # Data operator
                      flt, dat)

  def dottest(self,add=False):
    """ Performs the dot product test of the operator """
    # Create model and data
    m  = np.random.rand(self.__nlag).astype('float32')
    mh = np.zeros(m.shape,dtype='float32')
    d  = np.random.rand(self.__n).astype('float32')
    dh = np.zeros(d.shape,dtype='float32')

    if(add):
      self.forward(True,m ,dh)
      self.adjoint(True,mh,d )
      dotm = np.dot(m.flatten(),mh.flatten()); dotd = np.dot(d,dh)
      print("Dot product test (add==True):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))
    else:
      self.forward(False,m ,dh)
      self.adjoint(False,mh,d )
      dotm = np.dot(m.flatten(),mh.flatten()); dotd = np.dot(d,dh)
      print("Dot product test (add==False):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))

class pef1dmask(operator):
  """ Mask operator for not updating the zero lag coefficient """

  def forward(self,add,flt,msk):
    """ Applies the mask to the filter """
    if(flt.shape != msk.shape):
      raise Exception("model and data must have same shape")
    # Set the zero lag to zero
    msk[:] = flt[:]
    msk[0] = 0.0

  def adjoint(self,add,flt,msk):
    """ Applies adjoint mask """
    if(flt.shape != msk.shape):
      raise Exception("model and data must have same shape")
    # Set the zero lag to zero
    flt[:] = msk[:]
    flt[0] = 0.0

