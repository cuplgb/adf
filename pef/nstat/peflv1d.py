import math, numpy as np
from opt.opr8tr import operator
import pef.nstat.lvconv1d as lvop
from utils.ptyprint import create_inttag
import matplotlib.pyplot as plt

class peflv1d(operator):
  """ Linearly varying PEFs in 1D """

  def __init__(self,n,j,nlag,lags=None,aux=None,verb=False):
    """
    peflv1d constructor

    Parameters
      n    - length of data array
      j    - spacing (jump) between filter blocks
      nlag - number of lags for the filter (filter coefficients)
      lags - input int lag array (optional)
      aux  - input data array that will form the operator D 
    """
    self.__n = n
    # Compute number of filters
    self.nf = int(math.ceil((n-1)/j)) + 1
    # Compute lags for filter
    self.__nlag = nlag
    if(lags):
      self.lags = lags
      if(nlag != lags.shape[0]):
        raise Exception("nlag must be the same as the length of lags array")
    else:
      self.lags = np.arange(nlag,dtype='int32')
    # Build the blocks
    self.b,self.e = self.find_optimal_sizes(n,j,self.nf)
    self.nb = self.b.shape[0]
    if(verb):
      print("Total number of blocks: %d"%(self.nb+1))
      for ib in range(self.nb):
        print("Block %s [b=%s e=%s]"%(create_inttag(ib,self.nb),create_inttag(self.b[ib],np.max(self.b)),create_inttag(self.e[ib],np.max(self.e))))
    # Set the auxiliary image
    if(aux is not None):
      self.__aux = aux

  def find_optimal_sizes(self,n,j,nf):
    """
    Finds the optimal block sizes for linearly varying convolution

    Parameters
      n1 - length of input data array
      j  - spacing between blocks
      nf - total number of filters
    
    Returns
      b - int array containing beginning block indices
      e - int array containing ending block indices
    """
    b = []; e = []
    space = n
    b.append(0)
    for k in range(nf-1):
      sample = int(math.ceil(space/(nf-k-1)))
      e.append(sample + b[k] - 1)
      if(k != nf-2): b.append(e[k] + 1)
      space -= sample
    # Take care of the endoiunt
    e[nf-2] -= self.lags[-1]

    return np.asarray(b,dtype='int32'),np.asarray(e,dtype='int32')

  def plotfilters(self,flts=None,show=True,**kwargs):
    """ Plots the filters on the coarse grid on the image for QC """
    # Make data array
    dtmp = np.zeros(self.__n)
    # Make random filters
    if(flts is not None):
      ftmp = flts
    else:
      ftmp = np.random.rand(self.nf,self.__nlag)
    # Loop over each block
    for ib in range(self.nb):
      if(flts is not None):
        dtmp[self.b[ib]:self.b[ib]+self.__nlag] = ftmp[ib,:]
      else:
        # Put a one at the block beginning
        dtmp[self.b[ib]] = 1.0
        # Fill with random numbers
        dtmp[self.b[ib]:self.b[ib]+self.__nlag] = ftmp[ib,:]
    # Plot the filters
    fig = plt.figure(figsize=(kwargs.get("wbox",14),kwargs.get("hbox",7)))
    ax = fig.gca()
    ax.stem(dtmp)
    ax.set_xlabel(kwargs.get('xlabel',''),fontsize=kwargs.get('labelsize',18))
    ax.set_ylabel('Coefficient',fontsize=kwargs.get('labelsize',18))
    ax.tick_params(labelsize=kwargs.get('labelsize',18))
    if(show):
      plt.show()

  def setaux(self,aux):
    """ Sets the auxilliary image """
    self.__aux = aux

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
    if(self.__nlag != flt.shape[1]):
      raise Exception("number of filter lags (%d) must match nlag passed to constructor (%d)"%(flt.shape[1],self.__nlag))
    if(self.nf != flt.shape[0]):
      raise Exception("number of filters (%d) must match nf passed to constructor (%d)"%(flt.shape[0],self.nf))

    lvop.lvconv1df_fwd(self.nb, self.b, self.e, # Blocks
                       self.__nlag ,self.lags,  # Lags
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
    if(self.__nlag != flt.shape[1]):
      raise Exception("number of filter lags (%d) must match nlag passed to constructor (%d)"%(flt.shape[1],self.__nlag))
    if(self.nf != flt.shape[0]):
      raise Exception("number of filters (%d) must match nf passed to constructor (%d)"%(flt.shape[0],self.nf))

    lvop.lvconv1df_adj(self.nb, self.b, self.e, # Blocks
                       self.__nlag ,self.lags,  # Lags
                       self.__n, self.__aux,    # Data operator
                       flt, dat) 

