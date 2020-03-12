import numpy as np

def peflms1d(x,nw,mu,w0=None):
  """
  Performs PEF estimation in 1D using the LMS algorithm

  Parameters:
    x     - the input signal
    nw    - the number of filter coefficients
    mu    - adaptation constant
    w0    - an initial guess for w [None]

  Returns an estimate for w
  """
  n = x.shape[0]

  # Create output arrays
  err = np.zeros(n); pred = np.zeros(n)
  ws  = np.zeros([n,nw])

  w = np.zeros(nw); 
  if(w0 is not None): w[:] = w0

  # Loop over all samples
  for k in range(nw,n):
    xk = np.flip(x[k-nw:k],0)
    pred[k] = np.dot(xk,w)
    err[k]  = x[k] - pred[k]
    w = w + 2*mu*err[k]*xk
    # Save the ws
    ws[k,:] = w

  return w,ws,pred,err

