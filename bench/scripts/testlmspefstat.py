import numpy as np
from pef.nstat.peflms1d import peflms1d
import inpout.seppy as seppy
import matplotlib.pyplot as plt

sep = seppy.sep([])

n = 200
dat = np.zeros(n,dtype='float32')

axes = seppy.axes([n],[0.0],[1.0])

dat[0] = 1

for i in range(1,n):
  dat[i] = dat[i-1]*0.8

plt.plot(dat); plt.show()

nw = 1

w0 = np.zeros(nw)

mu = 0.5

w,pred,err = peflms1d(dat,nw,mu,w0=w0)

print(w)

plt.plot(err); plt.show()


