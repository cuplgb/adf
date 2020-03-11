import numpy as np
from pef.stat.pef1d import pef1d, pef1dmask
from opt.linopt.essops.identity import identity
from opt.linopt.combops import chainop
from opt.linopt.cd import cd
import inpout.seppy as seppy
import matplotlib.pyplot as plt

sep = seppy.sep([])

n = 200
dat = np.zeros(n,dtype='float32')

axes = seppy.axes([n],[0.0],[1.0])

dat[0] = 1; dat[100] = 1

for i in range(1,100):
  dat[i] = dat[i-1]*0.8

for i in range(101,n):
  dat[i] = dat[i-1]*0.5

plt.plot(dat); plt.show()

nlag = 2
spef = pef1d(n,nlag,aux=dat)

pdat = spef.create_data()

flt = np.zeros(nlag,dtype='float32')
flt[0] = 1.0

mask = pef1dmask()
idop = identity()
zro = np.zeros(flt.shape,dtype='float32')

dkop = chainop([mask,spef],spef.get_dims())

pefres = []
cd(dkop,pdat,flt,regop=idop,rdat=zro,eps=0.00,niter=20,ress=pefres)

plt.plot(pefres[0][0]); plt.show()

print(flt)

