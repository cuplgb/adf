import numpy as np
from pef.nstat.peflv1d import peflv1d, peflv1dmask
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

nlag = 2; j = 100
nspef = peflv1d(n,j,nlag,aux=dat,verb=True)

pdat = nspef.create_data()

flt = np.zeros([nspef.nf,nlag],dtype='float32')
flt[:,0] = 1.0

mask = peflv1dmask()
idop = identity()
zro = np.zeros(flt.shape,dtype='float32')

dkop = chainop([mask,nspef],nspef.get_dims())

pefres = []
cd(dkop,pdat,flt,regop=idop,rdat=zro,eps=0.00,niter=20,ress=pefres)

print(flt[0,:],flt[1,:])

plt.plot(pefres[0][0]); plt.show()

