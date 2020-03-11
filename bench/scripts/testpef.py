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

dat[0] = 1

for i in range(1,n):
  dat[i] = dat[i-1]*0.9

nlag = 2
spef = pef1d(n,nlag,aux=dat)

pdat = spef.create_data()

flt = np.zeros(nlag,dtype='float32')
flt[0] = 1.0

mask = pef1dmask()
idop = identity()
zro = np.zeros(nlag)

dkop = chainop([mask,spef],spef.getdims())

cd(dkop,pdat,flt,regop=idop,rdat=zro,eps=0.0,niter=20)

print(flt)

