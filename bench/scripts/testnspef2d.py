import numpy as np
from pef.nstat.peflv2d import peflv2d, peflv2dmask
from opt.linopt.essops.identity import identity
from opt.linopt.combops import chainop
from opt.linopt.cd import cd
import inpout.seppy as seppy
import matplotlib.pyplot as plt

sep = seppy.sep([])

faxes,fab = sep.read_file(None,"fabricmine.H")
fab = fab.reshape(faxes.n,order='F')

fab = np.ascontiguousarray(fab.T)

nd = fab.shape
nlag = [11,2]; j = [10,10]

nspef = peflv2d(nd,j,nlag,aux=fab,verb=False)

dat = nspef.create_data()

#nspef.dottest()

flt = np.zeros([nspef.nf[0],nspef.nf[1],nspef.nlag],dtype='float32')
flt[:,:,0] = 1.0

mask = peflv2dmask()
idop = identity()

zro = np.zeros(flt.shape,dtype='float32')

dkop = chainop([mask,nspef],nspef.get_dims())

pefres = []
cd(dkop,dat,flt,regop=idop,rdat=zro,eps=0.01,niter=200,ress=pefres)

plt.imshow(pefres[0][0],cmap='gray')
plt.show()

