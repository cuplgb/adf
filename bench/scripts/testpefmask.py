import numpy as np
from pef.nstat.peflv1d import peflv1d, peflv1dmask
from opt.combops import chainop
import matplotlib.pyplot as plt

# Create filter
nf = 5; nlag = 10
flt = np.zeros([nf,nlag],dtype='float32')
flt2 = np.zeros(flt.shape,dtype='float32')

flt[4,:] = np.asarray([0,0,0,0,0,0,0,0,0,0])
flt[3,:] = np.asarray([8,0,0,0,0,0,0,0,0,0])
flt[2,:] = np.asarray([8,4,0,0,0,0,0,0,0,0])
flt[1,:] = np.asarray([8,4,2,1,0,0,0,0,0,0])
flt[0,:] = np.asarray([8,7,6,5,4,3,2,1,0,0])

n = 100; j = 25
auxin = np.zeros(n,dtype='float32')
auxin[0] = 1; auxin[2] = 1

out = np.zeros(n,dtype='float32')

nspef1d = peflv1d(n,j,nlag,aux=auxin,verb=True)
mskop = peflv1dmask()
ops = [mskop,nspef1d]; 

pdim = {}; mdim = {};
pdim['nrows'] = out.shape; pdim['ncols'] = flt.shape
mdim['nrows'] = flt.shape; mdim['ncols'] = flt.shape

dims = [mdim,pdim]

dkop = chainop(ops,dims)

dkop.forward(False,flt,out)

dkop.adjoint(False,flt2,out)

plt.figure(1)
plt.stem(out)

flt2[:,0] = 1.0
nspef1d.plotfilters(flts=flt2,show=False)

plt.show()

m  = np.random.rand(*flt.shape).astype('float32')
mh = np.zeros(flt.shape,dtype='float32')
d  = np.random.rand(*out.shape).astype('float32')
dh = np.zeros(out.shape,dtype='float32')

dkop.forward(False,m ,dh)
dkop.adjoint(False,mh,d )

print(np.dot(m.flatten(),mh.flatten()))

print(np.dot(d.flatten(),dh.flatten()))

