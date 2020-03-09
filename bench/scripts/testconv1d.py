import numpy as np
from pef.nstat.peflv1d import peflv1d
import matplotlib.pyplot as plt

# Create filter
nf = 5; nlag = 10
flt = np.zeros([nf,nlag],dtype='float32')

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

nspef1d.plotfilters(flts=flt)

nspef1d.forward(False,flt,out)

plt.figure(1)
plt.stem(auxin)

plt.figure(2)
plt.stem(out)

plt.show()

