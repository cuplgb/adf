import numpy as np
from pef.stat.pef1d import pef1d
import matplotlib.pyplot as plt

# Create filter
nlag = 3
flt = np.zeros(nlag,dtype='float32')

flt[:] = np.asarray([1,1,1])

n = 3
auxin = np.zeros(n,dtype='float32')
auxin[0] = 1; auxin[1] = 1; auxin[2] = 1

out = np.zeros(n,dtype='float32')

spef1d = pef1d(n,nlag,aux=auxin,verb=True)

#spef1d.plotfilters(flt=flt)

spef1d.forward(False,flt,out)

#spef1d.dottest()

plt.figure(1)
plt.stem(auxin)

plt.figure(2)
plt.stem(out)

plt.show()

