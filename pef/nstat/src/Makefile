PYPATH=/opt/anaconda3/envs/py35
NPPATH=${PYPATH}/lib/python3.5/site-packages/numpy/core
MPL=/opt/matplotlib-cpp

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I${PYPATH}/include/python3.5m \
		-I${NPPATH}/include -I${MPL}/ -Wl,-rpath=${PYTPATH}/lib -L${PYPATH}/lib -lpython3.5m \
		-I. -c -fPIC -o $@ $<

lvconv1d: lvconv1d.o 
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g lvconv1d.o lvconv1dpy.cpp -o ../lvconv1d`${PYPATH}/bin/python3-config --extension-suffix`

lvconv2d: lvconv2d.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g lvconv2d.o lvconv2dpy.cpp -o ../lvconv2d`${PYPATH}/bin/python3-config --extension-suffix`
