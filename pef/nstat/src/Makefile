
%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -o $@ $<

lvconv1d: lvconv1d.o 
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g lvconv1d.o lvconv1dpy.cpp -o ../lvconv1d`${CONDA_PREFIX}/bin/python3-config --extension-suffix`
