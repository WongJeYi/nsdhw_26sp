CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC \
           $(shell python3-config --includes) \
           -I$(shell python3 -c "import numpy; print(numpy.get_include())") \
           -I/usr/include/pybind11 \
           -I/usr/include/mkl

LDFLAGS = -Wl,--no-as-needed \
          -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
          -lpthread -lm -ldl

TARGET = _matrix$(shell python3-config --extension-suffix)

.PHONY: all clean

all: $(TARGET)

$(TARGET): mod.cpp matrix.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f _matrix*.so _matrix*.pyd