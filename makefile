
CXX           = nvc++
CFLAGS        = -Wall
LDFLAGS       = -Wall #-fopenmp
CXXFLAGS      = -O3 -g -acc=gpu -lstdc++ -gpu=cc80 -gpu=managed -fast -fopenmp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcuda -lcusolverMg -lcudart -lcublasLt #hail mary 
# Need cc70 for racecar, cc86 for alleycat/arkanoid
 
default: cpp
 
all: cpp
 
cpp: test fast
 
clean:
	-rm -f test fast *.o *.mod
 
 
#Ya boy David Kotts c++ addition
 
test: test.cpp
	$(CXX) $(CXXFLAGS) -o test test.cpp

fast: fast.cpp utilities.h
	$(CXX) $(CXXFLAGS) -o fast fast.cpp
