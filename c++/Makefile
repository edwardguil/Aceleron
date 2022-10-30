CC := g++
NVCC := nvcc
CFLAGS := -Wall -pedantic -g -std=c++11
NVFLAGS = --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -pg -g
DEPEND := layers.cu losses.cu metrics.cu optimizers.cu matrix.cu cuda.cu
OPTIMISE = -O3

all: driverp driver tests

driver : driver.o
	$(NVCC) $(OPTIMISE) $(NVFLAGS) driver.o -o $@

driver.o: driver.cu  $(DEPEND)
	$(NVCC) $(OPTIMISE) $(NVFLAGS) -c -o driver.o driver.cu

driverp : driverp.o
	$(NVCC) $(OPTIMISE) $(NVFLAGS) driverp.o -o driverp

driverp.o: driverp.cu $(DEPEND)
	$(NVCC) $(OPTIMISE) $(NVFLAGS) -c -o driverp.o driverp.cu

tests : tests.o
	$(NVCC) $(OPTIMISE) $(NVFLAGS) tests.o -o tests

tests.o: tests.cu $(DEPEND)
	$(NVCC) $(OPTIMISE) $(NVFLAGS) -c -o tests.o tests.cu

%.o : %.cu 
	$(NVCC) $(OPTIMISE) $(NVFLAGS) -c $< -o $@ 

clean: 
	rm driver driver.o driverp driverp.o debug debug.o tests.o tests
