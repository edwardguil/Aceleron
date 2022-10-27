CC := g++
NVCC := nvcc
CFLAGS := -Wall -pedantic -g -std=c++11
NVFLAGS = --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -pg -g
DEPEND := layers.cu losses.cu metrics.cu optimizers.cu matrix.cu cuda.cu

all: driver driverp

driver : driver.o
	$(NVCC) $(NVFLAGS) driver.o -o $@

driver.o: driver.cu  $(DEPEND)
	$(NVCC) $(NVFLAGS) -c -o driver.o driver.cu

driverp : driverParrallel.o
	$(NVCC) $(NVFLAGS) driverParrallel.o -o driverp

driverParrallel.o: driverParrallel.cu $(DEPEND)
	$(NVCC) $(NVFLAGS) -c -o driverParrallel.o driverParrallel.cu

tests : tests.o
	$(NVCC) $(NVFLAGS) tests.o -o tests

tests.o: tests.cu $(DEPEND)
	$(NVCC) $(NVFLAGS) -c -o tests.o tests.cu

%.o : %.cu 
	$(NVCC) $(NVFLAGS) -c $< -o $@ 

clean: 
	rm driver driver.o driverp driverParrallel.o debug debug.o tests.o tests
