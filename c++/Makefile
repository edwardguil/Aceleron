CC := g++
NVCC := nvcc
CFLAGS := -Wall -pedantic -g -std=c++11
NVFLAGS = --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -pg -g
DEPEND := layers.cu losses.cu metrics.cu optimizers.cu matrix.cu cuda.o
OBJFILES := cuda.o

all: driverParrallel

driver : driver.o
	$(NVCC) $(NVFLAGS) $(OBJFILES) driver.o -o $@

driver.o: driver.cu  $(DEPEND)
	$(NVCC) $(NVFLAGS) -c -o driver.o driver.cu

driverParrallel : driverParrallel.o
	$(NVCC) $(NVFLAGS) $(OBJFILES) driverParrallel.o -o driver

driverParrallel.o: driverParrallel.cu $(DEPEND)
	$(NVCC) $(NVFLAGS) -c -o driverParrallel.o driverParrallel.cu

%.o : %.cu 
	$(NVCC) $(NVFLAGS) -c $< -o $@ 

clean: 
	rm driver driver.o driverParrallel driverParrallel.o  debug debug.o 
