CC := g++
NVCC := nvcc
CFLAGS := -Wall -pedantic -g -std=c++11
NVFLAGS = --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -pg -g
OBJFILES := layers.o losses.o metrics.o optimizers.o cuda.o
DEPEND := driver.cu layers.cu losses.cu metrics.cu optimizers.cu matrix.cu cuda.cu 
OBJFILESDRIVER := driver.o $(OBJFILES)
OBJFILESDEBUG :=  debug.o $(OBJFILES)

all: driver

driver : driver.o
	$(NVCC) $(NVFLAGS) driver.o -o $@

driver.o: $(DEPEND)
	$(NVCC) $(NVFLAGS) -c -o driver.o driver.cu


# driver : $(DEPEND) \
	$(NVCC) $(NVFLAGS) $(DEPEND) -o $@ \
\
debug : $(OBJFILESDEBUG) matrix.cu \
	$(CC) $(CFLAGS) $(OBJFILESDEBUG) -o $@ \
\
%.o : %.cpp \
	$(CC) $(CFLAGS) -c $< -o $@ \
\
%.o : %.cu \
	$(NVCC) $(NVFLAGS) -c $< -o $@ \

clean: 
	rm driver driver.o debug debug.o $(OBJFILES)
