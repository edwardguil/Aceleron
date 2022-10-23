CC := g++
CFLAGS := -Wall -pedantic -g -std=c++11
OBJFILES := driver.o layers.o losses.o metrics.o optimizers.o
TARGET := driver

all: driver

driver : $(OBJFILES) matrix.tpp
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES)

debug : $(OBJFILES) matrix.tpp
	$(CC) $(CFLAGS) -o debug debug.o $(OBJFILES)

driver.o : driver.cpp
	$(CC) $(CFLAGS) -c driver.cpp -o driver.o

debug.o : driver.cpp
	$(CC) $(CFLAGS) -c debug.cpp -o debug.o

layers.o : layers.cpp
	$(CC) $(CFLAGS) -c layers.cpp -o layers.o

losses.o : losses.cpp
	$(CC) $(CFLAGS) -c losses.cpp -o losses.o

metrics.o : metrics.cpp
	$(CC) $(CFLAGS) -c metrics.cpp -o metrics.o

optimizers.o : optimizers.cpp
	$(CC) $(CFLAGS) -c optimizers.cpp -o optimizers.o
clean: 
	rm $(TARGET) debug debug.o $(OBJFILES)
