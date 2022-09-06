CC := g++
CFLAGS := -Wall -pedantic -g -std=c++11
OBJFILES := driver.o layers.o losses.o metrics.o optimizers.o
OBJFILES2 := driver2.o layers.o losses.o metrics.o optimizers.o
TARGET := driver

all: driver2

driver2 : $(OBJFILES2) matrix.tpp
	$(CC) $(CFLAGS) -o driver2 $(OBJFILES2)

$(TARGET) : $(OBJFILES) matrix.tpp
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES)

driver2.o : driver2.cpp
	$(CC) $(CFLAGS) -c driver2.cpp -o driver2.o

driver.o : driver.cpp
	$(CC) $(CFLAGS) -c driver.cpp -o driver.o

layers.o : layers.cpp
	$(CC) $(CFLAGS) -c layers.cpp -o layers.o

losses.o : losses.cpp
	$(CC) $(CFLAGS) -c losses.cpp -o losses.o

metrics.o : metrics.cpp
	$(CC) $(CFLAGS) -c metrics.cpp -o metrics.o

optimizers.o : optimizers.cpp
	$(CC) $(CFLAGS) -c optimizers.cpp -o optimizers.o
clean: 
	rm $(TARGET) driver2.o driver2 $(OBJFILES)
