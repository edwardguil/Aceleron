CC := g++
CFLAGS := -Wall -pedantic -g -std=c++11
OBJFILES := layers.o losses.o metrics.o optimizers.o
OBJFILESDRIVER := driver.o $(OBJFILES)
OBJFILESDEBUG :=  debug.o $(OBJFILES)

all: driver

driver : $(OBJFILESDRIVER) matrix.tpp
	$(CC) $(CFLAGS) -o $@ $(OBJFILESDRIVER)

debug : $(OBJFILESDEBUG) matrix.tpp
	$(CC) $(CFLAGS) -o $@ $(OBJFILESDEBUG)

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
