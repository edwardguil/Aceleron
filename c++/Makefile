CC := g++
CFLAGS := -Wall -pedantic -g
OBJFILES := driver.o layers.o losses.o metrics.o
TARGET := driver

all: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES)

driver.o : driver.cpp
	$(CC) $(CFLAGS) -c driver.cpp -o driver.o

layers.o : layers.cpp
	$(CC) $(CFLAGS) -c layers.cpp -o layers.o

losses.o : losses.cpp
	$(CC) $(CFLAGS) -c losses.cpp -o losses.o

metrics.o : metrics.cpp
	$(CC) $(CFLAGS) -c metrics.cpp -o metrics.o
clean: 
	rm $(TARGET) $(OBJFILES)
