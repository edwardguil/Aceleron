CC := g++
CFLAGS := -Wall -pedantic -g
OBJFILES := driver.o layers.o losses.o
TARGET := driver

all: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES)

clean: 
	rm $(TARGET) $(OBJFILES)
