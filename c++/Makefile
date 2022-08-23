CC := g++
CFLAGS := -Wall -pedantic -g
OBJFILES := layers.o losses.o
TARGET := layers

all: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJFILES)

clean: 
	rm $(TARGET) $(OBJFILES)
