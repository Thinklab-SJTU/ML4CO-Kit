CC = g++
CFLAGS = -O3
LDFLAGS = -lm
SOURCES = main.cpp
EXECUTABLE = spctsp_ils

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)

.PHONY: all clean