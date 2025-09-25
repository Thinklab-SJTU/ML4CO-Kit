CC = g++
CFLAGS = -O3
LDFLAGS = -lm
SOURCES = main.cpp env.cpp cross.cpp evaluator.cpp indi.cpp rand.cpp kopt.cpp sort.cpp
EXECUTABLE = ga_eax_large_solver

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)

.PHONY: all clean