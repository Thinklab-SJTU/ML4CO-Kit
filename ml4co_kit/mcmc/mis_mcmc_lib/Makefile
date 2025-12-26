# Makefile for MIS MCMC C++ extension

CXX = g++
PYTHON = python3
PYBIND11_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_SUFFIX = $(shell $(PYTHON)-config --extension-suffix)

# Compiler flags
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND11_INCLUDES)

# Target shared library
TARGET = mis_mcmc_lib$(PYTHON_SUFFIX)

# Source files
SOURCES = mis.cpp bindings.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = mis.h

# Default target
all: $(TARGET)

# Build shared library
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

# Build object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean

