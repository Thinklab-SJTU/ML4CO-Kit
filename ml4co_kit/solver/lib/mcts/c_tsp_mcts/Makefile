# Specify the target shared library name
TARGET := tsp_mcts_decoder.so

# Specify the compiler and compilation options
CXX := g++
CXXFLAGS := -std=c++11 -fPIC -Wall

# Specify the source files and object files
CPP_SRC := tsp_mcts_decoder.cpp  # Specify your .cpp filename here
OBJ := $(CPP_SRC:.cpp=.o)

# Default target
all: $(TARGET)

# Generate the shared library
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

# Generate the object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean the generated files
clean:
	rm -f $(OBJ) $(TARGET)