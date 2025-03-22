# Specify the target shared library name
TARGET := cvrp_local_search.so

# Specify the compiler and compilation options
CXX := g++
CXXFLAGS := -std=c++11 -fPIC -Wall

# Specify the source files and object files
CPP_SRC := cvrp_local_search.cpp Params.cpp LocalSearch.cpp Individual.cpp Split.cpp # Add Params.cpp here
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