# 指定目标共享库名称
TARGET := ALKH.so

# 指定编译器和编译选项
CXX := g++
CXXFLAGS := -std=c++11 -fPIC -Wall

# 指定源文件和对象文件
CPP_SRC := ALKH.cpp  # 在这里指定你的.cpp文件名
OBJ := $(CPP_SRC:.cpp=.o)

# 默认目标
all: $(TARGET)

# 生成共享库
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

# 生成目标文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清理生成的文件
clean:
	rm -f $(OBJ) $(TARGET)