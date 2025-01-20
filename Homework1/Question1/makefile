CXX = g++

CXXFLAGS = -Wall -g

all: $(Q1_floatbench)

$(Q1_floatbench): $(Q1_floatbench.cpp)
    $(g++) $(CXXFLAGS) -o $(Q1_floatbench) $(Q1_floatbench.cpp)

clean:
    rm -f $(Q1_floatbench)