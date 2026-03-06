CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
INCLUDES = -I include

.PHONY: all test example_matmul example_manual clean

all: test example_matmul example_manual

test: tests/test_namiloop.cpp include/namiloop/namiloop.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o /tmp/namiloop_test tests/test_namiloop.cpp
	/tmp/namiloop_test

example_matmul: examples/matmul.cpp include/namiloop/namiloop.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o /tmp/namiloop_matmul examples/matmul.cpp
	/tmp/namiloop_matmul

example_manual: examples/manual.cpp include/namiloop/namiloop.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o /tmp/namiloop_manual examples/manual.cpp
	/tmp/namiloop_manual

clean:
	rm -f /tmp/namiloop_test /tmp/namiloop_matmul /tmp/namiloop_manual
	rm -f /tmp/namiloop_bench_*
	rm -f best_kernel.inc manual_kernel.inc
