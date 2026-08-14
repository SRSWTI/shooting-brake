# Shooting Brake Phase 1 — B70 Provider Build
#
# Usage:
#   source /opt/intel/oneapi/compiler/2026.1/env/vars.sh
#   make -C src/phase1
INCLUDES = -I$(QUIXICORE)/include -I/opt/intel/oneapi/compiler/2026.1/include
CXX = icpx
QUIXICORE = ../QuixiCore-XPU
CXXFLAGS = -std=gnu++20 -O2 -g -fPIC -Wall -fsycl $(INCLUDES)
LDFLAGS = -L$(QUIXICORE)/build-sycl -lquixicore_xpu -lquixicore_xpu_ops

TARGETS = b70_provider_test b70_provider_tests b70_provider

all: $(TARGETS)

b70_provider_test: b70_provider_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

b70_provider.o: b70_provider.cpp b70_provider.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ b70_provider.cpp

b70_provider_tests: b70_provider_tests.cpp b70_provider.o
	$(CXX) $(CXXFLAGS) -o $@ b70_provider_tests.cpp b70_provider.o $(LDFLAGS)

b70_provider: b70_provider_main.cpp b70_provider.o
	$(CXX) $(CXXFLAGS) -o $@ b70_provider_main.cpp b70_provider.o $(LDFLAGS)

clean:
	rm -f $(TARGETS) b70_provider.o

.PHONY: all clean
