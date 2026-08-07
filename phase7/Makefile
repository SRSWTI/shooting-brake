# Shooting Brake Phase 7 — native provider shared libraries
#
# Build both:
#   source /opt/intel/oneapi/setvars.sh --force
#   make -C phase7
#
# The CPU tier has no SYCL dependency and builds on its own with stock g++:
#   make -C phase7 cpu
#
# Produces: libsb_b70_provider.so, libsb_cpu_expert.so

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PHASE1 := $(ROOT)/../phase1
QUIXICORE := $(ROOT)/../QuixiCore-XPU

CXX := icpx
CXXFLAGS := -std=gnu++20 -O2 -g -fPIC -Wall -fsycl \
            -I$(ROOT) -I$(PHASE1) \
            -I$(QUIXICORE)/include \
            -I/opt/intel/oneapi/compiler/2026.1/include
LDFLAGS := -L$(QUIXICORE)/build-sycl \
           -lquixicore_xpu -lquixicore_xpu_ops \
           -fsycl \
           -L/opt/intel/oneapi/compiler/2026.1/lib \
           -L/opt/intel/oneapi/umf/1.1/lib

TARGET := libsb_b70_provider.so

# CPU expert tier: plain C++ / pthreads. Kept off icpx and out of the SYCL
# link line so it stays buildable (and testable) without oneAPI sourced.
# -march=native matters here: the projection kernels are hand-unrolled for
# the compiler to vectorise, and the whole tier is DRAM-bound.
CPU_CXX := g++
CPU_CXXFLAGS := -std=gnu++20 -O3 -march=native -g -fPIC -Wall -Wextra -pthread
CPU_TARGET := libsb_cpu_expert.so

.PHONY: all clean cpu b70

all: $(TARGET) $(CPU_TARGET)

b70: $(TARGET)

cpu: $(CPU_TARGET)

$(TARGET): b70_capi.cpp b70_capi.h \
           $(PHASE1)/b70_provider.cpp $(PHASE1)/b70_provider.hpp
	$(CXX) $(CXXFLAGS) -shared -o $(ROOT)/$(TARGET) \
	    b70_capi.cpp $(PHASE1)/b70_provider.cpp \
	    $(LDFLAGS)

$(CPU_TARGET): cpu_expert_capi.cpp cpu_expert_capi.h
	$(CPU_CXX) $(CPU_CXXFLAGS) -shared -o $(ROOT)/$(CPU_TARGET) \
	    cpu_expert_capi.cpp

clean:
	rm -f $(ROOT)/$(TARGET) $(ROOT)/$(CPU_TARGET)
