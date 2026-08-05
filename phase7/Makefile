# Shooting Brake Phase 7 — B70 Provider C ABI shared library
#
# Build:
#   source /opt/intel/oneapi/setvars.sh --force
#   make -C phase7
#
# Produces: libsb_b70_provider.so

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

.PHONY: all clean

all: $(TARGET)

$(TARGET): b70_capi.cpp b70_capi.h $(PHASE1)/b70_provider.hpp
	$(CXX) $(CXXFLAGS) -shared -o $(ROOT)/$(TARGET) \
	    b70_capi.cpp $(PHASE1)/b70_provider.cpp \
	    $(LDFLAGS)

clean:
	rm -f $(ROOT)/$(TARGET)
