# Shooting Brake Phase 7 — native provider shared libraries
#
# Build both:
#   source /opt/intel/oneapi/setvars.sh --force
#   make -C src/phase7
#
# The CPU tier has no SYCL dependency and builds on its own with stock g++:
#   make -C src/phase7 cpu
#
# Produces: libsb_b70_provider.so, libsb_cpu_expert.so

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PHASE1 := $(ROOT)/../phase1
QUIXICORE := $(ROOT)/../QuixiCore-XPU
NVFP4 := $(ROOT)/xe2_nvfp4
VKERN := $(ROOT)/../../vendor/intel-xpu/vllm-xpu/vllm-xpu-kernels
CUTLASS_SYCL := $(VKERN)/.deps/cutlass-sycl-src

CXX := icpx
CXXFLAGS := -std=gnu++20 -O2 -g -fPIC -Wall -fsycl \
            -I$(ROOT) -I$(PHASE1) \
            -I$(QUIXICORE)/include \
            -I/opt/intel/oneapi/compiler/2026.1/include \
            -I$(NVFP4) \
            -I$(ROOT)/../../vendor/compute-runtime/level_zero/include
LDFLAGS := -L$(QUIXICORE)/build-sycl \
           -lquixicore_xpu -lquixicore_xpu_ops \
           -fsycl \
           -L/opt/intel/oneapi/compiler/2026.1/lib \
           -L/opt/intel/oneapi/umf/1.1/lib \
           -lze_loader

GROUPED_FLAGS := -DCUTLASS_ENABLE_HEADERS_ONLY -DCUTLASS_ENABLE_SYCL \
                 -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED \
                 -DVLLM_XPU_ENABLE_XE2 \
                 -I$(VKERN)/csrc/xpu/grouped_gemm \
                 -I$(CUTLASS_SYCL)/include \
                 -I$(CUTLASS_SYCL)/tools/util/include -I$(VKERN)
SPIRV_EXT := -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate

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

$(ROOT)/grouped_moe.o: $(NVFP4)/grouped_moe.cpp $(NVFP4)/grouped_moe.hpp
	$(CXX) $(CXXFLAGS) $(GROUPED_FLAGS) $(SPIRV_EXT) -c -o $@ $(NVFP4)/grouped_moe.cpp

$(TARGET): b70_capi.cpp b70_capi.h \
           $(PHASE1)/b70_provider.cpp $(PHASE1)/b70_provider.hpp \
           $(ROOT)/grouped_moe.o
	$(CXX) $(CXXFLAGS) $(SPIRV_EXT) -shared -o $(ROOT)/$(TARGET) \
	    b70_capi.cpp $(PHASE1)/b70_provider.cpp $(ROOT)/grouped_moe.o \
	    $(LDFLAGS)

$(CPU_TARGET): cpu_expert_capi.cpp cpu_expert_capi.h
	$(CPU_CXX) $(CPU_CXXFLAGS) -shared -o $(ROOT)/$(CPU_TARGET) \
	    cpu_expert_capi.cpp

clean:
	rm -f $(ROOT)/$(TARGET) $(ROOT)/$(CPU_TARGET) $(ROOT)/grouped_moe.o
