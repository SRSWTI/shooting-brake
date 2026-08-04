# Shooting Brake Phase 2 — shared-ring integration and transport probes
#
# Run from the repository root after sourcing the full oneAPI environment.

NVCC ?= nvcc
ICPX ?= icpx
CXX ?= g++
QUIXICORE ?= ../QuixiCore-XPU

CUDA_FLAGS := -std=c++20 -O2 -lineinfo -arch=sm_120 -Xcompiler=-Wall,-Wextra,-pthread
SYCL_FLAGS := -std=c++20 -O2 -g -fsycl -Wall -Wextra -Wpedantic -pthread
HOST_FLAGS := -std=c++20 -O2 -g -Wall -Wextra -Wpedantic -pthread
B70_INCLUDES := -I../phase1 -I$(QUIXICORE)/include -I/opt/intel/oneapi/compiler/2026.1/include
B70_LDFLAGS := -L$(QUIXICORE)/build-sycl -lquixicore_xpu -lquixicore_xpu_ops -pthread

TRANSPORT_TARGETS := memfd_transport_host memfd_transport_provider
B70_RING_TARGETS := b70_ring_provider b70_ring_integration_test
RING_TARGETS := shared_ring_tests
TARGETS := $(TRANSPORT_TARGETS) $(RING_TARGETS) $(B70_RING_TARGETS)

all: transport-probe shared-ring-tests b70-ring-integration

transport-probe: $(TRANSPORT_TARGETS)

memfd_transport_host: memfd_transport_host.cu memfd_transport_protocol.hpp
	$(NVCC) $(CUDA_FLAGS) -o $@ memfd_transport_host.cu

memfd_transport_provider: memfd_transport_provider.cpp memfd_transport_protocol.hpp
	$(ICPX) $(SYCL_FLAGS) -o $@ memfd_transport_provider.cpp

shared-ring-tests: $(RING_TARGETS)

shared_ring_tests: shared_ring_tests.cpp shared_ring.cpp shared_ring.hpp ring_protocol.hpp
	$(CXX) $(HOST_FLAGS) -o $@ shared_ring_tests.cpp shared_ring.cpp

b70-ring-integration: $(B70_RING_TARGETS)

b70_ring_provider: b70_ring_provider.cpp shared_ring.cpp shared_ring.hpp ring_protocol.hpp \
		../phase1/b70_provider.cpp ../phase1/b70_provider.hpp
	$(ICPX) $(SYCL_FLAGS) $(B70_INCLUDES) -o $@ b70_ring_provider.cpp \
		shared_ring.cpp ../phase1/b70_provider.cpp $(B70_LDFLAGS)

b70_ring_integration_test: b70_ring_integration_test.cpp shared_ring.cpp \
		shared_ring.hpp ring_protocol.hpp
	$(ICPX) $(SYCL_FLAGS) $(B70_INCLUDES) -o $@ \
		b70_ring_integration_test.cpp shared_ring.cpp -pthread

clean:
	rm -f $(TARGETS)

.PHONY: all transport-probe shared-ring-tests b70-ring-integration clean
