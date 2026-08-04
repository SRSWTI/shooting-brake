# Shooting Brake Phase 3 — independent provider mathematics gate

ICPX ?= icpx
PYTHON ?= python3
QUIXICORE ?= ../QuixiCore-XPU
BUILD := .build

SYCL_FLAGS := -std=c++20 -O2 -g -fsycl -Wall -Wextra -Wpedantic -pthread
INCLUDES := -I. -I../phase1 -I../phase2 -I$(QUIXICORE)/include \
	-I/opt/intel/oneapi/compiler/2026.1/include
B70_LDFLAGS := -L$(QUIXICORE)/build-sycl -lquixicore_xpu \
	-lquixicore_xpu_ops -pthread

TEST_OBJECTS := $(BUILD)/provider_math_test.o $(BUILD)/reference_fixture.o \
	$(BUILD)/math_cases.o $(BUILD)/shared_ring.o $(BUILD)/b70_provider.o
SERVER_OBJECTS := $(BUILD)/b70_ring_provider.o $(BUILD)/shared_ring.o \
	$(BUILD)/b70_provider.o
FAULT_SERVER_OBJECTS := $(BUILD)/b70_ring_provider_fault.o \
	$(BUILD)/shared_ring.o $(BUILD)/b70_provider_fault.o
TARGETS := provider_math_test b70_ring_provider b70_ring_provider_fault

all: $(TARGETS)

reference-fixture: reference_fixture.bin

reference_fixture.bin: generate_reference.py
	$(PYTHON) generate_reference.py

provider_math_test: $(TEST_OBJECTS)
	$(ICPX) $(SYCL_FLAGS) -o $@ $(TEST_OBJECTS) $(B70_LDFLAGS)

b70_ring_provider: $(SERVER_OBJECTS)
	$(ICPX) $(SYCL_FLAGS) -o $@ $(SERVER_OBJECTS) $(B70_LDFLAGS)

b70_ring_provider_fault: $(FAULT_SERVER_OBJECTS)
	$(ICPX) $(SYCL_FLAGS) -o $@ $(FAULT_SERVER_OBJECTS) $(B70_LDFLAGS)

$(BUILD):
	mkdir -p $@

$(BUILD)/provider_math_test.o: provider_math_test.cpp math_cases.hpp \
		reference_fixture.hpp ../phase1/b70_provider.hpp \
		../phase2/b70_ring_control.hpp ../phase2/shared_ring.hpp \
		../phase2/ring_protocol.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/reference_fixture.o: reference_fixture.cpp reference_fixture.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/math_cases.o: math_cases.cpp math_cases.hpp reference_fixture.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/shared_ring.o: ../phase2/shared_ring.cpp ../phase2/shared_ring.hpp \
		../phase2/ring_protocol.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/b70_provider.o: ../phase1/b70_provider.cpp ../phase1/b70_provider.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/b70_ring_provider.o: ../phase2/b70_ring_provider.cpp \
		../phase2/b70_ring_control.hpp ../phase2/shared_ring.hpp \
		../phase2/ring_protocol.hpp ../phase1/b70_provider.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/b70_provider_fault.o: ../phase1/b70_provider.cpp \
		../phase1/b70_provider.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -DSHOOTING_BRAKE_ENABLE_TEST_FAULTS \
		-c $< -o $@

$(BUILD)/b70_ring_provider_fault.o: ../phase2/b70_ring_provider.cpp \
		../phase2/b70_ring_control.hpp ../phase2/shared_ring.hpp \
		../phase2/ring_protocol.hpp ../phase1/b70_provider.hpp | $(BUILD)
	$(ICPX) $(SYCL_FLAGS) $(INCLUDES) -DSHOOTING_BRAKE_ENABLE_TEST_FAULTS \
		-c $< -o $@

phase3-test: reference-fixture $(TARGETS)
	./provider_math_test

clean:
	rm -rf $(BUILD) $(TARGETS)

.PHONY: all reference-fixture phase3-test clean
