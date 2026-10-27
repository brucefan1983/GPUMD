# Use with src/makefile; build probes from the same objects and CUDA flags as nep.
# make -C src -f makefile -f ../tests_pytest/spin_probes.mk spin-probes
SPIN_TEST_BIN ?= $(abspath ../build/spin-probes)
SPIN_TEST_NAMES := restart_parity response_fitness spin_mforce_split
SPIN_TEST_TARGETS := $(addprefix $(SPIN_TEST_BIN)/,$(SPIN_TEST_NAMES))
SPIN_TEST_OBJECTS := $(filter-out main_nep/main.o main_nep/main.obj,$(OBJ_NEP))

.PHONY: spin-probes
spin-probes: $(SPIN_TEST_TARGETS)

$(SPIN_TEST_TARGETS): $(SPIN_TEST_BIN)/%: ../tests_pytest/%.cu ../tests_pytest/spin_test_access.cuh $(SPIN_TEST_OBJECTS)
	mkdir -p $(SPIN_TEST_BIN)
	$(CC) $(CFLAGS) $(INC) -I../tests_pytest $< $(SPIN_TEST_OBJECTS) -o $@ $(LIBS) $(NEP_LIBS)
