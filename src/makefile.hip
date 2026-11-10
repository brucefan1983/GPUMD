###########################################################
# Note:
# 1) Set HIP_ARCH to your GPU architecture, e.g.
#    make -f makefile.hip HIP_ARCH=gfx1100
#    (defaults to gfx90a).
# 2) Do not remove -DUSE_HIP or -ffp-contract=on. A CFLAGS=...
#    override on the command line replaces the whole line (and
#    is passed to gpumd_replica), so keep both flags in it.
# 3) Add -DUSE_PLUMED to CFLAGS when use the PLUMED plugin
#    and remove it otherwise.
###########################################################


###########################################################
# some flags
###########################################################
CC = hipcc
HIP_ARCH ?= gfx90a
# Leave the C++ standard at the compiler default (C++17): rocThrust/rocPRIM
# (pulled in by thrust::exclusive_scan in neighbor.cu etc.) require C++17 and
# hipcc defaults to it, so do not pin -std lower.
# -ffp-contract=on fuses multiply-adds only within one source expression, so
# every instantiation of the same code forms the same FMAs. hipcc's default
# (fast) also fuses across statements after inlining, which makes the per-slot
# and batched replica NEP kernels round differently.
CFLAGS = -O3 --offload-arch=$(HIP_ARCH) -DUSE_HIP -ffp-contract=on
INC = -I./
LDFLAGS = 
LIBS = -lhipblas -lhipsolver -lhipfft
TB_LIBS = -lhipblas -lhipsolver


###########################################################
# source files
###########################################################
SOURCES_GPUMD =                   \
	$(wildcard main_gpumd/*.cu)   \
	$(wildcard minimize/*.cu)     \
	$(wildcard phonon/*.cu)       \
	$(wildcard integrate/*.cu)    \
	$(wildcard integrate/*/*.cu) \
	$(wildcard mc/*.cu)           \
	$(wildcard force/*.cu)        \
	$(wildcard force/*/*.cu)      \
	$(wildcard measure/*.cu)      \
	$(wildcard measure/*/*.cu)      \
	$(wildcard model/*.cu)        \
	$(wildcard utilities/*.cu)   \
	$(wildcard utilities/*/*.cu)
SOURCES_NEP =                     \
	$(wildcard main_nep/*.cu)     \
	$(wildcard main_nep/*/*.cu)   \
	$(wildcard utilities/*.cu)   \
	$(wildcard utilities/*/*.cu)

SOURCES_TB =                      \
	$(wildcard main_tb/*.cu)      \
	utilities/error.cu           \
	utilities/main_common.cu


###########################################################
# object files
###########################################################
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
OBJ_TB = $(SOURCES_TB:.cu=.o)


###########################################################
# headers
###########################################################
HEADERS =                         \
	$(wildcard utilities/*.cuh)   \
	$(wildcard utilities/*/*.cuh) \
	$(wildcard main_gpumd/*.cuh)  \
	$(wildcard integrate/*.cuh)   \
	$(wildcard integrate/*/*.cuh) \
	$(wildcard mc/*.cuh)          \
	$(wildcard minimize/*.cuh)    \
	$(wildcard force/*.cuh)       \
	$(wildcard force/*/*.cuh)     \
	$(wildcard measure/*.cuh)     \
	$(wildcard measure/*/*.cuh)     \
	$(wildcard model/*.cuh)       \
	$(wildcard phonon/*.cuh)      \
	$(wildcard main_nep/*.cuh)    \
	$(wildcard main_nep/*/*.cuh)

HEADERS_TB = $(wildcard main_tb/*.cuh) $(wildcard utilities/*.cuh) $(wildcard utilities/*/*.cuh)


###########################################################
# executables
###########################################################
all: gpumd nep gpumd_replica tb

.PHONY: all gpumd_replica clean_replica clean

# Keep replica objects separate and share the selected compiler and flags.
gpumd_replica:
	$(MAKE) -C main_replica CC="$(CC)" CFLAGS="$(CFLAGS)" LIBS="-lhipfft"

gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gpumd executable is successfully compiled!
	@echo =================================================
nep: $(OBJ_NEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The nep executable is successfully compiled!
	@echo =================================================

tb: $(OBJ_TB)
	$(CC) $(LDFLAGS) $^ -o $@ $(TB_LIBS)
	@echo =================================================
	@echo The tb executable is successfully compiled!
	@echo =================================================


###########################################################
# rules for building object files
###########################################################
integrate/%.o: integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
mc/%.o: mc/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.o: minimize/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.o: force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
measure/%.o: measure/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gpumd/%.o: main_gpumd/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.o: utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.o: model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
phonon/%.o: phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_tb/%.o: main_tb/%.cu $(HEADERS_TB)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@


###########################################################
# clean up
###########################################################
clean_replica:
	$(MAKE) -C main_replica clean

clean: clean_replica
	rm -f */*.o */*/*.o gpumd nep tb

