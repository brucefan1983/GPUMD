###########################################################
# Note:
# 1) Set HIP_ARCH to your GPU architecture, e.g.
#    make -f makefile.hip HIP_ARCH=gfx1100
#    (defaults to gfx1100).
# 2) Do not remove -DUSE_HIP.
# 3) Add -DUSE_PLUMED to CFLAGS when use the PLUMED plugin
#    and remove it otherwise.
###########################################################


###########################################################
# some flags
###########################################################
CC = hipcc
HIP_ARCH ?= gfx1100
CFLAGS = -O3 -std=c++17 --offload-arch=$(HIP_ARCH) -DUSE_HIP
CFLAGS += -Wno-inconsistent-missing-override -Wno-unused-result -Wno-pass-failed
DEPFLAGS = -MMD -MP -MF $(basename $@).d -MT $@
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
# compiler-generated header dependencies
###########################################################
OBJECTS = $(sort $(OBJ_GPUMD) $(OBJ_NEP) $(OBJ_TB))
DEPFILES = $(addsuffix .d,$(basename $(OBJECTS)))


###########################################################
# executables
###########################################################
all: gpumd nep tb
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
$(OBJECTS): %.o: %.cu
	$(CC) $(CFLAGS) $(INC) $(DEPFLAGS) -c $< -o $@

# Rebuild objects whose dependency files are missing, including older builds.
$(foreach obj,$(OBJECTS),$(if $(wildcard $(basename $(obj)).d),,$(obj))): FORCE

.PHONY: all clean FORCE
FORCE:

# Keep dependency targets after all so they cannot change the default goal.
-include $(wildcard $(DEPFILES))


###########################################################
# clean up
###########################################################
clean:
	rm -f */*.o */*/*.o */*.d */*/*.d gpumd nep tb

