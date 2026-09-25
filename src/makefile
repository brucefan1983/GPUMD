###########################################################
# Note: 
# 1) You can modify -arch=sm_60 according to 
#    your GPU architecture. For compute capability < 6.0,
#    need to add -DUSE_KEPLER to CFLAGS.
# 2) For Windows systems, if you get errors like 
#    c1xx : fatal error C1083: Cannot open source file: ...
#    You can consider removing -Xcompiler "/wd 4819"
# 3) Add -DUSE_PLUMED to CFLAGS when use the PLUMED plugin
#    and remove it otherwise.
###########################################################


###########################################################
# some flags
###########################################################
CC = nvcc
CUDA_ARCH=-arch=sm_60
WARN_FLAGS = -Wall -Wextra
WARN_FLAGS += -Wno-sign-compare -Wno-unused-parameter
WARN_FLAGS += -Wnull-dereference -Wstringop-overflow -Wstringop-truncation -Wrestrict
WARN_FLAGS += -Wduplicated-cond -Wduplicated-branches -Wlogical-op
WARN_FLAGS += -Wsequence-point -Wswitch-default
WARN_FLAGS += -Wcast-align -Wcast-qual -Wvla
WARN_FLAGS += -Wformat -Wformat-security -Wformat-overflow -Wformat-truncation
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 $(CUDA_ARCH)
else # For linux
CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -Xcompiler="$(WARN_FLAGS)"
endif
ifeq ($(GNEP_TEST_DIAGNOSTICS),1)
CFLAGS += -DGNEP_TEST_DIAGNOSTICS
endif
ifeq ($(GNEP_TEST_FIXED_SCALER),1)
CFLAGS += -DUSE_FIXED_SCALER
endif
INC = -I./
LDFLAGS = 
LIBS = -lcublas -lcusolver -lcufft
NEP_LIBS =
GNEP_LIBS =
ifndef OS
NEP_LIBS += -ldl
GNEP_LIBS += -lpthread
endif

# Wisevolve Potential Engine (WPE) host integration.
#
# Expected source layout for the WPE-enabled build:
#   <workspace>/gpumd/   (this repository / upstream PR)
#   <workspace>/wpe/     (Wisevolve public API + GPUMD integration source)
#   <workspace>/wpe.so/  (prebuilt WPE shared-library package)
#
# Standard GPUMD remains the default.  The dedicated WPE build is enabled with
# WPE=1 (or the gpumd-wpe convenience target).  The public integration and
# binary package may be relocated independently with WPE_ROOT/WPE_SO_ROOT.
WPE ?= 0
WPE_ROOT ?= ../../wpe
WPE_SO_ROOT ?= ../../wpe.so
ifneq ($(filter $(WPE),0 1),$(WPE))
$(error WPE must be 0 or 1)
endif

WPE_INCLUDE = $(WPE_ROOT)/include
WPE_INTEGRATION = $(WPE_ROOT)/integration/gpumd
WPE_LIBDIR = $(WPE_SO_ROOT)/lib
WPELIB = $(WPE_LIBDIR)/libwisevolve_potential.so
WPE_GPUMD_API = lib/wpe/wisevolve_potential_api.h
WPE_EXTERNAL_API = $(WPE_INCLUDE)/wisevolve_potential_api.h
# Use the same WPE library location for linking and runtime lookup by default.
# Relative paths remain relocatable with respect to the gpumd executable,
# while absolute paths are embedded directly.
ifeq ($(filter /%,$(WPE_LIBDIR)),)
WPE_RPATH_DEFAULT = $$ORIGIN/$(WPE_LIBDIR)
else
WPE_RPATH_DEFAULT = $(WPE_LIBDIR)
endif

WPE_RPATH ?= $(WPE_RPATH_DEFAULT)
WPE_NVCC_FLAGS ?= -Wno-deprecated-gpu-targets

ifeq ($(WPE),1)
ifdef OS
$(error WPE integration currently supports Linux/CUDA builds only)
endif
CFLAGS += -DGPUMD_WPE_ENABLED=1 $(WPE_NVCC_FLAGS)
LDFLAGS += $(WPE_NVCC_FLAGS)
INC += -Iforce -Ilib/wpe -I$(WPE_INTEGRATION)
GPUMD_BUILD_MODE = wpe
else
GPUMD_BUILD_MODE = stock
endif
GPUMD_MODE_STAMP = .gpumd_mode_$(GPUMD_BUILD_MODE)


###########################################################
# source files
###########################################################
SOURCES_GPUMD =                   \
	$(wildcard main_gpumd/*.cu)   \
	$(wildcard minimize/*.cu)     \
	$(wildcard phonon/*.cu)       \
	$(wildcard integrate/*.cu)    \
	$(wildcard mc/*.cu)           \
	$(wildcard force/*.cu)        \
	$(wildcard measure/*.cu)      \
	$(wildcard model/*.cu)        \
	$(wildcard utilities/*.cu)
SOURCES_NEP =                     \
	$(wildcard main_nep/*.cu)     \
	$(wildcard utilities/*.cu)
SOURCES_GNEP =                     \
	$(wildcard main_gnep/*.cu)     \
	$(wildcard utilities/*.cu)

# WPE integration sources are built directly from the sibling WPE repository.


###########################################################
# object files
###########################################################
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_NEP = $(SOURCES_NEP:.cu=.obj)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.o)
endif

ifeq ($(WPE),1)
WPE_HOST_OBJECTS = wpe_gpumd_adapter.o wpe_gpumd_stage.o
OBJ_GPUMD += $(WPE_HOST_OBJECTS)
endif


###########################################################
# headers
###########################################################
HEADERS =                         \
	$(wildcard utilities/*.cuh)   \
	$(wildcard main_gpumd/*.cuh)  \
	$(wildcard integrate/*.cuh)   \
	$(wildcard mc/*.cuh)          \
	$(wildcard minimize/*.cuh)    \
	$(wildcard force/*.cuh)       \
	$(wildcard measure/*.cuh)     \
	$(wildcard model/*.cuh)       \
	$(wildcard phonon/*.cuh)      \
	$(wildcard main_nep/*.cuh)    \
	$(wildcard main_gnep/*.cuh)
ifeq ($(WPE),1)
HEADERS += $(wildcard $(WPE_INTEGRATION)/*.cuh) $(WPE_GPUMD_API) $(WPE_EXTERNAL_API)
endif


###########################################################
# GPUMD build mode stamp
###########################################################
# GNU Make does not track changes in CFLAGS.  Stock and WPE builds therefore
# share a small mode stamp so switching `make gpumd` <-> `make WPE=1 gpumd`
# always recompiles GPUMD objects with the correct preprocessor surface.
$(GPUMD_MODE_STAMP):
	rm -f .gpumd_mode_stock .gpumd_mode_wpe
	touch $@


###########################################################
# executables
###########################################################
all: gpumd nep gpumd_replica

.PHONY: all gpumd_replica clean_replica clean gpumd-wpe wpe-api-check

# Keep replica objects separate and share the selected compiler and architecture.
gpumd_replica:
	$(MAKE) -C main_replica CC="$(CC)" CUDA_ARCH="$(CUDA_ARCH)"

gpumd-wpe:
	$(MAKE) -f makefile gpumd WPE=1 WPE_ROOT="$(WPE_ROOT)" WPE_SO_ROOT="$(WPE_SO_ROOT)" CUDA_ARCH="$(CUDA_ARCH)" WPE_NVCC_FLAGS="$(WPE_NVCC_FLAGS)"

wpe-api-check:
	@test -f "$(WPE_GPUMD_API)" || { echo "ERROR: missing GPUMD WPE API header: $(WPE_GPUMD_API)"; exit 2; }
	@test -f "$(WPE_EXTERNAL_API)" || { echo "ERROR: missing WPE API header: $(WPE_EXTERNAL_API)"; exit 2; }
	@cmp -s "$(WPE_GPUMD_API)" "$(WPE_EXTERNAL_API)" || { \
		echo "ERROR: GPUMD and WPE copies of wisevolve_potential_api.h differ"; exit 2; }

ifeq ($(WPE),1)
gpumd: wpe-api-check $(OBJ_GPUMD) $(WPELIB)
	$(CC) $(LDFLAGS) $(OBJ_GPUMD) -L"$(WPE_LIBDIR)" -Xlinker --no-as-needed -lwisevolve_potential \
		-Xlinker -rpath -Xlinker '$(WPE_RPATH)' -o $@ $(LIBS)
else

gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
endif
	@echo =================================================
	@echo The gpumd executable is successfully compiled!
	@echo =================================================

ifeq ($(WPE),1)
$(WPELIB):
	@echo "ERROR: missing $(WPELIB)"
	@echo "Build/install/download the WPE binary release under WPE_SO_ROOT=$(WPE_SO_ROOT)."
	@exit 2
endif

nep: $(OBJ_NEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS) $(NEP_LIBS)
	@echo =================================================
	@echo The nep executable is successfully compiled!
	@echo =================================================
gnep: $(OBJ_GNEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS) $(GNEP_LIBS)
	@echo =================================================
	@echo The gnep executable is successfully compiled!
	@echo =================================================


###########################################################
# rules for building object files
###########################################################
ifdef OS # for Windows
integrate/%.obj: integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
mc/%.obj: mc/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.obj: minimize/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.obj: force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
measure/%.obj: measure/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gpumd/%.obj: main_gpumd/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.obj: utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.obj: model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
phonon/%.obj: phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.obj: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.obj: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else # for Linux
integrate/%.o: integrate/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
mc/%.o: mc/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.o: minimize/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.o: force/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
ifeq ($(WPE),1)
wpe_gpumd_adapter.o: $(WPE_INTEGRATION)/wpe_adapter.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
wpe_gpumd_stage.o: $(WPE_INTEGRATION)/wpe_stage.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif
measure/%.o: measure/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gpumd/%.o: main_gpumd/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.o: utilities/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.o: model/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
phonon/%.o: phonon/%.cu $(HEADERS) $(GPUMD_MODE_STAMP)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.o: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif


###########################################################
# clean up
###########################################################
clean_replica:
	$(MAKE) -C main_replica clean

clean: clean_replica
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm -f */*.o wpe_gpumd_adapter.o wpe_gpumd_stage.o gpumd nep gnep .gpumd_mode_stock .gpumd_mode_wpe
endif
