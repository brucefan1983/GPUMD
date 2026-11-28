###########################################################
# Note: 
# 1) You can modify -arch=sm_60 according to 
#    your GPU architecture.
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
# nvcc supports -MMD only on Linux.
DEPFLAGS = -MD
else # For linux
CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -Xcompiler="$(WARN_FLAGS)"
DEPFLAGS = -MMD
endif
DEPFLAGS += -MP -MF $(basename $@).d -MT $@
ifeq ($(GNEP_TEST_DIAGNOSTICS),1)
CFLAGS += -DGNEP_TEST_DIAGNOSTICS
endif
ifeq ($(GNEP_TEST_FIXED_SCALER),1)
CFLAGS += -DUSE_FIXED_SCALER
endif
INC = -I./
LDFLAGS = 
LIBS = -lcublas -lcusolver -lcufft
NESK_LIBS = -lcublas -lcusolver
NEP_LIBS =
GNEP_LIBS =
ifndef OS
LIBS += -lpthread
NEP_LIBS += -ldl
GNEP_LIBS += -lpthread
endif


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
SOURCES_GNEP =                     \
	$(wildcard main_gnep/*.cu)     \
	$(wildcard utilities/*.cu)   \
	$(wildcard utilities/*/*.cu)

SOURCES_NESK =                    \
	$(wildcard main_nesk/*.cu)    \
	utilities/error.cu           \
	utilities/main_common.cu


###########################################################
# object files
###########################################################
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_NEP = $(SOURCES_NEP:.cu=.obj)
OBJ_NESK = $(SOURCES_NESK:.cu=.obj)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
OBJ_NESK = $(SOURCES_NESK:.cu=.o)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.o)
endif


###########################################################
# compiler-generated header dependencies
###########################################################
OBJECTS = $(sort $(OBJ_GPUMD) $(OBJ_NEP) $(OBJ_NESK) $(OBJ_GNEP))
DEPFILES = $(addsuffix .d,$(basename $(OBJECTS)))


###########################################################
# executables
###########################################################
all: gpumd nep nesk

gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gpumd executable is successfully compiled!
	@echo =================================================
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

nesk: $(OBJ_NESK)
	$(CC) $(LDFLAGS) $^ -o $@ $(NESK_LIBS)
	@echo =================================================
	@echo The nesk executable is successfully compiled!
	@echo =================================================


###########################################################
# rules for building object files
###########################################################
ifdef OS # for Windows
$(OBJECTS): %.obj: %.cu
	$(CC) $(CFLAGS) $(INC) $(DEPFLAGS) -c $< -o $@
else # for Linux
$(OBJECTS): %.o: %.cu
	$(CC) $(CFLAGS) $(INC) $(DEPFLAGS) -c $< -o $@
endif

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
ifdef OS
	del /s *.obj *.d *.exp *.lib *.exe
else
	rm -f */*.o */*/*.o */*.d */*/*.d gpumd nep gnep nesk
endif
