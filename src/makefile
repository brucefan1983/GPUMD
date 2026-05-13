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
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 $(CUDA_ARCH)
else # For linux
CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -Xcompiler -fPIC
endif
INC = -I./
LDFLAGS = 
LIBS = -lcublas -lcusolver -lcufft

# ---- Python / pybind11 configuration (for pygpumd target) ----
PYBIND11_INC  := $(shell python3 -m pybind11 --includes 2>/dev/null)
PYTHON_CFLAGS := $(shell python3-config --cflags 2>/dev/null)
PYTHON_LDFLAGS:= $(shell python3-config --ldflags --embed 2>/dev/null)
# If python3-config fails, fall back to simpler flags
ifeq ($(PYTHON_CFLAGS),)
  PYTHON_CFLAGS = -I/usr/include/python3
  PYTHON_LDFLAGS = -lpython3
endif
# CUDA include path: needed because run.cuh transitively includes
# headers (e.g. add_random_force.cuh) that require curand_kernel.h.
# nvcc 11.x does not support --print-search-dirs, so we check common paths.
ifneq ($(wildcard /usr/local/cuda-11.6/include/curand_kernel.h),)
  CUDA_INC := /usr/local/cuda-11.6/include
else ifneq ($(wildcard /usr/local/cuda-11.7/include/curand_kernel.h),)
  CUDA_INC := /usr/local/cuda-11.7/include
else ifneq ($(wildcard /usr/local/cuda-11.8/include/curand_kernel.h),)
  CUDA_INC := /usr/local/cuda-11.8/include
else ifneq ($(wildcard /usr/local/cuda-12.2/include/curand_kernel.h),)
  CUDA_INC := /usr/local/cuda-12.2/include
else ifneq ($(wildcard /usr/local/cuda/include/curand_kernel.h),)
  CUDA_INC := /usr/local/cuda/include
else
  # last resort: user must override
  CUDA_INC := /usr/local/cuda/targets/x86_64-linux/include
endif
# CUDA library path: needed for linking cublas, cusolver, cufft, cudart
# nvcc 11.x does not support --print-search-dirs, so we rely on pkg-config
# or a set of common CUDA install paths.
CUDA_LIB := $(shell \
  pkg-config --libs cuda 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($$i ~ /^-L/) print substr($$i,3)}' | head -1 || \
  echo '/usr/local/cuda/targets/x86_64-linux/lib')
# As a fallback, check known CUDA 11.x locations
ifneq ($(wildcard $(CUDA_LIB)/libcusolver.so),)
  # path is valid
else ifneq ($(wildcard /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolver.so),)
  CUDA_LIB := /usr/local/cuda-11.6/targets/x86_64-linux/lib
else ifneq ($(wildcard /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcusolver.so),)
  CUDA_LIB := /usr/local/cuda-11.7/targets/x86_64-linux/lib
else ifneq ($(wildcard /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcusolver.so),)
  CUDA_LIB := /usr/local/cuda-12.2/targets/x86_64-linux/lib
else ifneq ($(wildcard /usr/local/cuda/lib64/libcusolver.so),)
  CUDA_LIB := /usr/local/cuda/lib64
else
  # last resort: user must override
endif
# --------------------------------------------------------------


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


###########################################################
# executables
###########################################################
all: gpumd nep 
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
gnep: $(OBJ_GNEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gnep executable is successfully compiled!
	@echo =================================================


# ---- pygpumd: Python extension module for PySAGES integration ----
# NOTE: run "make clean" before "make pygpumd" if you previously built
# the standard gpumd executable, because this target adds -DUSE_PYSAGES
# to CFLAGS and the object files must be recompiled.
PYGPUMD_SRC = main_gpumd/gpumd_python.cpp

ifdef OS
PYGPUMD_OBJ = $(PYGPUMD_SRC:.cpp=.obj)
else
PYGPUMD_OBJ = $(PYGPUMD_SRC:.cpp=.o)
endif

# Exclude main.cu (contains int main) from the shared library.
PY_OBJ_GPUMD = $(filter-out %/main.o %/main.obj,$(OBJ_GPUMD)) $(PYGPUMD_OBJ)

# Enable PYSAGES hooks in the GPUMD core when building the Python module.
pygpumd: CFLAGS += -DUSE_PYSAGES
pygpumd: $(PY_OBJ_GPUMD)
ifdef OS
	@echo Windows Python module build not yet implemented in makefile.
else
	# Link with CUDA runtime (-lcudart) and set RPATH so Python can find
	# the libraries at load time without needing LD_LIBRARY_PATH.
	g++ -shared -fPIC $(LDFLAGS) $^ -o gpumd.so \
	  -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) \
	  $(LIBS) -lcudart $(PYTHON_LDFLAGS)
	@echo =================================================
	@echo The gpumd Python module is successfully compiled!
	@echo =================================================
endif
# -----------------------------------------------------------------


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
main_gnep/%.o: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

# Compile the pybind11 wrapper with the host C++ compiler.
# Do NOT pass $(CFLAGS) here — it contains nvcc-only flags (-arch, -Xcompiler).
# We add -DUSE_PYSAGES explicitly so the wrapper sees the callback members.
main_gpumd/%.o: main_gpumd/%.cpp $(HEADERS)
	g++ -std=c++14 -O3 -fPIC -DUSE_PYSAGES $(PYBIND11_INC) $(PYTHON_CFLAGS) -I$(CUDA_INC) $(INC) -c $< -o $@
endif


###########################################################
# clean up
###########################################################
clean:
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm -f */*.o main_gpumd/*.o gpumd nep gnep gpumd.so
endif

