###########################################################
# Note:
# 1) Modify -arch=sm_60 according to your GPU architecture
# 2) Add -DUSE_PLUMED to CFLAGS when using PLUMED
###########################################################

###########################################################
# compilers
###########################################################
NVCC    = nvcc
MPI_CC  = mpicxx

###########################################################
# CUDA / MPI paths
###########################################################
CUDA_ARCH = -arch=sm_89

MPI_HOME = /opt/apps/mpi/openmpi-5.0.7-cuda12.4-classic
MPI_INC  = -I$(MPI_HOME)/include
MPI_LIB  = -L$(MPI_HOME)/lib -lmpi

###########################################################
# flags
###########################################################
CFLAGS  = -std=c++14 -O3 $(CUDA_ARCH)
INC     = -I./ $(MPI_INC)
LDFLAGS =
LIBS    = -lcublas -lcusolver -lcufft

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

SOURCES_GNEP =                    \
	$(wildcard main_gnep/*.cu)    \
	$(wildcard utilities/*.cu)

###########################################################
# object files
###########################################################
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP   = $(SOURCES_NEP:.cu=.o)
OBJ_GNEP  = $(SOURCES_GNEP:.cu=.o)

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
all: gpumd nep nep_mpi gnep

gpumd: $(OBJ_GPUMD)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo "================================================="
	@echo "The gpumd executable is successfully compiled!"
	@echo "================================================="

nep: $(OBJ_NEP)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo "================================================="
	@echo "The nep executable is successfully compiled!"
	@echo "================================================="

nep_mpi: $(OBJ_NEP)
	nvcc -ccbin mpicxx $(LDFLAGS) $^ -o $@ \
	$(LIBS) $(MPI_LIB)
	@echo "================================================="
	@echo "The nep_mpi executable is successfully compiled!"
	@echo "================================================="

gnep: $(OBJ_GNEP)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo "================================================="
	@echo "The gnep executable is successfully compiled!"
	@echo "================================================="

###########################################################
# build rules
###########################################################
integrate/%.o: integrate/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

mc/%.o: mc/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

minimize/%.o: minimize/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

force/%.o: force/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

measure/%.o: measure/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

main_gpumd/%.o: main_gpumd/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

utilities/%.o: utilities/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

model/%.o: model/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

phonon/%.o: phonon/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

main_gnep/%.o: main_gnep/%.cu $(HEADERS)
	$(NVCC) $(CFLAGS) $(INC) -c $< -o $@

###########################################################
# clean
###########################################################
clean:
	rm -f */*.o gpumd nep nep_mpi gnep
