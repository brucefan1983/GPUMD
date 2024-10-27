###########################################################
# Note: 
# 1) You can modify gfx90a according to your GPU
#    architecture.
# 2) Do not remove -DUSE_HIP.
# 3) Add -DUSE_PLUMED to CFLAGS when use the PLUMED plugin
#    and remove it otherwise.
# 4) Add -DUSE_TABLE to speed up MD simulations with NEP
#    using pre-computed radial functions in the descriptors
###########################################################


###########################################################
# some flags
###########################################################
CC = hipcc
CFLAGS = -std=c++14 -O1 --offload-arch=gfx90a -DUSE_HIP
INC = -I./
LDFLAGS = 
LIBS = -lhipblas -lhipsolver


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


###########################################################
# object files
###########################################################
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)


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
	$(wildcard main_nep/*.cuh)


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


###########################################################
# clean up
###########################################################
clean:
	rm -f */*.o gpumd nep

