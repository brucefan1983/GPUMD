###########################################################
# Standalone build for GPUMD replica dynamics
###########################################################

CC = nvcc
CUDA_ARCH = -arch=sm_60
WARN_FLAGS = -Wall -Wextra
WARN_FLAGS += -Wno-sign-compare -Wno-unused-parameter
WARN_FLAGS += -Wnull-dereference -Wstringop-overflow -Wstringop-truncation -Wrestrict
WARN_FLAGS += -Wduplicated-cond -Wduplicated-branches -Wlogical-op
WARN_FLAGS += -Wsequence-point -Wswitch-default
WARN_FLAGS += -Wcast-align -Wcast-qual -Wvla
WARN_FLAGS += -Wformat -Wformat-security -Wformat-overflow -Wformat-truncation
ifdef OS
CFLAGS = -O3 $(CUDA_ARCH)
else
CFLAGS = -std=c++14 -O3 $(CUDA_ARCH) -Xcompiler="$(WARN_FLAGS)"
endif
INC = -I./ -I./integrate
LDFLAGS =
LIBS = -lcufft
TARGET = ../gpumd_replica

SOURCES =                         \
	main.cu                         \
	replica.cu                      \
	replica_common.cu               \
	replica_prd.cu                  \
	replica_remd.cu                 \
	velocity.cu                     \
	force/ewald.cu                  \
	force/force.cu                  \
	force/neighbor.cu               \
	force/nep.cu                    \
	force/nep_charge.cu             \
	force/pppm.cu                   \
	force/potential.cu              \
	integrate/ensemble.cu           \
	integrate/ensemble_bdp.cu       \
	minimize/minimizer.cu           \
	minimize/minimizer_fire.cu      \
	model/atom.cu                   \
	model/box.cu                    \
	model/group.cu                  \
	model/read_xyz.cu               \
	utilities/error.cu              \
	utilities/main_common.cu        \
	utilities/read_file.cu

HEADERS =                       \
	$(wildcard *.cuh)            \
	$(wildcard force/*.cuh)      \
	$(wildcard integrate/*.cuh)  \
	$(wildcard minimize/*.cuh)   \
	$(wildcard model/*.cuh)      \
	$(wildcard utilities/*.cuh)

ifdef OS
OBJECTS = $(SOURCES:.cu=.obj)
else
OBJECTS = $(SOURCES:.cu=.o)
endif

.PHONY: all gpumd_replica clean

all: gpumd_replica

gpumd_replica: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gpumd_replica executable is successfully compiled!
	@echo =================================================

ifdef OS
%.obj: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.obj: force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
integrate/%.obj: integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.obj: minimize/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.obj: model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.obj: utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else
%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
force/%.o: force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
integrate/%.o: integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
minimize/%.o: minimize/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
model/%.o: model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
utilities/%.o: utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif

clean:
ifdef OS
	del /s *.obj
	del ..\gpumd_replica.exe
else
	rm -f *.o force/*.o integrate/*.o minimize/*.o model/*.o utilities/*.o $(TARGET)
endif
