###########################################################
# Note: 
# 1) Add -DUSE_FCP to CFLAGS when use the FCP potential 
#    and remove it otherwise.
# 2) Add -DDEBUG to CFLAGS if you want to have the same
#    outputs for the same inputs and remove it otherwise.
# 3) You can modify -arch=sm_35 according to 
#    your GPU architecture.
# 4) For Windows systems, if you got errors like 
#    c1xx : fatal error C1083: Cannot open source file: ...
#    You can consider removing "-Xcompiler "/wd 4819"
###########################################################


###########################################################
# some flags
###########################################################
CC = nvcc
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 -arch=sm_35 -Xcompiler "/wd 4819" -DDEBUG
else # For linux
CFLAGS = -std=c++11 -O3 -arch=sm_35 
endif
INC = -I./
LDFLAGS = 
LIBS = -lcublas -lcusolver


###########################################################
# source files
###########################################################
SOURCES_GPUMD =                   \
	$(wildcard main_gpumd/*.cu)   \
	$(wildcard minimize/*.cu)     \
	$(wildcard integrate/*.cu)    \
	$(wildcard force/*.cu)        \
	$(wildcard measure/*.cu)      \
	$(wildcard model/*.cu)        \
	$(wildcard utilities/*.cu)
SOURCES_PHONON =                  \
	$(wildcard main_phonon/*.cu)  \
	$(wildcard minimize/*.cu)     \
	$(wildcard force/*.cu)        \
	$(wildcard model/*.cu)        \
	$(wildcard utilities/*.cu)
SOURCES_NEP =                     \
	$(wildcard main_nep/*.cu)


###########################################################
# object files
###########################################################
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_PHONON = $(SOURCES_PHONON:.cu=.obj)
OBJ_NEP = $(SOURCES_NEP:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_PHONON = $(SOURCES_PHONON:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
endif


###########################################################
# headers
###########################################################
HEADERS =                         \
	$(wildcard utilities/*.cuh)   \
	$(wildcard main_gpumd/*.cuh)  \
	$(wildcard integrate/*.cuh)   \
	$(wildcard minimize/*.cuh)    \
	$(wildcard force/*.cuh)       \
	$(wildcard measure/*.cuh)     \
	$(wildcard model/*.cuh)       \
	$(wildcard main_phonon/*.cuh) \
	$(wildcard main_nep/*.cuh)


###########################################################
# executables
###########################################################
all: gpumd phonon nep
gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
phonon: $(OBJ_PHONON)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
nep: $(OBJ_NEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	

###########################################################
# rules for building object files
###########################################################
ifdef OS # for Windows
integrate/%.obj: integrate/%.cu $(HEADERS)
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
main_phonon/%.obj: main_phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.obj: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else # for Linux
%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
integrate/%.o: integrate/%.cu $(HEADERS)
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
main_phonon/%.o: main_phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif


###########################################################
# clean up
###########################################################
clean:
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm */*.o gpumd phonon nep
endif

