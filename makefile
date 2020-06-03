###########################################################
# Note: 
# 1) Add -DUSE_FCP to CFLAGS when use the FCP potential 
#    and remove it otherwise
# 2) Remove -DDEBUG for production run. 
# 3) You can modify -arch=sm_35 according to 
#    your GPU architecture
###########################################################


###########################################################
# some flags
###########################################################
CC = nvcc
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 -arch=sm_35 -DDEBUG -Xcompiler "/wd 4819" 
else # For linux
CFLAGS = -std=c++11 -O3 -arch=sm_35 -DDEBUG
endif
INC = -Isrc/
LDFLAGS = 
LIBS = -lcublas -lcusolver


###########################################################
# source files
###########################################################
SOURCES_GPUMD = $(wildcard src/utilities/*.cu) \
          $(wildcard src/gpumd/*.cu)           \
          $(wildcard src/integrate/*.cu)       \
          $(wildcard src/force/*.cu)           \
          $(wildcard src/measure/*.cu)         \
	      $(wildcard src/model/*.cu)
SOURCES_PHONON = $(wildcard src/utilities/*.cu) \
          $(wildcard src/phonon/*.cu)           \
          $(wildcard src/force/*.cu)            \
	      $(wildcard src/model/*.cu) 


###########################################################
# objective files
###########################################################
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_PHONON = $(SOURCES_PHONON:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_PHONON = $(SOURCES_PHONON:.cu=.o)
endif


###########################################################
# headers
###########################################################
HEADERS = $(wildcard src/utilities/*.cuh)  \
          $(wildcard src/gpumd/*.cuh)      \
	      $(wildcard src/integrate/*.cuh)  \
	      $(wildcard src/force/*.cuh)      \
	      $(wildcard src/measure/*.cuh)    \
	      $(wildcard src/model/*.cuh)      \
		  $(wildcard src/phonon/*.cuh)


###########################################################
# executables
###########################################################
all: gpumd phonon
gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
phonon: $(OBJ_PHONON)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	

###########################################################
# rules for building objective files
###########################################################
ifdef OS # for Windows
src/integrate/%.obj: src/integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/force/%.obj: src/force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/measure/%.obj: src/measure/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/gpumd/%.obj: src/gpumd/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/utilities/%.obj: src/utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/model/%.obj: src/model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/phonon/%.obj: src/phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else # for Linux
%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/integrate/%.o: src/integrate/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/force/%.o: src/force/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/measure/%.o: src/measure/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/gpumd/%.o: src/gpumd/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/utilities/%.o: src/utilities/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/model/%.o: src/model/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
src/phonon/%.o: src/phonon/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif


###########################################################
# clean up
###########################################################
clean:
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm */*.o gpumd phonon
endif

