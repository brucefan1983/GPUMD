###########################################################
# Note: 
# 2) Add -DDEBUG to CFLAGS if you want to have the same
#    outputs for the same inputs and remove it otherwise.
# 3) You can modify -arch=sm_35 according to 
#    your GPU architecture.
###########################################################


###########################################################
# some flags
###########################################################
CC = nvcc
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 -arch=sm_35 --use_fast_math -Xcompiler "/wd 4819" -DDEBUG
else # For linux
CFLAGS = -std=c++14 -O3 -arch=sm_35 --use_fast_math
endif
INC = -I./
LDFLAGS = 
LIBS =


###########################################################
# source files
###########################################################
SOURCES = $(wildcard *.cu)


###########################################################
# object files
###########################################################
ifdef OS # For Windows with the cl.exe compiler
OBJ = $(SOURCES:.cu=.obj)
else
OBJ = $(SOURCES:.cu=.o)
endif


###########################################################
# headers
###########################################################
HEADERS = $(wildcard *.cuh)


###########################################################
# executables
###########################################################
all: nep
nep: $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	

###########################################################
# rules for building object files
###########################################################
ifdef OS # for Windows
%.obj: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else # for Linux
%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif


###########################################################
# clean up
###########################################################
clean:
ifdef OS
	del *.obj *.exp *.lib *.exe
else
	rm *.o nep
endif
