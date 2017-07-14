all: gpumd

CC = nvcc
CFLAGS = -O3 -use_fast_math -DUSE_DP -arch=sm_35
#CFLAGS = -O3 -use_fast_math -DUSE_DP -arch=sm_37
#CFLAGS = -O3 -use_fast_math -DUSE_DP -arch=sm_20
#CFLAGS = -O3 -use_fast_math -DUSE_DP -arch=sm_61

gpumd: common.o eam_zhou_2004.o eam_dai_2006.o \
       tersoff_1989_1.o tersoff_1989_2.o \
       sw_1985.o lj1.o ri.o force.o validate.o \
       vac.o hac.o shc.o dump.o heat.o integrate.o \
       neighbor_ON1.o neighbor_ON2.o neighbor.o \
       initialize.o finalize.o run.o velocity.o potential.o \
       parse.o gpumd.o main.o
 
	$(CC) -o gpumd \
       common.o eam_zhou_2004.o eam_dai_2006.o \
       tersoff_1989_1.o tersoff_1989_2.o \
       sw_1985.o lj1.o ri.o force.o validate.o \
       vac.o hac.o shc.o dump.o heat.o integrate.o \
       neighbor_ON1.o neighbor_ON2.o neighbor.o \
       initialize.o finalize.o run.o velocity.o potential.o \
       parse.o gpumd.o main.o

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu
gpumd.o: gpumd.cu
	$(CC) $(CFLAGS) -c gpumd.cu
run.o: run.cu
	$(CC) $(CFLAGS) -c run.cu
parse.o: parse.cu
	$(CC) $(CFLAGS) -c parse.cu
initialize.o: initialize.cu
	$(CC) $(CFLAGS) -c initialize.cu
finalize.o: finalize.cu
	$(CC) $(CFLAGS) -c finalize.cu
velocity.o: velocity.cu
	$(CC) $(CFLAGS) -c velocity.cu
potential.o: potential.cu
	$(CC) $(CFLAGS) -c potential.cu
neighbor.o: neighbor.cu
	$(CC) $(CFLAGS) -c neighbor.cu
neighbor_ON1.o: neighbor_ON1.cu
	$(CC) $(CFLAGS) -c neighbor_ON1.cu
neighbor_ON2.o: neighbor_ON2.cu
	$(CC) $(CFLAGS) -c neighbor_ON2.cu
integrate.o: integrate.cu
	$(CC) $(CFLAGS) -c integrate.cu
heat.o: heat.cu
	$(CC) $(CFLAGS) -c heat.cu
dump.o: dump.cu
	$(CC) $(CFLAGS) -c dump.cu
shc.o: shc.cu
	$(CC) $(CFLAGS) -c shc.cu
vac.o: vac.cu
	$(CC) $(CFLAGS) -c vac.cu
hac.o: hac.cu
	$(CC) $(CFLAGS) -c hac.cu
validate.o: validate.cu
	$(CC) $(CFLAGS) -c validate.cu
force.o: force.cu
	$(CC) $(CFLAGS) -c force.cu
lj1.o: lj1.cu
	$(CC) $(CFLAGS) -c lj1.cu
ri.o: ri.cu
	$(CC) $(CFLAGS) -c ri.cu
eam_zhou_2004.o: eam_zhou_2004.cu
	$(CC) $(CFLAGS) -c eam_zhou_2004.cu	
eam_dai_2006.o: eam_dai_2006.cu
	$(CC) $(CFLAGS) -c eam_dai_2006.cu
sw_1985.o: sw_1985.cu
	$(CC) $(CFLAGS) -c sw_1985.cu
tersoff_1989_1.o: tersoff_1989_1.cu
	$(CC) $(CFLAGS) -c tersoff_1989_1.cu
tersoff_1989_2.o: tersoff_1989_2.cu
	$(CC) $(CFLAGS) -c tersoff_1989_2.cu
common.o: common.cu
	$(CC) $(CFLAGS) -c common.cu

clean:
	rm *o gpumd


