all: gpumd

CC = nvcc
CFLAGS = -std=c++11 -O3 -DUSE_LDG -DUSE_DP -arch=sm_35

gpumd: common.o eam.o tersoff.o rebo_mos2.o \
       sw.o vashishta.o pair.o validate.o hnemd_kappa.o \
       vac.o hac.o shc.o heat.o integrate.o ensemble_lan.o ensemble_bdp.o \
       ensemble.o ensemble_nve.o ensemble_ber.o ensemble_nhc.o \
       neighbor_ON1.o neighbor_ON2.o neighbor.o velocity.o \
       measure.o force.o potential.o parse.o gpumd.o main.o
 
	$(CC) -o gpumd \
       common.o eam.o tersoff.o rebo_mos2.o \
       sw.o vashishta.o pair.o validate.o hnemd_kappa.o \
       vac.o hac.o shc.o heat.o integrate.o ensemble_lan.o ensemble_bdp.o \
       ensemble.o ensemble_nve.o ensemble_ber.o ensemble_nhc.o \
       neighbor_ON1.o neighbor_ON2.o neighbor.o velocity.o \
       measure.o force.o potential.o parse.o gpumd.o main.o

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu
gpumd.o: gpumd.cu
	$(CC) $(CFLAGS) -c gpumd.cu
parse.o: parse.cu
	$(CC) $(CFLAGS) -c parse.cu
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
measure.o: measure.cu
	$(CC) $(CFLAGS) -c measure.cu
integrate.o: integrate.cu
	$(CC) $(CFLAGS) -c integrate.cu
ensemble.o: ensemble.cu
	$(CC) $(CFLAGS) -c ensemble.cu
ensemble_nve.o: ensemble_nve.cu
	$(CC) $(CFLAGS) -c ensemble_nve.cu
ensemble_ber.o: ensemble_ber.cu
	$(CC) $(CFLAGS) -c ensemble_ber.cu
ensemble_nhc.o: ensemble_nhc.cu
	$(CC) $(CFLAGS) -c ensemble_nhc.cu
ensemble_lan.o: ensemble_lan.cu
	$(CC) $(CFLAGS) -c ensemble_lan.cu
ensemble_bdp.o: ensemble_bdp.cu
	$(CC) $(CFLAGS) -c ensemble_bdp.cu
heat.o: heat.cu
	$(CC) $(CFLAGS) -c heat.cu
shc.o: shc.cu
	$(CC) $(CFLAGS) -c shc.cu
vac.o: vac.cu
	$(CC) $(CFLAGS) -c vac.cu
hac.o: hac.cu
	$(CC) $(CFLAGS) -c hac.cu
hnemd_kappa.o: hnemd_kappa.cu
	$(CC) $(CFLAGS) -c hnemd_kappa.cu
validate.o: validate.cu
	$(CC) $(CFLAGS) -c validate.cu
force.o: force.cu
	$(CC) $(CFLAGS) -c force.cu
pair.o: pair.cu
	$(CC) $(CFLAGS) -c pair.cu
eam.o: eam.cu
	$(CC) $(CFLAGS) -c eam.cu	
sw.o: sw.cu
	$(CC) $(CFLAGS) -c sw.cu
vashishta.o: vashishta.cu
	$(CC) $(CFLAGS) -c vashishta.cu
tersoff.o: tersoff.cu
	$(CC) $(CFLAGS) -c tersoff.cu
rebo_mos2.o: rebo_mos2.cu
	$(CC) $(CFLAGS) -c rebo_mos2.cu
common.o: common.cu
	$(CC) $(CFLAGS) -c common.cu

clean:
	rm *o gpumd


