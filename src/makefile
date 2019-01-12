CC = nvcc
CFLAGS = -std=c++11 -O3 -arch=sm_35

SOURCES = main.cu gpumd.cu parse.cu error.cu validate.cu \
	atom.cu velocity.cu neighbor.cu neighbor_ON1.cu neighbor_ON2.cu \
	force.cu potential.cu pair.cu eam.cu sw.cu vashishta.cu tersoff.cu \
	rebo_mos2.cu \
	integrate.cu ensemble.cu ensemble_nve.cu ensemble_ber.cu \
	ensemble_nhc.cu ensemble_lan.cu ensemble_bdp.cu \
	measure.cu compute.cu shc.cu vac.cu hac.cu hnemd_kappa.cu

_OBJ = main.o gpumd.o parse.o error.o validate.o \
	atom.o velocity.o neighbor.o neighbor_ON1.o neighbor_ON2.o \
	force.o potential.o pair.o eam.o sw.o vashishta.o tersoff.o rebo_mos2.o \
	integrate.o ensemble.o ensemble_nve.o ensemble_ber.o \
	ensemble_nhc.o ensemble_lan.o ensemble_bdp.o \
	measure.o compute.o shc.o vac.o hac.o hnemd_kappa.o

HEADERS = gpumd.cuh error.cuh validate.cuh ldg.cuh common.cuh \
	atom.cuh \
	force.cuh potential.cuh pair.cuh eam.cuh sw.cuh vashishta.cuh tersoff.cuh \
	rebo_mos2.cuh \
	integrate.cuh ensemble.cuh ensemble_nve.cuh ensemble_ber.cuh \
	ensemble_nhc.cuh ensemble_lan.cuh ensemble_bdp.cuh \
	measure.cuh compute.cuh shc.cuh vac.cuh hac.cuh hnemd_kappa.cuh \

ODIR = obj
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: gpumd

$(ODIR)/%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

gpumd: $(OBJ)
	$(CC) $^ -o $@

clean:
	rm $(ODIR)/*o gpumd

