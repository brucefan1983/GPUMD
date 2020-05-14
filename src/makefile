CC = nvcc

# Note: add -DUSE_FCP to CFLAGS when you use the FCP potential
# and remove it when you use any other potentials
 
ifdef OS # For Windows with the cl.exe compiler
CFLAGS = -O3 -arch=sm_35 -DDEBUG -Xcompiler "/wd 4819"
else # For linux
CFLAGS = -std=c++11 -O3 -arch=sm_35 -DDEBUG
endif

INC =
LDFLAGS = 
LIBS = -lcublas

SOURCES = main.cu gpumd.cu read_file.cu run.cu error.cu validate.cu \
	mic.cu atom.cu velocity.cu neighbor.cu neighbor_ON1.cu neighbor_ON2.cu \
	force.cu potential.cu ri.cu lj.cu eam.cu sw.cu vashishta.cu fcp.cu \
	tersoff1988.cu tersoff1989.cu tersoff_modc.cu rebo_mos2.cu tersoff_mini.cu \
	integrate.cu ensemble.cu ensemble_nve.cu ensemble_ber.cu \
	ensemble_nhc.cu ensemble_lan.cu ensemble_bdp.cu \
	measure.cu compute.cu shc.cu vac.cu hac.cu hnemd_kappa.cu box.cu \
	dump_netcdf.cu dump_xyz.cu modal_analysis.cu group.cu

ifdef OS # For Windows with the cl.exe compiler
_OBJ = main.obj gpumd.obj read_file.obj run.obj error.obj validate.obj \
	mic.obj atom.obj velocity.obj neighbor.obj neighbor_ON1.obj neighbor_ON2.obj \
	force.obj potential.obj ri.obj lj.obj eam.obj sw.obj vashishta.obj fcp.obj \
	tersoff1988.obj tersoff1989.obj tersoff_modc.obj rebo_mos2.obj tersoff_mini.obj \
	integrate.obj ensemble.obj ensemble_nve.obj ensemble_ber.obj \
	ensemble_nhc.obj ensemble_lan.obj ensemble_bdp.obj \
	measure.obj compute.obj shc.obj vac.obj hac.obj hnemd_kappa.obj box.obj \
	dump_netcdf.obj dump_xyz.obj modal_analysis.obj group.obj
else
_OBJ = main.o gpumd.o read_file.o run.o error.o validate.o \
	mic.o atom.o velocity.o neighbor.o neighbor_ON1.o neighbor_ON2.o \
	force.o potential.o ri.o lj.o eam.o sw.o vashishta.o fcp.o \
	tersoff1988.o tersoff1989.o tersoff_modc.o rebo_mos2.o tersoff_mini.o \
	integrate.o ensemble.o ensemble_nve.o ensemble_ber.o \
	ensemble_nhc.o ensemble_lan.o ensemble_bdp.o \
	measure.o compute.o shc.o vac.o hac.o hnemd_kappa.o box.o \
	dump_netcdf.o dump_xyz.o modal_analysis.o group.o
endif

HEADERS = gpumd.cuh read_file.cuh run.cuh error.cuh validate.cuh \
	mic.cuh common.cuh atom.cuh velocity.cuh \
	force.cuh potential.cuh ri.cuh lj.cuh eam.cuh sw.cuh vashishta.cuh fcp.cuh \
	tersoff1988.cuh tersoff1989.cuh tersoff_modc.cuh rebo_mos2.cuh tersoff_mini.cuh \
	integrate.cuh ensemble.cuh ensemble_nve.cuh ensemble_ber.cuh \
	ensemble_nhc.cuh ensemble_lan.cuh ensemble_bdp.cuh \
	measure.cuh compute.cuh shc.cuh vac.cuh hac.cuh hnemd_kappa.cuh box.cuh \
	dump_pos.cuh dump_netcdf.cuh dump_xyz.cuh modal_analysis.cuh \
	gpu_vector.cuh group.cuh

ODIR = obj
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: gpumd
	
ifdef OS # for Windows
$(ODIR)/%.obj: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
else # for Linux
$(ODIR)/%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
endif

gpumd: $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)

clean:
ifdef OS
	del $(ODIR)\*obj gpumd.exe gpumd.exp gpumd.lib
else
	rm $(ODIR)/*o gpumd
endif

