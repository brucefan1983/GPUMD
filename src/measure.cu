/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/


/*----------------------------------------------------------------------------80
The driver class dealing with measurement.
------------------------------------------------------------------------------*/


#include "measure.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "atom.cuh"
#include "error.cuh"
#include "read_file.cuh"
#include "dump_xyz.cuh"
#ifdef USE_NETCDF
#include "dump_netcdf.cuh"
#endif

#define DIM 3
#define NUM_OF_HEAT_COMPONENTS 5
#define NUM_OF_PROPERTIES      5 


Measure::Measure(char *input_dir)
{
    dump_thermo = 0;
    dump_restart = 0;
    dump_velocity = 0;
    dump_force = 0;
    dump_potential = 0;
    dump_virial = 0;
    dump_heat = 0;
    dump_pos = NULL; // to avoid deleting random memory in run
    strcpy(file_thermo, input_dir);
    strcpy(file_restart, input_dir);
    strcpy(file_velocity, input_dir);
    strcpy(file_force, input_dir);
    strcpy(file_potential, input_dir);
    strcpy(file_virial, input_dir);
    strcpy(file_heat, input_dir);
    strcat(file_thermo, "/thermo.out");
    strcat(file_restart, "/restart.out");
    strcat(file_velocity, "/v.out");
    strcat(file_force, "/f.out");
    strcat(file_potential, "/potential.out");
    strcat(file_virial, "/virial.out");
    strcat(file_heat, "/heat.out");
}


Measure::~Measure(void)
{
    // nothing
}


void Measure::initialize(char* input_dir, Atom *atom)
{
    if (dump_thermo)    {fid_thermo   = my_fopen(file_thermo,   "a");}
    if (dump_velocity)  {fid_velocity = my_fopen(file_velocity, "a");}
    if (dump_force)     {fid_force    = my_fopen(file_force,    "a");}
    if (dump_potential) {fid_potential= my_fopen(file_potential,"a");}
    if (dump_virial)    {fid_virial   = my_fopen(file_virial,   "a");}
    if (dump_heat)      {fid_heat     = my_fopen(file_heat,     "a");}
    if (dump_pos)       {dump_pos->initialize(input_dir);}
    vac.preprocess(atom);
    dos.preprocess(atom, &vac);
    hac.preprocess(atom);
    shc.preprocess(atom);
    compute.preprocess(input_dir, atom);
    hnemd.preprocess(atom);
    gkma.preprocess(input_dir, atom);
    hnema.preprocess(input_dir, atom);
}


void Measure::finalize
(char *input_dir, Atom *atom, Integrate *integrate)
{
    if (dump_thermo)    {fclose(fid_thermo);    dump_thermo    = 0;}
    if (dump_restart)   {                       dump_restart   = 0;}
    if (dump_velocity)  {fclose(fid_velocity);  dump_velocity  = 0;}
    if (dump_force)     {fclose(fid_force);     dump_force     = 0;}
    if (dump_potential) {fclose(fid_potential); dump_potential = 0;}
    if (dump_virial)    {fclose(fid_virial);    dump_virial    = 0;}
    if (dump_heat)      {fclose(fid_heat);      dump_heat      = 0;}
    if (dump_pos)       {dump_pos->finalize();}
    vac.postprocess(input_dir, atom, &dos, &sdc);
    hac.postprocess(input_dir, atom, integrate);
    shc.postprocess(input_dir);
    compute.postprocess(atom, integrate);
    hnemd.postprocess(atom);
    gkma.postprocess();
    hnema.postprocess();
}


void Measure::dump_thermos
(FILE *fid, Atom *atom, Integrate *integrate, int step)
{
    if (!dump_thermo) return;
    if ((step + 1) % sample_interval_thermo != 0) return;
    real *thermo; MY_MALLOC(thermo, real, NUM_OF_PROPERTIES);
    int m1 = sizeof(real) * NUM_OF_PROPERTIES;
    CHECK(cudaMemcpy(thermo, atom->thermo, m1, cudaMemcpyDeviceToHost));
    int N_fixed = (integrate->fixed_group == -1) ? 0 :
        atom->group[0].cpu_size[integrate->fixed_group];
    real energy_kin = (0.5 * DIM) * (atom->N - N_fixed) * K_B * thermo[0];
    fprintf(fid, "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e", thermo[0],
        energy_kin, thermo[1], thermo[2]*PRESSURE_UNIT_CONVERSION,
        thermo[3]*PRESSURE_UNIT_CONVERSION, thermo[4]*PRESSURE_UNIT_CONVERSION);
    int number_of_box_variables = atom->box.triclinic ? 9 : 3;
    for (int m = 0; m < number_of_box_variables; ++m)
    {
        fprintf(fid, "%20.10e", atom->box.cpu_h[m]);
    }
    fprintf(fid, "\n"); fflush(fid); MY_FREE(thermo);
}


static void gpu_dump_3(int N, FILE *fid, real *a, real *b, real *c)
{
    real *cpu_a, *cpu_b, *cpu_c;
    MY_MALLOC(cpu_a, real, N);
    MY_MALLOC(cpu_b, real, N);
    MY_MALLOC(cpu_c, real, N);
    CHECK(cudaMemcpy(cpu_a, a, sizeof(real) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_b, b, sizeof(real) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cpu_c, c, sizeof(real) * N, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%g %g %g\n", cpu_a[n], cpu_b[n], cpu_c[n]);
    }
    fflush(fid);
    MY_FREE(cpu_a); MY_FREE(cpu_b); MY_FREE(cpu_c);
}


void Measure::dump_restarts(Atom *atom, int step)
{
    if (!dump_restart) return;
    if ((step + 1) % sample_interval_restart != 0) return;
    int memory = sizeof(real) * atom->N;
    CHECK(cudaMemcpy(atom->cpu_x, atom->x, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_y, atom->y, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_z, atom->z, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_vx, atom->vx, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_vy, atom->vy, memory, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(atom->cpu_vz, atom->vz, memory, cudaMemcpyDeviceToHost));
    fid_restart = my_fopen(file_restart, "w"); 
    fprintf(fid_restart, "%d %d %g %d %d %d\n", atom->N, atom->neighbor.MN,
        atom->neighbor.rc, atom->box.triclinic, 1,
        atom->num_of_grouping_methods);
    if (atom->box.triclinic == 0)
    {
        fprintf(fid_restart, "%d %d %d %g %g %g\n", atom->box.pbc_x,
            atom->box.pbc_y, atom->box.pbc_z, atom->box.cpu_h[0],
            atom->box.cpu_h[1], atom->box.cpu_h[2]);
    }
    else
    {
        fprintf(fid_restart, "%d %d %d\n", atom->box.pbc_x,
            atom->box.pbc_y, atom->box.pbc_z);
        for (int d1 = 0; d1 < 3; ++d1)
        {
            for (int d2 = 0; d2 < 3; ++d2)
            {
                fprintf(fid_restart, "%g ", atom->box.cpu_h[d1 * 3 + d2]);
            }
            fprintf(fid_restart, "\n");
        }
    }
    for (int n = 0; n < atom->N; n++)
    {
        fprintf(fid_restart, "%d %g %g %g %g %g %g %g ", atom->cpu_type[n],
            atom->cpu_x[n], atom->cpu_y[n], atom->cpu_z[n], atom->cpu_mass[n],
            atom->cpu_vx[n], atom->cpu_vy[n], atom->cpu_vz[n]);
        for (int m = 0; m < atom->num_of_grouping_methods; ++m)
        {
            fprintf(fid_restart, "%d ", atom->group[m].cpu_label[n]);
        }
        fprintf(fid_restart, "\n");
    }
    fflush(fid_restart);
    fclose(fid_restart);
}


void Measure::dump_velocities(FILE *fid, Atom *atom, int step)
{
    if (!dump_velocity) return;
    if ((step + 1) % sample_interval_velocity != 0) return;
    gpu_dump_3(atom->N, fid, atom->vx, atom->vy, atom->vz);
}


void Measure::dump_forces(FILE *fid, Atom *atom, int step)
{
    if (!dump_force) return;
    if ((step + 1) % sample_interval_force != 0) return;
    gpu_dump_3(atom->N, fid, atom->fx, atom->fy, atom->fz);
}


void Measure::dump_virials(FILE *fid, Atom *atom, int step)
{
    if (!dump_virial) return;
    if ((step + 1) % sample_interval_virial != 0) return;
    gpu_dump_3
    (
        atom->N, fid,
        atom->virial_per_atom,
        atom->virial_per_atom + atom->N,
        atom->virial_per_atom + atom->N * 2
    );
}


static void gpu_dump_1(int N, FILE *fid, real *a)
{
    real *cpu_a; MY_MALLOC(cpu_a, real, N);
    CHECK(cudaMemcpy(cpu_a, a, sizeof(real) * N, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++) { fprintf(fid, "%g\n", cpu_a[n]); }
    fflush(fid); MY_FREE(cpu_a);
}


void Measure::dump_potentials(FILE *fid, Atom *atom, int step)
{
    if (!dump_potential) return;
    if ((step + 1) % sample_interval_potential != 0) return;
    gpu_dump_1(atom->N, fid, atom->potential_per_atom);
}


// calculate the per-atom heat current 
static __global__ void gpu_get_peratom_heat
(
    int N, real *sxx, real *sxy, real *sxz, real *syx, real *syy, real *syz,
    real *szx, real *szy, real *szz, real *vx, real *vy, real *vz, 
    real *jx_in, real *jx_out, real *jy_in, real *jy_out, real *jz
)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N)
    {
        jx_in[n] = sxx[n] * vx[n] + sxy[n] * vy[n];
        jx_out[n] = sxz[n] * vz[n];
        jy_in[n] = syx[n] * vx[n] + syy[n] * vy[n];
        jy_out[n] = syz[n] * vz[n];
        jz[n] = szx[n] * vx[n] + szy[n] * vy[n] + szz[n] * vz[n];
    }
}


void Measure::dump_heats(FILE *fid, Atom *atom, int step)
{
    if (!dump_heat) return;
    if ((step + 1) % sample_interval_heat != 0) return;

    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_get_peratom_heat<<<(atom->N - 1) / 128 + 1, 128>>>
    (
        atom->N, 
        atom->virial_per_atom, 
        atom->virial_per_atom + atom->N * 3,
        atom->virial_per_atom + atom->N * 4,
        atom->virial_per_atom + atom->N * 6,
        atom->virial_per_atom + atom->N * 1,
        atom->virial_per_atom + atom->N * 5,
        atom->virial_per_atom + atom->N * 7,
        atom->virial_per_atom + atom->N * 8,
        atom->virial_per_atom + atom->N * 2,
        atom->vx, atom->vy, atom->vz, 
        atom->heat_per_atom, 
        atom->heat_per_atom + atom->N,
        atom->heat_per_atom + atom->N * 2,
        atom->heat_per_atom + atom->N * 3,
        atom->heat_per_atom + atom->N * 4
    );
    CUDA_CHECK_KERNEL

    gpu_dump_1(atom->N * NUM_OF_HEAT_COMPONENTS, fid, atom->heat_per_atom);
}


void Measure::process
(char *input_dir, Atom *atom, Integrate *integrate, int step)
{
    dump_thermos(fid_thermo, atom, integrate, step);
    dump_restarts(atom, step);
    dump_velocities(fid_velocity, atom, step);
    dump_forces(fid_force, atom, step);
    dump_potentials(fid_potential, atom, step);
    dump_virials(fid_virial, atom, step);
    dump_heats(fid_heat, atom, step);
    compute.process(step, atom, integrate);
    vac.process(step, atom);
    hac.process(step, input_dir, atom);
    shc.process(step, atom);
    hnemd.process(step, input_dir, atom, integrate);
    gkma.process(step, atom);
    hnema.process(step, atom, integrate, hnemd.fe);
    if (dump_pos) dump_pos->dump(atom, step);

}


void Measure::parse_dump_thermo(char **param,  int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_thermo should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_thermo))
    {
        PRINT_INPUT_ERROR("thermo dump interval should be an integer number.\n");
    }
    dump_thermo = 1;
    printf("Dump thermo every %d steps.\n", sample_interval_thermo);
}


void Measure::parse_dump_position(char **param, int num_param, Atom *atom)
{
	int interval;

    if (num_param < 2)
    {
        PRINT_INPUT_ERROR("dump_position should have at least 1 parameter.\n");
    }
    if (num_param > 6)
    {
    	PRINT_INPUT_ERROR("dump_position has too many parameters.\n");
    }

    // sample interval
    if (!is_valid_int(param[1], &interval))
    {
        PRINT_INPUT_ERROR("position dump interval should be an integer number.\n");
    }

    int format = 0; // default xyz
    int precision = 0; // default normal (unlesss netCDF -> 64 bit)
    // Process optional arguments
    for (int k = 2; k < num_param; k++)
    {
    	// format check
    	if (strcmp(param[k], "format") == 0)
    	{
    		// check if there are enough inputs
    		if (k + 2 > num_param)
    		{
    			PRINT_INPUT_ERROR("Not enough arguments for optional "
    					" 'format' dump_position command.\n");
    		}
    		if ((strcmp(param[k+1], "xyz") != 0) &&
				(strcmp(param[k+1], "netcdf") != 0))
    		{
    			PRINT_INPUT_ERROR("Invalid format for dump_position command.\n");
    		}
    		else if(strcmp(param[k+1], "netcdf") == 0)
    		{
    			format = 1;
    			k++;
    		}
    	}
    	// precision check
    	else if(strcmp(param[k], "precision") == 0)
    	{
    		// check for enough inputs
    		if (k + 2 > num_param)
			{
				PRINT_INPUT_ERROR("Not enough arguments for optional "
						" 'precision' dump_position command.\n");
			}
    		if ((strcmp(param[k+1], "single") != 0) &&
				(strcmp(param[k+1], "double") != 0))
			{
				PRINT_INPUT_ERROR("Invalid precision for dump_position command.\n");
			}
			else
			{
				if(strcmp(param[k+1], "single") == 0)
				{
					precision = 1;
				}
				else if(strcmp(param[k+1], "double") == 0)
                {
                    precision = 2;
                }
				k++;
			}
    	}
    }

    if (format == 1) // netcdf output
    {
#ifdef USE_NETCDF
    	DUMP_NETCDF *dump_netcdf = new DUMP_NETCDF(atom->N, atom->global_time);
    	dump_pos = dump_netcdf;
    	if (!precision) precision = 2; // double precision default
#else
    	PRINT_INPUT_ERROR("USE_NETCDF flag is not set. NetCDF output not available.\n");
#endif
    }
    else // xyz default output
    {
    	DUMP_XYZ *dump_xyz = new DUMP_XYZ();
    	dump_pos = dump_xyz;
    }
    dump_pos->interval = interval;
    dump_pos->precision = precision;


    if (precision == 1 && format)
    {
    	printf("Note: Single precision netCDF output does not follow AMBER conventions.\n"
    	       "      However, it will still work for many readers.\n");
    }

    printf("Dump position every %d steps.\n",
        dump_pos->interval);
}


void Measure::parse_dump_restart(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_restart should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_restart))
    {
        PRINT_INPUT_ERROR("restart dump interval should be an integer number.\n");
    }
    dump_restart = 1;
    printf("Dump restart every %d steps.\n", sample_interval_restart);
}


void Measure::parse_dump_velocity(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_velocity should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_velocity))
    {
        PRINT_INPUT_ERROR("velocity dump interval should be an integer number.\n");
    }
    dump_velocity = 1;
    printf("Dump velocity every %d steps.\n",
        sample_interval_velocity);
}


void Measure::parse_dump_force(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_force should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_force))
    {
        PRINT_INPUT_ERROR("force dump interval should be an integer number.\n");
    }
    dump_force = 1;
    printf("Dump force every %d steps.\n", sample_interval_force);
}


void Measure::parse_dump_potential(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_potential should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_potential))
    {
        PRINT_INPUT_ERROR("potential dump interval should be an integer number.\n");
    }
    dump_potential = 1;
    printf("Dump potential every %d steps.\n",
        sample_interval_potential);
}


void Measure::parse_dump_virial(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_virial should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_virial))
    {
        PRINT_INPUT_ERROR("virial dump interval should be an integer number.\n");
    }
    dump_virial = 1;
    printf("Dump virial every %d steps.\n",
        sample_interval_virial);
}


void Measure::parse_dump_heat(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_heat should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &sample_interval_heat))
    {
        PRINT_INPUT_ERROR("heat dump interval should be an integer number.\n");
    }
    dump_heat = 1;
    printf("Dump heat every %d steps.\n", sample_interval_heat);
}


// Helper functions for parse_compute_dos
void Measure::parse_group(char **param, int *k, Group *group)
{
	// grouping_method
	if (!is_valid_int(param[*k+1], &vac.grouping_method))
	{
		PRINT_INPUT_ERROR("grouping method for VAC should be an integer number.\n");
	}
	if (vac.grouping_method < 0 || vac.grouping_method > 2)
	{
		PRINT_INPUT_ERROR("grouping method for VAC should be 0 <= x <= 2.\n");
	}
	// group
	if (!is_valid_int(param[*k+2], &vac.group))
	{
		PRINT_INPUT_ERROR("group for VAC should be an integer number.\n");
	}
	if (vac.group < 0 ||
			vac.group > group[vac.grouping_method].number)
	{
		PRINT_INPUT_ERROR("group for VAC must be >= 0 and < number of groups.\n");
	}
	*k += 2; // update index for next command
}


void Measure::parse_num_dos_points(char **param, int *k)
{
	// number of DOS points
	if (!is_valid_int(param[*k+1], &dos.num_dos_points))
	{
		PRINT_INPUT_ERROR("number of DOS points for VAC should be an integer "
				"number.\n");
	}
	if (dos.num_dos_points < 1)
	{
		PRINT_INPUT_ERROR("number of DOS points for DOS must be > 0.\n");
	}
	*k += 1; //
}


void Measure::parse_compute_dos(char **param,  int num_param, Group *group)
{
    printf("Compute phonon DOS.\n");
    vac.compute_dos = 1;

    if (num_param < 4)
    {
        PRINT_INPUT_ERROR("compute_dos should have at least 3 parameters.\n");
    }
    if (num_param > 9)
	{
		PRINT_INPUT_ERROR("compute_dos has too many parameters.\n");
	}

    // sample interval
    if (!is_valid_int(param[1], &vac.sample_interval))
    {
        PRINT_INPUT_ERROR("sample interval for VAC should be an integer number.\n");
    }
    if (vac.sample_interval <= 0)
    {
        PRINT_INPUT_ERROR("sample interval for VAC should be positive.\n");
    }
    printf("    sample interval is %d.\n", vac.sample_interval);

    // number of correlation steps
    if (!is_valid_int(param[2], &vac.Nc))
    {
        PRINT_INPUT_ERROR("Nc for VAC should be an integer number.\n");
    }
    if (vac.Nc <= 0)
    {
        PRINT_INPUT_ERROR("Nc for VAC should be positive.\n");
    }
    printf("    Nc is %d.\n", vac.Nc);

    // maximal omega
    if (!is_valid_real(param[3], &dos.omega_max))
    {
        PRINT_INPUT_ERROR("omega_max should be a real number.\n");
    }
    if (dos.omega_max <= 0)
    {
        PRINT_INPUT_ERROR("omega_max should be positive.\n");
    }
    printf("    omega_max is %g THz.\n", dos.omega_max);

    // Process optional arguments
    for (int k = 4; k < num_param; k++)
    {
    	if (strcmp(param[k], "group") == 0)
    	{
    		// check if there are enough inputs
    		if (k + 3 > num_param)
    		{
    			PRINT_INPUT_ERROR("Not enough arguments for optional "
    					"'group' DOS command.\n");
    		}
    		parse_group(param, &k, group);
    		printf("    grouping_method is %d and group is %d.\n",
    				vac.grouping_method, vac.group);
    	}
    	else if (strcmp(param[k], "num_dos_points") == 0)
    	{
    		// check if there are enough inputs
    		if (k + 2 > num_param)
    		{
    			PRINT_INPUT_ERROR("Not enough arguments for optional "
						"'group' dos command.\n");
    		}
    		parse_num_dos_points(param, &k);
    		printf("    num_dos_points is %d.\n",dos.num_dos_points);
    	}
    	else
    	{
    		PRINT_INPUT_ERROR("Unrecognized argument in compute_dos.\n");
    	}
    }
}


void Measure::parse_compute_sdc(char **param,  int num_param, Group *group)
{
    printf("Compute SDC.\n");
    vac.compute_sdc = 1;

    if (num_param < 3)
    {
        PRINT_INPUT_ERROR("compute_sdc should have at least 2 parameters.\n");
    }
    if (num_param > 6)
    {
    	PRINT_INPUT_ERROR("compute_sdc has too many parameters.\n");
    }

    // sample interval
    if (!is_valid_int(param[1], &vac.sample_interval))
    {
        PRINT_INPUT_ERROR("sample interval for VAC should be an integer number.\n");
    }
    if (vac.sample_interval <= 0)
    {
        PRINT_INPUT_ERROR("sample interval for VAC should be positive.\n");
    }
    printf("    sample interval is %d.\n", vac.sample_interval);

    // number of correlation steps
    if (!is_valid_int(param[2], &vac.Nc))
    {
        PRINT_INPUT_ERROR("Nc for VAC should be an integer number.\n");
    }
    if (vac.Nc <= 0)
    {
        PRINT_INPUT_ERROR("Nc for VAC should be positive.\n");
    }
    printf("    Nc is %d.\n", vac.Nc);

    // Process optional arguments
	for (int k = 3; k < num_param; k++)
	{
		if (strcmp(param[k], "group") == 0)
		{
			// check if there are enough inputs
			if (k + 3 > num_param)
			{
				PRINT_INPUT_ERROR("Not enough arguments for optional "
						"'group' SDC command.\n");
			}
			parse_group(param, &k, group);
			printf("    grouping_method is %d and group is %d.\n",
					vac.grouping_method, vac.group);
		}
		else
		{
			PRINT_INPUT_ERROR("Unrecognized argument in compute_sdc.\n");
		}
	}
}


void Measure::parse_compute_hac(char **param, int num_param)
{
    hac.compute = 1;

    printf("Compute HAC.\n");

    if (num_param != 4)
    {
        PRINT_INPUT_ERROR("compute_hac should have 3 parameters.\n");
    }

    if (!is_valid_int(param[1], &hac.sample_interval))
    {
        PRINT_INPUT_ERROR("sample interval for HAC should be an integer number.\n");
    }
    printf("    sample interval is %d.\n", hac.sample_interval);

    if (!is_valid_int(param[2], &hac.Nc))
    {
        PRINT_INPUT_ERROR("Nc for HAC should be an integer number.\n");
    }
    printf("    Nc is %d\n", hac.Nc);

    if (!is_valid_int(param[3], &hac.output_interval))
    {
        PRINT_INPUT_ERROR("output_interval for HAC should be an integer number.\n");
    }
    printf("    output_interval is %d\n", hac.output_interval);
}


void Measure::parse_compute_gkma(char **param, int num_param, Atom* atom)
{
    gkma.compute = 1;

    printf("Compute modal heat current using GKMA method.\n");

    /*
     * There is a hidden feature that allows for specification of atom
     * types to included (must be contiguously defined like potentials)
     * -- Works for types only, not groups --
     */

    if (num_param != 6 && num_param != 9)
    {
        PRINT_INPUT_ERROR("compute_gkma should have 5 parameters.\n");
    }
    if (!is_valid_int(param[1], &gkma.sample_interval) ||
        !is_valid_int(param[2], &gkma.first_mode)      ||
        !is_valid_int(param[3], &gkma.last_mode)       )
    {
        PRINT_INPUT_ERROR("A parameter for GKMA should be an integer.\n");
    }

    if (strcmp(param[4], "bin_size") == 0)
    {
        gkma.f_flag = 0;
        if(!is_valid_int(param[5], &gkma.bin_size))
        {
            PRINT_INPUT_ERROR("GKMA bin_size must be an integer.\n");
        }
    }
    else if (strcmp(param[4], "f_bin_size") == 0)
    {
        gkma.f_flag = 1;
        if(!is_valid_real(param[5], &gkma.f_bin_size))
        {
            PRINT_INPUT_ERROR("GKMA f_bin_size must be a real number.\n");
        }
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid binning keyword for compute_gkma.\n");
    }

    GKMA *g = &gkma;
    // Parameter checking
    if (g->sample_interval < 1  || g->first_mode < 1 || g->last_mode < 1)
        PRINT_INPUT_ERROR("compute_gkma parameters must be positive integers.\n");
    if (g->first_mode > g->last_mode)
        PRINT_INPUT_ERROR("first_mode <= last_mode required.\n");

    printf("    sample_interval is %d.\n"
           "    first_mode is %d.\n"
           "    last_mode is %d.\n",
          g->sample_interval, g->first_mode, g->last_mode);

    if (g->f_flag)
    {
        if (g->f_bin_size <= 0.0)
        {
            PRINT_INPUT_ERROR("bin_size must be greater than zero.\n");
        }
        printf("    Bin by frequency.\n"
               "    f_bin_size is %f THz.\n", g->f_bin_size);
    }
    else
    {
        if (g->bin_size < 1)
        {
            PRINT_INPUT_ERROR("compute_gkma parameters must be positive integers.\n");
        }
        int num_modes = g->last_mode - g->first_mode + 1;
        if (num_modes % g->bin_size != 0)
            PRINT_INPUT_ERROR("number of modes must be divisible by bin_size.\n");
        printf("    Bin by modes.\n"
               "    bin_size is %d THz.\n", g->bin_size);
    }

    // Hidden feature implementation
    if (num_param == 9)
    {
        if (strcmp(param[6], "atom_range") == 0)
        {
            if(!is_valid_int(param[7], &gkma.atom_begin) ||
               !is_valid_int(param[8], &gkma.atom_end))
            {
                PRINT_INPUT_ERROR("GKMA atom_begin & atom_end must be integers.\n");
            }
            if (gkma.atom_begin > gkma.atom_end)
            {
                PRINT_INPUT_ERROR("atom_begin must be less than atom_end.\n");
            }
            if (gkma.atom_begin < 0)
            {
                PRINT_INPUT_ERROR("atom_begin must be greater than 0.\n");
            }
            if (gkma.atom_end >= atom->number_of_types)
            {
                PRINT_INPUT_ERROR("atom_end must be greater than 0.\n");
            }
        }
        else
        {
            PRINT_INPUT_ERROR("Invalid GKMA keyword.\n");
        }
        printf("    Use select atom range.\n"
               "    Atom types %d to %d.\n",
               gkma.atom_begin, gkma.atom_end);
    }
    else // default behavior
    {
        gkma.atom_begin = 0;
        gkma.atom_end = atom->number_of_types - 1;
    }

}


void Measure::parse_compute_hnema(char **param, int num_param, Atom* atom)
{
    hnema.compute = 1;

    printf("Compute modal thermal conductivity using HNEMA method.\n");

    /*
     * There is a hidden feature that allows for specification of atom
     * types to included (must be contiguously defined like potentials)
     * -- Works for types only, not groups --
     */

    if (num_param != 10 && num_param != 13)
    {
        PRINT_INPUT_ERROR("compute_hnema should have 9 parameters.\n");
    }
    if (!is_valid_int(param[1], &hnema.sample_interval) ||
        !is_valid_int(param[2], &hnema.output_interval) ||
        !is_valid_int(param[6], &hnema.first_mode)      ||
        !is_valid_int(param[7], &hnema.last_mode)       )
    {
        PRINT_INPUT_ERROR("A parameter for HNEMA should be an integer.\n");
    }

    // HNEMD driving force parameters -> Use HNEMD object
    if (!is_valid_real(param[3], &hnemd.fe_x))
    {
        PRINT_INPUT_ERROR("fe_x for HNEMD should be a real number.\n");
    }
    printf("    fe_x = %g /A\n", hnemd.fe_x);
    if (!is_valid_real(param[4], &hnemd.fe_y))
    {
        PRINT_INPUT_ERROR("fe_y for HNEMD should be a real number.\n");
    }
    printf("    fe_y = %g /A\n", hnemd.fe_y);
    if (!is_valid_real(param[5], &hnemd.fe_z))
    {
        PRINT_INPUT_ERROR("fe_z for HNEMD should be a real number.\n");
    }
    printf("    fe_z = %g /A\n", hnemd.fe_z);
    // magnitude of the vector
    hnemd.fe  = hnemd.fe_x * hnemd.fe_x;
    hnemd.fe += hnemd.fe_y * hnemd.fe_y;
    hnemd.fe += hnemd.fe_z * hnemd.fe_z;
    hnemd.fe  = sqrt(hnemd.fe);


    if (strcmp(param[8], "bin_size") == 0)
    {
        hnema.f_flag = 0;
        if(!is_valid_int(param[9], &hnema.bin_size))
        {
            PRINT_INPUT_ERROR("HNEMA bin_size must be an integer.\n");
        }
    }
    else if (strcmp(param[8], "f_bin_size") == 0)
    {
        hnema.f_flag = 1;
        if(!is_valid_real(param[9], &hnema.f_bin_size))
        {
            PRINT_INPUT_ERROR("HNEMA f_bin_size must be a real number.\n");
        }
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid binning keyword for compute_hnema.\n");
    }

    HNEMA *h = &hnema;
    // Parameter checking
    if (h->sample_interval < 1  || h->output_interval < 1 ||
            h->first_mode < 1 || h->last_mode < 1)
        PRINT_INPUT_ERROR("compute_hnema parameters must be positive integers.\n");
    if (h->first_mode > h->last_mode)
        PRINT_INPUT_ERROR("first_mode <= last_mode required.\n");
    if (h->output_interval % h->sample_interval != 0)
            PRINT_INPUT_ERROR("sample_interval must divide output_interval an integer\n"
                    " number of times.\n");

    printf("    sample_interval is %d.\n"
           "    output_interval is %d.\n"
           "    first_mode is %d.\n"
           "    last_mode is %d.\n",
          h->sample_interval, h->output_interval, h->first_mode, h->last_mode);

    if (h->f_flag)
    {
        if (h->f_bin_size <= 0.0)
        {
            PRINT_INPUT_ERROR("bin_size must be greater than zero.\n");
        }
        printf("    Bin by frequency.\n"
               "    f_bin_size is %f THz.\n", h->f_bin_size);
    }
    else
    {
        if (h->bin_size < 1)
        {
            PRINT_INPUT_ERROR("compute_hnema parameters must be positive integers.\n");
        }
        int num_modes = h->last_mode - h->first_mode + 1;
        if (num_modes % h->bin_size != 0)
            PRINT_INPUT_ERROR("number of modes must be divisible by bin_size.\n");
        printf("    Bin by modes.\n"
               "    bin_size is %d THz.\n", h->bin_size);
    }

    // Hidden feature implementation
    if (num_param == 13)
    {
        if (strcmp(param[10], "atom_range") == 0)
        {
            if(!is_valid_int(param[11], &hnema.atom_begin) ||
               !is_valid_int(param[12], &hnema.atom_end))
            {
                PRINT_INPUT_ERROR("HNEMA atom_begin & atom_end must be integers.\n");
            }
            if (hnema.atom_begin > hnema.atom_end)
            {
                PRINT_INPUT_ERROR("atom_begin must be less than atom_end.\n");
            }
            if (hnema.atom_begin < 0)
            {
                PRINT_INPUT_ERROR("atom_begin must be greater than 0.\n");
            }
            if (hnema.atom_end >= atom->number_of_types)
            {
                PRINT_INPUT_ERROR("atom_end must be greater than 0.\n");
            }
        }
        else
        {
            PRINT_INPUT_ERROR("Invalid HNEMA keyword.\n");
        }
        printf("    Use select atom range.\n"
               "    Atom types %d to %d.\n",
               hnema.atom_begin, hnema.atom_end);
    }
    else // default behavior
    {
        hnema.atom_begin = 0;
        hnema.atom_end = atom->number_of_types - 1;
    }

}


void Measure::parse_compute_hnemd(char **param, int num_param)
{
    hnemd.compute = 1;

    printf("Compute thermal conductivity using the HNEMD method.\n");

    if (num_param != 5)
    {
        PRINT_INPUT_ERROR("compute_hnemd should have 4 parameters.\n");
    }

    if (!is_valid_int(param[1], &hnemd.output_interval))
    {
        PRINT_INPUT_ERROR("output_interval for HNEMD should be an integer number.\n");
    }
    printf("    output_interval = %d\n", hnemd.output_interval);
    if (hnemd.output_interval < 1)
    {
        PRINT_INPUT_ERROR("output_interval for HNEMD should be larger than 0.\n");
    }
    if (!is_valid_real(param[2], &hnemd.fe_x))
    {
        PRINT_INPUT_ERROR("fe_x for HNEMD should be a real number.\n");
    }
    printf("    fe_x = %g /A\n", hnemd.fe_x);
    if (!is_valid_real(param[3], &hnemd.fe_y))
    {
        PRINT_INPUT_ERROR("fe_y for HNEMD should be a real number.\n");
    }
    printf("    fe_y = %g /A\n", hnemd.fe_y);
    if (!is_valid_real(param[4], &hnemd.fe_z))
    {
        PRINT_INPUT_ERROR("fe_z for HNEMD should be a real number.\n");
    }
    printf("    fe_z = %g /A\n", hnemd.fe_z);

    // magnitude of the vector
    hnemd.fe  = hnemd.fe_x * hnemd.fe_x;
    hnemd.fe += hnemd.fe_y * hnemd.fe_y;
    hnemd.fe += hnemd.fe_z * hnemd.fe_z;
    hnemd.fe  = sqrt(hnemd.fe);
}


void Measure::parse_compute_shc(char **param, int num_param)
{
    printf("Compute SHC.\n");
    shc.compute = 1;

    // check the number of parameters
    if ((num_param != 4) && (num_param != 5) && (num_param != 6))
    {
        PRINT_INPUT_ERROR("compute_shc should have 3 or 4 or 5 parameters.\n");
    }

    // group method and group id
    int offset = 0;
    if (num_param == 4)
    {
        shc.group_method = -1;
        printf("    for the whole system.\n");
    }
    else if (num_param == 5)
    {
        offset = 1;
        shc.group_method = 0;
        if (!is_valid_int(param[1], &shc.group_id))
        {
            PRINT_INPUT_ERROR("grouping id should be an integer.\n");
        }
        printf("    for atoms in group %d.\n", shc.group_id);
        printf("    using group method 0.\n");
    }
    else
    {
        offset = 2;
        if (!is_valid_int(param[1], &shc.group_method))
        {
            PRINT_INPUT_ERROR("group method should be an integer.\n");
        }
        if (!is_valid_int(param[2], &shc.group_id))
        {
            PRINT_INPUT_ERROR("grouping id should be an integer.\n");
        }
        printf("    for atoms in group %d.\n", shc.group_id);
        printf("    using group method %d.\n", shc.group_method);
    }

    // sample interval 
    if (!is_valid_int(param[1+offset], &shc.sample_interval))
    {
        PRINT_INPUT_ERROR("shc.sample_interval should be an integer.\n");
    }
    printf
    ("    sample interval for SHC is %d.\n", shc.sample_interval);

    // number of correlation data
    if (!is_valid_int(param[2+offset], &shc.Nc))
    {
        PRINT_INPUT_ERROR("Nc for SHC should be an integer.\n");
    }
    printf("    number of correlation data is %d.\n", shc.Nc);

    // transport direction
    if (!is_valid_int(param[3+offset], &shc.direction))
    {
        PRINT_INPUT_ERROR("direction for SHC should be an integer.\n");
    }
    printf("    transport direction is %d.\n", shc.direction);
}


void Measure::parse_compute(char **param, int num_param)
{
    printf("Compute group average of:\n");
    if (num_param < 5)
        PRINT_INPUT_ERROR("compute should have at least 4 parameters.\n");
    if (!is_valid_int(param[1], &compute.grouping_method))
    {
        PRINT_INPUT_ERROR("grouping method of compute should be integer.\n");
    }
    if (!is_valid_int(param[2], &compute.sample_interval))
    {
        PRINT_INPUT_ERROR("sampling interval of compute should be integer.\n");
    }
    if (!is_valid_int(param[3], &compute.output_interval))
    {
        PRINT_INPUT_ERROR("output interval of compute should be integer.\n");
    }
    for (int k = 0; k < num_param - 4; ++k)
    {
        if (strcmp(param[k + 4], "temperature") == 0)
        {
            compute.compute_temperature = 1;
            printf("    temperature\n");
        }
        else if (strcmp(param[k + 4], "potential") == 0)
        {
            compute.compute_potential = 1;
            printf("    potential energy\n");
        }
        else if (strcmp(param[k + 4], "force") == 0)
        {
            compute.compute_force = 1;
            printf("    force\n");
        }
        else if (strcmp(param[k + 4], "virial") == 0)
        {
            compute.compute_virial = 1;
            printf("    virial\n");
        }
        else if (strcmp(param[k + 4], "jp") == 0)
        {
            compute.compute_jp = 1;
            printf("    potential part of heat current\n");
        }
        else if (strcmp(param[k + 4], "jk") == 0)
        {
            compute.compute_jk = 1;
            printf("    kinetic part of heat current\n");
        }
    }
    printf("    using grouping method %d.\n",
        compute.grouping_method);
    printf("    with sampling interval %d.\n",
        compute.sample_interval);
    printf("    and output interval %d.\n",
        compute.output_interval);
}


