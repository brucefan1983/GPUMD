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
#include "atom.cuh"
#include "error.cuh"
#include "read_file.cuh"
#include "dump_xyz.cuh"
#ifdef USE_NETCDF
#include "dump_netcdf.cuh"
#endif
#define NUM_OF_HEAT_COMPONENTS 5


Measure::Measure(char *input_dir)
{
    dump_thermo = 0;
    dump_velocity = 0;
    dump_restart = 0;
    dump_pos = NULL; // to avoid deleting random memory in run
    strcpy(file_thermo, input_dir);
    strcpy(file_velocity, input_dir);
    strcpy(file_restart, input_dir);
    strcat(file_thermo, "/thermo.out");
    strcat(file_velocity, "/velocity.out");
    strcat(file_restart, "/restart.out");
}


Measure::~Measure(void)
{
    // nothing
}


void Measure::initialize(char* input_dir, Atom *atom)
{
    if (dump_thermo)    {fid_thermo   = my_fopen(file_thermo,   "a");}
    if (dump_velocity)  {fid_velocity = my_fopen(file_velocity, "a");}
    if (dump_pos)       {dump_pos->initialize(input_dir);}
    vac.preprocess(atom->time_step, atom->group, atom->mass);
    hac.preprocess(atom->number_of_steps);
    shc.preprocess(atom->N, atom->group);
    compute.preprocess(atom->N, input_dir, atom->group);
    hnemd.preprocess();
    modal_analysis.preprocess(input_dir, atom);
}


void Measure::finalize
(
    char *input_dir,
    Atom *atom,
    const double temperature
)
{
    if (dump_thermo)    {fclose(fid_thermo);    dump_thermo    = 0;}
    if (dump_velocity)  {fclose(fid_velocity);  dump_velocity  = 0;}
    if (dump_restart)   {                       dump_restart   = 0;}
    if (dump_pos)       {dump_pos->finalize();}
    vac.postprocess(input_dir);
    hac.postprocess
    (
        atom->number_of_steps,
        input_dir,
        temperature,
        atom->time_step,
        atom->box.get_volume()
    );
    shc.postprocess(input_dir);
    compute.postprocess();
    hnemd.postprocess();
    modal_analysis.postprocess();
}


void Measure::dump_thermos
(
    FILE *fid,
    const int step,
    const int number_of_atoms,
    const int number_of_atoms_fixed,
    GPU_Vector<double>& gpu_thermo,
    const Box& box
)
{
    if (!dump_thermo) return;
    if ((step + 1) % sample_interval_thermo != 0) return;

    std::vector<double> thermo(5);
    gpu_thermo.copy_to_host(thermo.data(), 5);

    const int number_of_atoms_moving = number_of_atoms- number_of_atoms_fixed;
    double energy_kin = 1.5 * number_of_atoms_moving * K_B * thermo[0];

    fprintf
    (
        fid,
        "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e",
        thermo[0],
        energy_kin,
        thermo[1],
        thermo[2] * PRESSURE_UNIT_CONVERSION,
        thermo[3] * PRESSURE_UNIT_CONVERSION,
        thermo[4] * PRESSURE_UNIT_CONVERSION
    );

    int number_of_box_variables = box.triclinic ? 9 : 3;
    for (int m = 0; m < number_of_box_variables; ++m)
    {
        fprintf(fid, "%20.10e", box.cpu_h[m]);
    }

    fprintf(fid, "\n");
    fflush(fid);
}


void Measure::dump_velocities
(
    FILE* fid,
    const int step,
    GPU_Vector<double>& velocity_per_atom,
    std::vector<double>& cpu_velocity_per_atom
)
{
    if (!dump_velocity) return;
    if ((step + 1) % sample_interval_velocity != 0) return;

    const int number_of_atoms = velocity_per_atom.size() / 3;

    velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());

    for (int n = 0; n < number_of_atoms; n++)
    {
        fprintf
        (
            fid, "%g %g %g\n", 
            cpu_velocity_per_atom[n],
            cpu_velocity_per_atom[n + number_of_atoms],
            cpu_velocity_per_atom[n + 2 * number_of_atoms]
        );
    }

    fflush(fid);
}


void Measure::dump_restarts
(
    const int step,
    const Neighbor& neighbor,
    const Box& box,
    const std::vector<Group>& group,
    const std::vector<int>& cpu_type,
    const std::vector<double>& cpu_mass,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    std::vector<double>& cpu_position_per_atom,
    std::vector<double>& cpu_velocity_per_atom
)
{
    if (!dump_restart) return;
    if ((step + 1) % sample_interval_restart != 0) return;

    const int number_of_atoms = cpu_mass.size();

    position_per_atom.copy_to_host(cpu_position_per_atom.data());
    velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());

    fid_restart = my_fopen(file_restart, "w"); 

    fprintf
    (
        fid_restart, "%d %d %g %d %d %d\n",
        number_of_atoms,
        neighbor.MN,
        neighbor.rc,
        box.triclinic,
        1,
        int(group.size())
    );

    if (box.triclinic == 0)
    {
        fprintf
        (
            fid_restart,
            "%d %d %d %g %g %g\n",
            box.pbc_x,
            box.pbc_y,
            box.pbc_z,
            box.cpu_h[0],
            box.cpu_h[1],
            box.cpu_h[2]
        );
    }
    else
    {
        fprintf
        (
            fid_restart,
            "%d %d %d %g %g %g %g %g %g %g %g %g\n",
            box.pbc_x,
            box.pbc_y,
            box.pbc_z,
            box.cpu_h[0],
            box.cpu_h[3],
            box.cpu_h[6],
            box.cpu_h[1],
            box.cpu_h[4],
            box.cpu_h[7],
            box.cpu_h[2],
            box.cpu_h[5],
            box.cpu_h[8]
        );
    }

    for (int n = 0; n < number_of_atoms; n++)
    {
        fprintf
        (
            fid_restart,
            "%d %g %g %g %g %g %g %g ",
            cpu_type[n],
            cpu_position_per_atom[n],
            cpu_position_per_atom[n + number_of_atoms],
            cpu_position_per_atom[n + 2 * number_of_atoms],
            cpu_mass[n],
            cpu_velocity_per_atom[n],
            cpu_velocity_per_atom[n + number_of_atoms],
            cpu_velocity_per_atom[n + 2 * number_of_atoms]
        );

        for (int m = 0; m < group.size(); ++m)
        {
            fprintf(fid_restart, "%d ", group[m].cpu_label[n]);
        }

        fprintf(fid_restart, "\n");
    }

    fflush(fid_restart);
    fclose(fid_restart);
}


void Measure::process
(
    char *input_dir,
    Atom *atom,
    const int fixed_group,
    const double temperature,
    const double energy_transferred[],
    int step
)
{
    dump_thermos
    (
        fid_thermo,
        step,
        atom->N,
        (fixed_group < 0) ? 0 : atom->group[0].cpu_size[fixed_group],
        atom->thermo,
        atom->box
    );

    dump_velocities
    (
        fid_velocity,
        step,
        atom->velocity_per_atom,
        atom->cpu_velocity_per_atom
    );

    dump_restarts
    (
        step,
        atom->neighbor,
        atom->box,
        atom->group,
        atom->cpu_type,
        atom->cpu_mass,
        atom->position_per_atom,
        atom->velocity_per_atom,
        atom->cpu_position_per_atom,
        atom->cpu_velocity_per_atom
    );

    compute.process
    (
        step,
        energy_transferred,
        atom->group,
        atom->mass,
        atom->potential_per_atom,
        atom->force_per_atom,
        atom->velocity_per_atom,
        atom->virial_per_atom
    );

    vac.process
    (
        step,
        atom->group,
        atom->velocity_per_atom
    );

    hac.process
    (
        atom->number_of_steps,
        step,
        input_dir,
        atom->velocity_per_atom,
        atom->virial_per_atom,
        atom->heat_per_atom
    );

    shc.process
    (
        step,
        atom->group,
        atom->velocity_per_atom,
        atom->virial_per_atom
    );

    hnemd.process
    (
        step,
        input_dir,
        temperature,
        atom->box.get_volume(),
        atom->velocity_per_atom,
        atom->virial_per_atom,
        atom->heat_per_atom
    );

    modal_analysis.process(step, atom, temperature, hnemd.fe);
    if (dump_pos) dump_pos->dump(atom, step);
}


void Measure::parse_dump_thermo(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_thermo should have 1 parameter.");
    }
    if (!is_valid_int(param[1], &sample_interval_thermo))
    {
        PRINT_INPUT_ERROR("thermo dump interval should be an integer.");
    }
    if (0 >= sample_interval_thermo)
    {
        PRINT_INPUT_ERROR("thermo dump interval should > 0.");
    }

    dump_thermo = 1;
    printf("Dump thermo every %d steps.\n", sample_interval_thermo);
}


void Measure::parse_dump_velocity(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_velocity should have 1 parameter.");
    }
    if (!is_valid_int(param[1], &sample_interval_velocity))
    {
        PRINT_INPUT_ERROR("velocity dump interval should be an integer.");
    }
    if (0 >= sample_interval_velocity)
    {
        PRINT_INPUT_ERROR("velocity dump interval should > 0.");
    }

    dump_velocity = 1;
    printf("Dump velocity every %d steps.\n", sample_interval_velocity);
}


void Measure::parse_dump_position(char **param, int num_param, Atom *atom)
{
    int interval;

    if (num_param < 2)
    {
        PRINT_INPUT_ERROR("dump_position should have at least 1 parameter.");
    }
    if (num_param > 6)
    {
        PRINT_INPUT_ERROR("dump_position has too many parameters.");
    }

    // sample interval
    if (!is_valid_int(param[1], &interval))
    {
        PRINT_INPUT_ERROR("position dump interval should be an integer.");
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

    printf("Dump position every %d steps.\n", dump_pos->interval);
}


void Measure::parse_dump_restart(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("dump_restart should have 1 parameter.");
    }
    if (!is_valid_int(param[1], &sample_interval_restart))
    {
        PRINT_INPUT_ERROR("restart dump interval should be an integer.");
    }
    if (0 >= sample_interval_restart)
    {
        PRINT_INPUT_ERROR("restart dump interval should > 0.");
    }

    dump_restart = 1;
    printf("Dump restart every %d steps.\n", sample_interval_restart);
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
    if (!is_valid_int(param[*k+1], &vac.num_dos_points))
    {
        PRINT_INPUT_ERROR("number of DOS points for VAC should be an integer "
            "number.\n");
    }
    if (vac.num_dos_points < 1)
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
    if (!is_valid_real(param[3], &vac.omega_max))
    {
        PRINT_INPUT_ERROR("omega_max should be a real number.\n");
    }
    if (vac.omega_max <= 0)
    {
        PRINT_INPUT_ERROR("omega_max should be positive.\n");
    }
    printf("    omega_max is %g THz.\n", vac.omega_max);

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
            printf("    num_dos_points is %d.\n", vac.num_dos_points);
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
    modal_analysis.compute = 1;
    if (modal_analysis.method == GKMA_METHOD)
    { // TODO add warning macro
        printf("*******************************************************"
                "WARNING: GKMA method already defined for this run.\n"
                "         Parameters will be overwritten\n"
                "*******************************************************");
    }
    else if (modal_analysis.method == HNEMA_METHOD)
    {
        printf("*******************************************************"
                "WARNING: HNEMA method already defined for this run.\n"
                "         GKMA will now run instead.\n"
                "*******************************************************");
    }
    modal_analysis.method = GKMA_METHOD;

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
    if (!is_valid_int(param[1], &modal_analysis.sample_interval) ||
        !is_valid_int(param[2], &modal_analysis.first_mode)      ||
        !is_valid_int(param[3], &modal_analysis.last_mode)       )
    {
        PRINT_INPUT_ERROR("A parameter for GKMA should be an integer.\n");
    }

    if (strcmp(param[4], "bin_size") == 0)
    {
        modal_analysis.f_flag = 0;
        if(!is_valid_int(param[5], &modal_analysis.bin_size))
        {
            PRINT_INPUT_ERROR("GKMA bin_size must be an integer.\n");
        }
    }
    else if (strcmp(param[4], "f_bin_size") == 0)
    {
        modal_analysis.f_flag = 1;
        if(!is_valid_real(param[5], &modal_analysis.f_bin_size))
        {
            PRINT_INPUT_ERROR("GKMA f_bin_size must be a real number.\n");
        }
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid binning keyword for compute_gkma.\n");
    }

    MODAL_ANALYSIS *g = &modal_analysis;
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
            if(!is_valid_int(param[7], &modal_analysis.atom_begin) ||
               !is_valid_int(param[8], &modal_analysis.atom_end))
            {
                PRINT_INPUT_ERROR("GKMA atom_begin & atom_end must be integers.\n");
            }
            if (modal_analysis.atom_begin > modal_analysis.atom_end)
            {
                PRINT_INPUT_ERROR("atom_begin must be less than atom_end.\n");
            }
            if (modal_analysis.atom_begin < 0)
            {
                PRINT_INPUT_ERROR("atom_begin must be greater than 0.\n");
            }
            if (modal_analysis.atom_end >= atom->number_of_types)
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
               modal_analysis.atom_begin, modal_analysis.atom_end);
    }
    else // default behavior
    {
        modal_analysis.atom_begin = 0;
        modal_analysis.atom_end = atom->number_of_types - 1;
    }

}

void Measure::parse_compute_hnema(char **param, int num_param, Atom* atom)
{
    modal_analysis.compute = 1;
    if (modal_analysis.method == HNEMA_METHOD)
    {
        printf("*******************************************************\n"
                "WARNING: HNEMA method already defined for this run.\n"
                "         Parameters will be overwritten\n"
                "*******************************************************\n");
    }
    else if (modal_analysis.method == GKMA_METHOD)
    {
        printf("*******************************************************\n"
                "WARNING: GKMA method already defined for this run.\n"
                "         HNEMA will now run instead.\n"
                "*******************************************************\n");
    }
    modal_analysis.method = HNEMA_METHOD;

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
    if (!is_valid_int(param[1], &modal_analysis.sample_interval) ||
        !is_valid_int(param[2], &modal_analysis.output_interval) ||
        !is_valid_int(param[6], &modal_analysis.first_mode)      ||
        !is_valid_int(param[7], &modal_analysis.last_mode)       )
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
        modal_analysis.f_flag = 0;
        if(!is_valid_int(param[9], &modal_analysis.bin_size))
        {
            PRINT_INPUT_ERROR("HNEMA bin_size must be an integer.\n");
        }
    }
    else if (strcmp(param[8], "f_bin_size") == 0)
    {
        modal_analysis.f_flag = 1;
        if(!is_valid_real(param[9], &modal_analysis.f_bin_size))
        {
            PRINT_INPUT_ERROR("HNEMA f_bin_size must be a real number.\n");
        }
    }
    else
    {
        PRINT_INPUT_ERROR("Invalid binning keyword for compute_hnema.\n");
    }

    MODAL_ANALYSIS *h = &modal_analysis;
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
        printf("    Bin by modes.\n"
               "    bin_size is %d modes.\n", h->bin_size);
    }

    // Hidden feature implementation
    if (num_param == 13)
    {
        if (strcmp(param[10], "atom_range") == 0)
        {
            if(!is_valid_int(param[11], &modal_analysis.atom_begin) ||
               !is_valid_int(param[12], &modal_analysis.atom_end))
            {
                PRINT_INPUT_ERROR("HNEMA atom_begin & atom_end must be integers.\n");
            }
            if (modal_analysis.atom_begin > modal_analysis.atom_end)
            {
                PRINT_INPUT_ERROR("atom_begin must be less than atom_end.\n");
            }
            if (modal_analysis.atom_begin < 0)
            {
                PRINT_INPUT_ERROR("atom_begin must be greater than 0.\n");
            }
            if (modal_analysis.atom_end >= atom->number_of_types)
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
               modal_analysis.atom_begin, modal_analysis.atom_end);
    }
    else // default behavior
    {
        modal_analysis.atom_begin = 0;
        modal_analysis.atom_end = atom->number_of_types - 1;
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


void Measure::parse_compute_shc(char **param, int num_param, Atom *atom)
{
    printf("Compute SHC.\n");
    shc.compute = 1;

    // check the number of parameters
    if ((num_param != 4) && (num_param != 5) && (num_param != 6))
    {
        PRINT_INPUT_ERROR("compute_shc should have 3 or 4 or 5 parameters.");
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
            PRINT_INPUT_ERROR("group id should be an integer.");
        }
        if (shc.group_id < 0)
        {
            PRINT_INPUT_ERROR("group id should >= 0.");
        }
        if (shc.group_id >= atom->group[0].number)
        {
            PRINT_INPUT_ERROR("group id should < #groups.");
        }
        printf("    for atoms in group %d.\n", shc.group_id);
        printf("    using grouping method 0.\n");
    }
    else
    {
        offset = 2;
        // grouping method
        if (!is_valid_int(param[1], &shc.group_method))
        {
            PRINT_INPUT_ERROR("grouping method should be an integer.");
        }
        if (shc.group_method < 0)
        {
            PRINT_INPUT_ERROR("grouping method should >= 0.");
        }
        if (shc.group_method >= atom->group.size())
        {
            PRINT_INPUT_ERROR("grouping method exceeds the bound.");
        }

        // group id
        if (!is_valid_int(param[2], &shc.group_id))
        {
            PRINT_INPUT_ERROR("group id should be an integer.");
        }
        if (shc.group_id < 0)
        {
            PRINT_INPUT_ERROR("group id should >= 0.");
        }
        if (shc.group_id >= atom->group[shc.group_method].number)
        {
            PRINT_INPUT_ERROR("group id should < #groups.");
        }
        printf("    for atoms in group %d.\n", shc.group_id);
        printf("    using group method %d.\n", shc.group_method);
    }

    // sample interval 
    if (!is_valid_int(param[1 + offset], &shc.sample_interval))
    {
        PRINT_INPUT_ERROR("Sampling interval for SHC should be an integer.");
    }
    if (shc.sample_interval < 1)
    {
        PRINT_INPUT_ERROR("Sampling interval for SHC should >= 1.");
    }
    if (shc.sample_interval > 10)
    {
        PRINT_INPUT_ERROR("Sampling interval for SHC should <= 10 (trust me).");
    }
    printf("    sampling interval for SHC is %d.\n", shc.sample_interval);

    // number of correlation data
    if (!is_valid_int(param[2 + offset], &shc.Nc))
    {
        PRINT_INPUT_ERROR("Nc for SHC should be an integer.");
    }
    if (shc.Nc < 100)
    {
        PRINT_INPUT_ERROR("Nc for SHC should >= 100 (trust me).");
    }
    if (shc.Nc > 1000)
    {
        PRINT_INPUT_ERROR("Nc for SHC should <= 1000 (trust me).");
    }
    printf("    number of correlation data is %d.\n", shc.Nc);

    // transport direction
    if (!is_valid_int(param[3 + offset], &shc.direction))
    {
        PRINT_INPUT_ERROR("direction for SHC should be an integer.");
    }
    if (shc.direction == 0)
    {
        printf("    transport direction is x.\n");
    }
    else if (shc.direction == 1)
    {
        printf("    transport direction is y.\n");
    }
    else if (shc.direction == 2)
    {
        printf("    transport direction is z.\n");
    }
    else
    {
        PRINT_INPUT_ERROR("Transport direction should be x or y or z.");
    }
    
}


void Measure::parse_compute(char **param, int num_param, Atom *atom)
{
    printf("Compute space and/or time average of:\n");
    if (num_param < 5)
    {
        PRINT_INPUT_ERROR("compute should have at least 4 parameters.");
    }

    // grouping_method
    if (!is_valid_int(param[1], &compute.grouping_method))
    {
        PRINT_INPUT_ERROR("grouping method of compute should be integer.");
    }
    if (compute.grouping_method < 0)
    {
        PRINT_INPUT_ERROR("grouping method should >= 0.");
    }
    if (compute.grouping_method >= atom->group.size())
    {
        PRINT_INPUT_ERROR("grouping method exceeds the bound.");
    }

    // sample_interval
    if (!is_valid_int(param[2], &compute.sample_interval))
    {
        PRINT_INPUT_ERROR("sampling interval of compute should be integer.");
    }
    if (compute.sample_interval <= 0)
    {
        PRINT_INPUT_ERROR("sampling interval of compute should > 0.");
    }

    // output_interval
    if (!is_valid_int(param[3], &compute.output_interval))
    {
        PRINT_INPUT_ERROR("output interval of compute should be integer.");
    }
    if (compute.output_interval <= 0)
    {
        PRINT_INPUT_ERROR("output interval of compute should > 0.");
    }

    // temperature potential force virial jp jk (order is not important)
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
        else
        {
            PRINT_INPUT_ERROR("Invalid property for compute.");
        }
    }

    printf("    using grouping method %d.\n", compute.grouping_method);
    printf("    with sampling interval %d.\n", compute.sample_interval);
    printf("    and output interval %d.\n", compute.output_interval);
}


