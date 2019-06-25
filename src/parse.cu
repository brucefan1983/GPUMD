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
Parse the commands in run.in.
------------------------------------------------------------------------------*/


#include "run.cuh"
#include "atom.cuh"
#include "ensemble.cuh"
#include "error.cuh"
#include "force.cuh"
#include "integrate.cuh"
#include "measure.cuh"
#include "hessian.cuh"
#include "read_file.cuh"
#include "dump_xyz.cuh"

#ifdef USE_NETCDF
#include "dump_netcdf.cuh"
#endif

// a single potential
void parse_potential(char **param, int num_param, Force *force)
{
    if (num_param != 2)
    {
        print_error("potential should have 1 parameter.\n");
    }
    strcpy(force->file_potential[0], param[1]);
    force->num_of_potentials = 1;
}


// multiple potentials
void parse_potentials(char **param, int num_param, Force *force)
{
    if (num_param == 6)
    {
        force->num_of_potentials = 2;
    }
    else if (num_param == 9)
    {
        force->num_of_potentials = 3;
    }
    else
    {
        print_error("potentials should have 5 or 8 parameters.\n");
    }

    // two-body part
    strcpy(force->file_potential[0], param[1]);
    if (!is_valid_int(param[2], &force->interlayer_only))
    {
        print_error("interlayer_only should be an integer.\n");
    }

    // the first many-body part
    strcpy(force->file_potential[1], param[3]);
    if (!is_valid_int(param[4], &force->type_begin[1]))
    {
        print_error("type_begin should be an integer.\n");
    }
    if (!is_valid_int(param[5], &force->type_end[1]))
    {
        print_error("type_end should be an integer.\n");
    }

    // the second many-body part
    if (force->num_of_potentials > 2)
    {
        strcpy(force->file_potential[2], param[6]);
        if (!is_valid_int(param[7], &force->type_begin[2]))
        {
            print_error("type_begin should be an integer.\n");
        }
        if (!is_valid_int(param[8], &force->type_end[2]))
        {
            print_error("type_end should be an integer.\n");
        }
    }
}


void parse_velocity(char **param, int num_param, Atom *atom)
{
    if (num_param != 2)
    {
        print_error("velocity should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->initial_temperature))
    {
        print_error("initial temperature should be a real number.\n");
    }
    if (atom->initial_temperature <= 0.0)
    {
        print_error("initial temperature should be a positive number.\n");
    }
}


// coding conventions:
//0:     NVE
//1-10:  NVT
//11-20: NPT
//21-30: heat (NEMD method for heat conductivity)
void parse_ensemble 
(char **param,  int num_param, Atom *atom, Integrate *integrate)
{
    // 1. Determine the integration method
    if (strcmp(param[1], "nve") == 0)
    {
        integrate->type = 0;
        if (num_param != 2)
        {
            print_error("ensemble nve should have 0 parameter.\n");
        }
    }
    else if (strcmp(param[1], "nvt_ber") == 0)
    {
        integrate->type = 1;
        if (num_param != 5)
        {
            print_error("ensemble nvt_ber should have 3 parameters.\n");
        }
    }
    else if (strcmp(param[1], "nvt_nhc") == 0)
    {
        integrate->type = 2;
        if (num_param != 5)
        {
            print_error("ensemble nvt_nhc should have 3 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "nvt_lan") == 0)
    {
        integrate->type = 3;
        if (num_param != 5)
        {
            print_error("ensemble nvt_lan should have 3 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "nvt_bdp") == 0)
    {
        integrate->type = 4;
        if (num_param != 5)
        {
            print_error("ensemble nvt_bdp should have 3 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "npt_ber") == 0)
    {
        integrate->type = 11;
        if (num_param != 9)
        {
            print_error("ensemble npt_ber should have 7 parameters.\n"); 
        } 
    }
    else if (strcmp(param[1], "heat_nhc") == 0)
    {
        integrate->type = 21;
        if (num_param != 7)
        {
            print_error("ensemble heat_nhc should have 5 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "heat_lan") == 0)
    {
        integrate->type = 22;
        if (num_param != 7)
        {
            print_error("ensemble heat_lan should have 5 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "heat_bdp") == 0)
    {
        integrate->type = 23;
        if (num_param != 7)
        {
            print_error("ensemble heat_bdp should have 5 parameters.\n"); 
        }
    }
    else
    {
        print_error("invalid ensemble type.\n");
    }

    // 2. Temperatures and temperature_coupling (NVT and NPT)
    if (integrate->type >= 1 && integrate->type <= 20)
    {
        // initial temperature
        if (!is_valid_real(param[2], &atom->temperature1))
        {
            print_error("ensemble temperature should be a real number.\n");
        }
        if (atom->temperature1 <= 0.0)
        {
            print_error("ensemble temperature should be a positive number.\n");
        }

        // final temperature
        if (!is_valid_real(param[3], &atom->temperature2))
        {
            print_error("ensemble temperature should be a real number.\n");
        }
        if (atom->temperature2 <= 0.0)
        {
            print_error("ensemble temperature should be a positive number.\n");
        }

        integrate->temperature = atom->temperature1;

        // temperature_coupling
        if (!is_valid_real(param[4], &integrate->temperature_coupling))
        {
            print_error("temperature_coupling should be a real number.\n");
        }
        if (integrate->temperature_coupling <= 0.0)
        {
            print_error("temperature_coupling should be a positive number.\n");
        }
    }

    // 3. Pressures and pressure_coupling (NPT)
    real pressure[3];
    if (integrate->type >= 11 && integrate->type <= 20)
    {  
        // pressures:   
        for (int i = 0; i < 3; i++)
        {
            if (!is_valid_real(param[5+i], &pressure[i]))
            {
                print_error("ensemble pressure should be a real number.\n");
            }
        }
        // Change the unit of pressure form GPa to that used in the code
        integrate->pressure_x = pressure[0] / PRESSURE_UNIT_CONVERSION;
        integrate->pressure_y = pressure[1] / PRESSURE_UNIT_CONVERSION;
        integrate->pressure_z = pressure[2] / PRESSURE_UNIT_CONVERSION;

        // pressure_coupling:
        if (!is_valid_real(param[8], &integrate->pressure_coupling))
        {
            print_error("pressure_coupling should be a real number.\n");
        }
        if (integrate->pressure_coupling <= 0.0)
        {
            print_error("pressure_coupling should be a positive number.\n");
        }
    }

    // 4. heating and cooling wiht fixed temperatures
    if (integrate->type >= 21 && integrate->type <= 30)
    {
        // temperature
        if (!is_valid_real(param[2], &integrate->temperature))
        {
            print_error("ensemble temperature should be a real number.\n");
        }
        if (integrate->temperature <= 0.0)
        {
            print_error("ensemble temperature should be a positive number.\n");
        }

        // temperature_coupling
        if (!is_valid_real(param[3], &integrate->temperature_coupling))
        {
            print_error("temperature_coupling should be a real number.\n");
        }
        if (integrate->temperature_coupling <= 0.0)
        {
            print_error("temperature_coupling should be a positive number.\n");
        }

        // temperature difference
        if (!is_valid_real(param[4], &integrate->delta_temperature))
        {
            print_error("delta_temperature should be a real number.\n");
        }

        // group labels of heat source and sink
        if (!is_valid_int(param[5], &integrate->source))
        {
            print_error("heat.source should be an integer.\n");
        }
        if (!is_valid_int(param[6], &integrate->sink))
        {
            print_error("heat.sink should be an integer.\n");
        }
    }

    switch (integrate->type)
    {
        case 0:
            printf("Use NVE ensemble for this run.\n");
            break;
        case 1:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Berendsen method.\n"); 
            printf("    initial temperature is %g K.\n", atom->temperature1);
            printf("    final temperature is %g K.\n", atom->temperature2);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 2:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Nose-Hoover chain method.\n"); 
            printf("    initial temperature is %g K.\n", atom->temperature1);
            printf("    final temperature is %g K.\n", atom->temperature2);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 3:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Langevin method.\n"); 
            printf("    initial temperature is %g K.\n", atom->temperature1);
            printf("    final temperature is %g K.\n", atom->temperature2);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 4:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Bussi-Donadio-Parrinello method.\n"); 
            printf("    initial temperature is %g K.\n", atom->temperature1);
            printf("    final temperature is %g K.\n", atom->temperature2);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 11:
            printf("Use NPT ensemble for this run.\n");
            printf("    choose the Berendsen method.\n");      
            printf("    initial temperature is %g K.\n", atom->temperature1);
            printf("    final temperature is %g K.\n", atom->temperature2);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            printf("    pressure_x is %g GPa.\n", pressure[0]);
            printf("    pressure_y is %g GPa.\n", pressure[1]);
            printf("    pressure_z is %g GPa.\n", pressure[2]);
            printf("    p_coupling is %g.\n", integrate->pressure_coupling);
            break;
        case 21:
            printf("Integrate with heating and cooling for this run.\n");
            printf("    choose the Nose-Hoover chain method.\n"); 
            printf("    temperature is %g K.\n", integrate->temperature);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            printf("    delta_T is %g K.\n", integrate->delta_temperature);
            printf("    heat source is group %d.\n", integrate->source);
            printf("    heat sink is group %d.\n", integrate->sink);
            break; 
        case 22:
            printf("Integrate with heating and cooling for this run.\n");
            printf("    choose the Langevin method.\n"); 
            printf("    temperature is %g K.\n", integrate->temperature);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            printf("    delta_T is %g K.\n", integrate->delta_temperature);
            printf("    heat source is group %d.\n", integrate->source);
            printf("    heat sink is group %d.\n", integrate->sink);
            break;
        case 23:
            printf("Integrate with heating and cooling for this run.\n");
            printf("    choose the Bussi-Donadio-Parrinello method.\n"); 
            printf("    temperature is %g K.\n", integrate->temperature);
            printf("    T_coupling is %g.\n", integrate->temperature_coupling);
            printf("    delta_T is %g K.\n", integrate->delta_temperature);
            printf("    heat source is group %d.\n", integrate->source);
            printf("    heat sink is group %d.\n", integrate->sink);
            break;
        default:
            print_error("invalid ensemble type.\n");
            break; 
    }
}


void parse_time_step (char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        print_error("time_step should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->time_step))
    {
        print_error("time_step should be a real number.\n");
    }
    printf("Time step for this run is %g fs.\n", atom->time_step);
    atom->time_step /= TIME_UNIT_CONVERSION;
}


void parse_neighbor
(char **param,  int num_param, Atom* atom, Force *force)
{
    atom->neighbor.update = 1;

    if (num_param != 2)
    {
        print_error("neighbor should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->neighbor.skin))
    {
        print_error("neighbor list skin should be a number.\n");
    }
    printf("Build neighbor list with a skin of %g A.\n", atom->neighbor.skin);

    // change the cutoff
    atom->neighbor.rc = force->rc_max + atom->neighbor.skin;
}


void parse_dump_thermo(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_thermo should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_thermo))
    {
        print_error("thermo dump interval should be an integer number.\n");
    }
    measure->dump_thermo = 1;
    printf("Dump thermo every %d steps.\n", measure->sample_interval_thermo);
}


void parse_dump_position(char **param,  int num_param, Measure *measure,
		Atom *atom)
{
	int interval;
	int format = 0; // default xyz
	int precision = 0; // default normal (unlesss netCDF -> 64 bit)
	int precision_defined = 0;

    if (num_param < 2)
    {
        print_error("dump_position should have at least 1 parameter.\n");
    }
    if (num_param > 6)
    {
    	print_error("dump_position has too many parameters.\n");
    }

    // sample interval
    if (!is_valid_int(param[1], &interval))
    {
        print_error("position dump interval should be an integer number.\n");
    }

    // Process optional arguments
    for (int k = 2; k < num_param; k++)
    {
    	// format check
    	if (strcmp(param[k], "format") == 0)
    	{
    		// check if there are enough inputs
    		if (k + 2 > num_param)
    		{
    			print_error("Not enough arguments for optional "
    					" 'format' dump_position command.\n");
    		}
    		if ((strcmp(param[k+1], "xyz") != 0) &&
				(strcmp(param[k+1], "netcdf") != 0))
    		{
    			print_error("Invalid format for dump_position command.\n");
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
				print_error("Not enough arguments for optional "
						" 'precision' dump_position command.\n");
			}
    		if ((strcmp(param[k+1], "normal") != 0) &&
				(strcmp(param[k+1], "high") != 0))
			{
				print_error("Invalid precision for dump_position command.\n");
			}
			else
			{
				precision_defined = 1;
				if(strcmp(param[k+1], "high") == 0)
				{
					precision = 1;
				}
				k++;
			}
    	}
    }

    if (format == 1) // netcdf output
    {
    	DUMP_NETCDF *dump_netcdf = new DUMP_NETCDF(atom->N, atom->global_time);
    	measure->dump_pos = dump_netcdf;
    }
    else // xyz default output
    {
    	DUMP_XYZ *dump_xyz = new DUMP_XYZ(precision);
    	measure->dump_pos = dump_xyz;
    }
    measure->dump_pos->interval = interval;


    if (precision_defined && format)
    	printf("Note: netCDF output files only output double precision.\n");

    printf("Dump position every %d steps.\n",
        measure->dump_pos->interval);
}


void parse_dump_restart(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_restart should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_restart))
    {
        print_error("restart dump interval should be an integer number.\n");
    }
    measure->dump_restart = 1;
    printf("Dump restart every %d steps.\n", measure->sample_interval_restart);
}


void parse_dump_velocity(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_velocity should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_velocity))
    {
        print_error("velocity dump interval should be an integer number.\n");
    }
    measure->dump_velocity = 1;
    printf("Dump velocity every %d steps.\n",
        measure->sample_interval_velocity);
}


void parse_dump_force(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_force should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_force))
    {
        print_error("force dump interval should be an integer number.\n");
    }
    measure->dump_force = 1;
    printf("Dump force every %d steps.\n", measure->sample_interval_force);
}


void parse_dump_potential(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_potential should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_potential))
    {
        print_error("potential dump interval should be an integer number.\n");
    }
    measure->dump_potential = 1;
    printf("Dump potential every %d steps.\n",
        measure->sample_interval_potential);
}


void parse_dump_virial(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_virial should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_virial))
    {
        print_error("virial dump interval should be an integer number.\n");
    }
    measure->dump_virial = 1;
    printf("Dump virial every %d steps.\n",
        measure->sample_interval_virial);
}


void parse_dump_heat(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_heat should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_heat))
    {
        print_error("heat dump interval should be an integer number.\n");
    }
    measure->dump_heat = 1;
    printf("Dump heat every %d steps.\n", measure->sample_interval_heat);
}

// Helper functions for parse_compute_dos
void parse_group(char **param, Measure *measure, int *k, Group *group)
{
	// grouping_method
	if (!is_valid_int(param[*k+1], &measure->vac.grouping_method))
	{
		print_error("grouping method for VAC should be an integer number.\n");
	}
	if (measure->vac.grouping_method < 0 || measure->vac.grouping_method > 2)
	{
		print_error("grouping method for VAC should be 0 <= x <= 2.\n");
	}
	// group
	if (!is_valid_int(param[*k+2], &measure->vac.group))
	{
		print_error("group for VAC should be an integer number.\n");
	}
	if (measure->vac.group < 0 ||
			measure->vac.group > group[measure->vac.grouping_method].number)
	{
		print_error("group for VAC must be >= 0 and < number of groups.\n");
	}
	*k += 2; // update index for next command
}

void parse_num_dos_points(char **param, Measure *measure, int *k)
{
	// number of DOS points
	if (!is_valid_int(param[*k+1], &measure->dos.num_dos_points))
	{
		print_error("number of DOS points for VAC should be an integer "
				"number.\n");
	}
	if (measure->dos.num_dos_points < 1)
	{
		print_error("number of DOS points for DOS must be > 0.\n");
	}
	*k += 1; //
}

void parse_compute_dos(char **param,  int num_param, Measure *measure,
		Group *group)
{
    printf("Compute phonon DOS.\n");
    measure->vac.compute_dos = 1;

    if (num_param < 4)
    {
        print_error("compute_dos should have at least 3 parameters.\n");
    }
    if (num_param > 9)
	{
		print_error("compute_dos has too many parameters.\n");
	}

    // sample interval
    if (!is_valid_int(param[1], &measure->vac.sample_interval))
    {
        print_error("sample interval for VAC should be an integer number.\n");
    }
    if (measure->vac.sample_interval <= 0)
    {
        print_error("sample interval for VAC should be positive.\n");
    }
    printf("    sample interval is %d.\n", measure->vac.sample_interval);

    // number of correlation steps
    if (!is_valid_int(param[2], &measure->vac.Nc))
    {
        print_error("Nc for VAC should be an integer number.\n");
    }
    if (measure->vac.Nc <= 0)
    {
        print_error("Nc for VAC should be positive.\n");
    }
    printf("    Nc is %d.\n", measure->vac.Nc);

    // maximal omega
    if (!is_valid_real(param[3], &measure->dos.omega_max))
    {
        print_error("omega_max should be a real number.\n");
    }
    if (measure->dos.omega_max <= 0)
    {
        print_error("omega_max should be positive.\n");
    }
    printf("    omega_max is %g THz.\n", measure->dos.omega_max);

    // Process optional arguments
    for (int k = 4; k < num_param; k++)
    {
    	if (strcmp(param[k], "group") == 0)
    	{
    		// check if there are enough inputs
    		if (k + 3 > num_param)
    		{
    			print_error("Not enough arguments for optional "
    					"'group' DOS command.\n");
    		}
    		parse_group(param, measure,  &k, group);
    		printf("    grouping_method is %d and group is %d.\n",
    				measure->vac.grouping_method, measure->vac.group);
    	}
    	else if (strcmp(param[k], "num_dos_points") == 0)
    	{
    		// check if there are enough inputs
    		if (k + 2 > num_param)
    		{
    			print_error("Not enough arguments for optional "
						"'group' dos command.\n");
    		}
    		parse_num_dos_points(param, measure, &k);
    		printf("    num_dos_points is %d.\n",measure->dos.num_dos_points);
    	}
    	else
    	{
    		print_error("Unrecognized argument in compute_dos.\n");
    	}
    }
}

void parse_compute_sdc(char **param,  int num_param, Measure *measure,
		Group *group)
{
    printf("Compute SDC.\n");
    measure->vac.compute_sdc = 1;

    if (num_param < 3)
    {
        print_error("compute_sdc should have at least 2 parameters.\n");
    }
    if (num_param > 6)
    {
    	print_error("compute_sdc has too many parameters.\n");
    }

    // sample interval
    if (!is_valid_int(param[1], &measure->vac.sample_interval))
    {
        print_error("sample interval for VAC should be an integer number.\n");
    }
    if (measure->vac.sample_interval <= 0)
    {
        print_error("sample interval for VAC should be positive.\n");
    }
    printf("    sample interval is %d.\n", measure->vac.sample_interval);

    // number of correlation steps
    if (!is_valid_int(param[2], &measure->vac.Nc))
    {
        print_error("Nc for VAC should be an integer number.\n");
    }
    if (measure->vac.Nc <= 0)
    {
        print_error("Nc for VAC should be positive.\n");
    }
    printf("    Nc is %d.\n", measure->vac.Nc);

    // Process optional arguments
	for (int k = 3; k < num_param; k++)
	{
		if (strcmp(param[k], "group") == 0)
		{
			// check if there are enough inputs
			if (k + 3 > num_param)
			{
				print_error("Not enough arguments for optional "
						"'group' SDC command.\n");
			}
			parse_group(param, measure,  &k, group);
			printf("    grouping_method is %d and group is %d.\n",
					measure->vac.grouping_method, measure->vac.group);
		}
		else
		{
			print_error("Unrecognized argument in compute_sdc.\n");
		}
	}
}


void parse_compute_hac(char **param,  int num_param, Measure* measure)
{
    measure->hac.compute = 1;

    printf("Compute HAC.\n");

    if (num_param != 4)
    {
        print_error("compute_hac should have 3 parameters.\n");
    }

    if (!is_valid_int(param[1], &measure->hac.sample_interval))
    {
        print_error("sample interval for HAC should be an integer number.\n");
    }
    printf("    sample interval is %d.\n", measure->hac.sample_interval);

    if (!is_valid_int(param[2], &measure->hac.Nc))
    {
        print_error("Nc for HAC should be an integer number.\n");
    }
    printf("    Nc is %d\n", measure->hac.Nc);

    if (!is_valid_int(param[3], &measure->hac.output_interval))
    {
        print_error("output_interval for HAC should be an integer number.\n");
    }
    printf("    output_interval is %d\n", measure->hac.output_interval);
}


void parse_compute_hnemd(char **param, int num_param, Measure* measure)
{
    measure->hnemd.compute = 1;

    printf("Compute thermal conductivity using the HNEMD method.\n");

    if (num_param != 5)
    {
        print_error("compute_hnemd should have 4 parameters.\n");
    }

    if (!is_valid_int(param[1], &measure->hnemd.output_interval))
    {
        print_error("output_interval for HNEMD should be an integer number.\n");
    }
    printf("    output_interval = %d\n", measure->hnemd.output_interval);
    if (measure->hnemd.output_interval < 1)
    {
        print_error("output_interval for HNEMD should be larger than 0.\n");
    }
    if (!is_valid_real(param[2], &measure->hnemd.fe_x))
    {
        print_error("fe_x for HNEMD should be a real number.\n");
    }
    printf("    fe_x = %g /A\n", measure->hnemd.fe_x);
    if (!is_valid_real(param[3], &measure->hnemd.fe_y))
    {
        print_error("fe_y for HNEMD should be a real number.\n");
    }
    printf("    fe_y = %g /A\n", measure->hnemd.fe_y);
    if (!is_valid_real(param[4], &measure->hnemd.fe_z))
    {
        print_error("fe_z for HNEMD should be a real number.\n");
    }
    printf("    fe_z = %g /A\n", measure->hnemd.fe_z);

    // magnitude of the vector
    measure->hnemd.fe  = measure->hnemd.fe_x * measure->hnemd.fe_x;
    measure->hnemd.fe += measure->hnemd.fe_y * measure->hnemd.fe_y;
    measure->hnemd.fe += measure->hnemd.fe_z * measure->hnemd.fe_z;
    measure->hnemd.fe  = sqrt(measure->hnemd.fe);
}


void parse_compute_shc(char **param,  int num_param, Measure* measure)
{
    printf("Compute SHC.\n");
    measure->shc.compute = 1;

    if (num_param != 6)
    {
        print_error("compute_shc should have 5 parameters.\n");
    }

    // sample interval 
    if (!is_valid_int(param[1], &measure->shc.sample_interval))
    {
        print_error("shc.sample_interval should be an integer.\n");
    }
    printf
    ("    sample interval for SHC is %d.\n", measure->shc.sample_interval);

    // number of correlation data
    if (!is_valid_int(param[2], &measure->shc.Nc))
    {
        print_error("Nc for SHC should be an integer.\n");
    }
    printf("    number of correlation data is %d.\n", measure->shc.Nc);

    // number of time origins 
    if (!is_valid_int(param[3], &measure->shc.M))
    {
        print_error("M for SHC should be an integer.\n");
    }
    printf("    number of time origions is %d.\n", measure->shc.M);

    // block A 
    if (!is_valid_int(param[4], &measure->shc.block_A))
    {
        print_error("block_A for SHC should be an integer.\n");
    }
    printf("    record the heat flowing from group %d.\n",
        measure->shc.block_A);
    // block B 
    if (!is_valid_int(param[5], &measure->shc.block_B))
    {
        print_error("block_B for SHC should be an integer.\n");
    }
    printf("    record the heat flowing into group %d.\n",
        measure->shc.block_B);
}


void parse_deform(char **param,  int num_param, Integrate* integrate)
{
    printf("Deform the box.\n");

    if (num_param != 5)
    {
        print_error("deform should have 4 parameters.\n");
    }

    // strain rate
    if (!is_valid_real(param[1], &integrate->deform_rate))
    {
        print_error("defrom rate should be a number.\n");
    }
    printf("    strain rate is %g A / step.\n",
        integrate->deform_rate);

    // direction
    if (!is_valid_int(param[2], &integrate->deform_x))
    {
        print_error("deform_x should be integer.\n");
    }
    if (!is_valid_int(param[3], &integrate->deform_y))
    {
        print_error("deform_y should be integer.\n");
    }
    if (!is_valid_int(param[4], &integrate->deform_z))
    {
        print_error("deform_z should be integer.\n");
    }

    if (integrate->deform_x)
    {
        printf("    apply strain in x direction.\n");
    }
    if (integrate->deform_y)
    {
        printf("    apply strain in y direction.\n");
    }
    if (integrate->deform_z)
    {
        printf("    apply strain in z direction.\n");
    }
}


void parse_compute(char **param,  int num_param, Measure* measure)
{
    printf("Compute group average of:\n");
    if (num_param < 5)
        print_error("compute should have at least 4 parameters.\n");
    if (!is_valid_int(param[1], &measure->compute.grouping_method))
    {
        print_error("grouping method of compute should be integer.\n");
    }
    if (!is_valid_int(param[2], &measure->compute.sample_interval))
    {
        print_error("sampling interval of compute should be integer.\n");
    }
    if (!is_valid_int(param[3], &measure->compute.output_interval))
    {
        print_error("output interval of compute should be integer.\n");
    }
    for (int k = 0; k < num_param - 4; ++k)
    {
        if (strcmp(param[k + 4], "temperature") == 0)
        {
            measure->compute.compute_temperature = 1;
            printf("    temperature\n");
        }
        else if (strcmp(param[k + 4], "potential") == 0)
        {
            measure->compute.compute_potential = 1;
            printf("    potential energy\n");
        }
        else if (strcmp(param[k + 4], "force") == 0)
        {
            measure->compute.compute_force = 1;
            printf("    force\n");
        }
        else if (strcmp(param[k + 4], "virial") == 0)
        {
            measure->compute.compute_virial = 1;
            printf("    virial\n");
        }
        else if (strcmp(param[k + 4], "jp") == 0)
        {
            measure->compute.compute_jp = 1;
            printf("    potential part of heat current\n");
        }
        else if (strcmp(param[k + 4], "jk") == 0)
        {
            measure->compute.compute_jk = 1;
            printf("    kinetic part of heat current\n");
        }
    }
    printf("    using grouping method %d.\n",
        measure->compute.grouping_method);
    printf("    with sampling interval %d.\n",
        measure->compute.sample_interval);
    printf("    and output interval %d.\n",
        measure->compute.output_interval);
}


void parse_fix(char **param, int num_param, Atom *atom)
{
    if (num_param != 2)
    {
        print_error("fix should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &atom->fixed_group))
    {
        print_error("fixed_group should be an integer.\n");
    }
    printf("Group %d will be fixed.\n", atom->fixed_group);
}


void parse_run(char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        print_error("run should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &atom->number_of_steps))
    {
        print_error("number of steps should be an integer.\n");
    }
    printf("Run %d steps.\n", atom->number_of_steps);
}


void parse_cutoff(char **param, int num_param, Hessian* hessian)
{
    if (num_param != 2)
    {
        print_error("cutoff should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &hessian->cutoff))
    {
        print_error("cutoff for hessian should be a number.\n");
    }
    if (hessian->cutoff <= 0)
    {
        print_error("cutoff for hessian should be positive.\n");
    }
    printf("Cutoff distance for hessian = %g A.\n", hessian->cutoff);
}


void parse_delta(char **param, int num_param, Hessian* hessian)
{
    if (num_param != 2)
    {
        print_error("compute_hessian should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &hessian->dx))
    {
        print_error("displacement for hessian should be a number.\n");
    }
    if (hessian->dx <= 0)
    {
        print_error("displacement for hessian should be positive.\n");
    }
    printf("Displacement for hessian = %g A.\n", hessian->dx);
}


