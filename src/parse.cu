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



#include "parse.cuh"

#include "atom.cuh"
#include "ensemble.cuh"
#include "error.cuh"
#include "force.cuh"
#include "integrate.cuh"
#include "measure.cuh"


#include <errno.h>

#ifdef USE_DP
    #define PRESSURE_UNIT_CONVERSION 1.602177e+2
    #define TIME_UNIT_CONVERSION     1.018051e+1
#else
    #define PRESSURE_UNIT_CONVERSION 1.602177e+2f
    #define TIME_UNIT_CONVERSION     1.018051e+1f
#endif




static int is_valid_int (const char *s, int *result)
{
    if (s == NULL || *s == '\0')
        return 0;

    char *p;
    errno = 0;
    *result = (int) strtol (s, &p, 0);

    if (errno != 0 || s == p || *p != 0)
        return 0;
    else
        return 1;
}




static int is_valid_real (const char *s, real *result)
{
    if (s == NULL || *s == '\0')
        return 0;

    char *p;
    errno = 0;
    *result = strtod (s, &p);

    if (errno != 0 || s == p || *p != 0)
        return 0;
    else
        return 1;
}




// a single potential
static void parse_potential(char **param, int num_param, Force *force)
{
    if (force->num_of_potentials != 0)
    {
        print_error("cannot have both 'potential' and 'potentials'.\n");
    }
    if (num_param != 2)
    {
        print_error("potential should have 1 parameter.\n");
    }
    strcpy(force->file_potential[0], param[1]);
    force->num_of_potentials = 1;
    printf("INPUT: use a single potential.\n");
}




// multiple potentials
static void parse_potentials(char **param, int num_param, Force *force)
{ 
    if (force->num_of_potentials != 0)
    {
        print_error("cannot have both 'potential' and 'potentials'.\n");
    }
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
    printf("INPUT: use %d potentials.\n", force->num_of_potentials);

    // two-body part
    strcpy(force->file_potential[0], param[1]);
    if (!is_valid_int(param[2], &force->interlayer_only))
    {
        print_error("interlayer_only should be an integer.\n");
    }
    if (force->interlayer_only == 0)
    {
        printf("INPUT: the 2-body part includes intralayer interactions.\n");
    }
    else
    {
        printf("INPUT: the 2-body part excludes intralayer interactions.\n");
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




static void parse_velocity(char **param, int num_param, Atom *atom)
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
    printf("INPUT: initial temperature is %g K.\n", atom->initial_temperature);
}




// coding conventions:
//0:     NVE
//1-10:  NVT
//11-20: NPT
//21-30: heat (NEMD method for heat conductivity)
static void parse_ensemble 
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
            printf("INPUT: Use NVE ensemble for this run.\n");
            break;
        case 1:
            printf("INPUT: Use NVT ensemble for this run.\n");
            printf("       choose the Berendsen method.\n"); 
            printf("       initial temperature is %g K.\n", atom->temperature1);
            printf("       final temperature is %g K.\n", atom->temperature2);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 2:
            printf("INPUT: Use NVT ensemble for this run.\n");
            printf("       choose the Nose-Hoover chain method.\n"); 
            printf("       initial temperature is %g K.\n", atom->temperature1);
            printf("       final temperature is %g K.\n", atom->temperature2);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 3:
            printf("INPUT: Use NVT ensemble for this run.\n");
            printf("       choose the Langevin method.\n"); 
            printf("       initial temperature is %g K.\n", atom->temperature1);
            printf("       final temperature is %g K.\n", atom->temperature2);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 4:
            printf("INPUT: Use NVT ensemble for this run.\n");
            printf("       choose the Bussi-Donadio-Parrinello method.\n"); 
            printf("       initial temperature is %g K.\n", atom->temperature1);
            printf("       final temperature is %g K.\n", atom->temperature2);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            break;
        case 11:
            printf("INPUT: Use NPT ensemble for this run.\n");
            printf("       choose the Berendsen method.\n");      
            printf("       initial temperature is %g K.\n", atom->temperature1);
            printf("       final temperature is %g K.\n", atom->temperature2);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            printf("       pressure_x is %g GPa.\n", pressure[0]);
            printf("       pressure_y is %g GPa.\n", pressure[1]);
            printf("       pressure_z is %g GPa.\n", pressure[2]);
            printf("       p_coupling is %g.\n", integrate->pressure_coupling);
            break;
        case 21:
            printf("INPUT: Integrate with heating and cooling for this run.\n");
            printf("       choose the Nose-Hoover chain method.\n"); 
            printf("       temperature is %g K.\n", integrate->temperature);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            printf("       delta_T is %g K.\n", integrate->delta_temperature);
            printf("       heat source is group %d.\n", integrate->source);
            printf("       heat sink is group %d.\n", integrate->sink);
            break; 
        case 22:
            printf("INPUT: Integrate with heating and cooling for this run.\n");
            printf("       choose the Langevin method.\n"); 
            printf("       temperature is %g K.\n", integrate->temperature);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            printf("       delta_T is %g K.\n", integrate->delta_temperature);
            printf("       heat source is group %d.\n", integrate->source);
            printf("       heat sink is group %d.\n", integrate->sink);
            break;
        case 23:
            printf("INPUT: Integrate with heating and cooling for this run.\n");
            printf("       choose the Bussi-Donadio-Parrinello method.\n"); 
            printf("       temperature is %g K.\n", integrate->temperature);
            printf("       T_coupling is %g.\n", integrate->temperature_coupling);
            printf("       delta_T is %g K.\n", integrate->delta_temperature);
            printf("       heat source is group %d.\n", integrate->source);
            printf("       heat sink is group %d.\n", integrate->sink);
            break;
        default:
            print_error("invalid ensemble type.\n");
            break; 
    }
}




static void parse_time_step (char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        print_error("time_step should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &atom->time_step))
    {
        print_error("time_step should be a real number.\n");
    } 
    printf("INPUT: time_step for this run is %g fs.\n", atom->time_step);
    atom->time_step /= TIME_UNIT_CONVERSION;
}




static void parse_neighbor
(
    char **param,  int num_param, 
    Atom* atom, Force *force
)
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
    printf
    ("INPUT: build neighbor list with a skin of %g A.\n", atom->neighbor.skin);

    // change the cutoff
    atom->neighbor.rc = force->rc_max + atom->neighbor.skin;
}




static void parse_dump_thermo(char **param,  int num_param, Measure *measure)
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
    printf
    ("INPUT: dump thermo every %d steps.\n", measure->sample_interval_thermo);
}




static void parse_dump_position(char **param,  int num_param, Measure *measure)
{
    if (num_param != 2)
    {
        print_error("dump_position should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->sample_interval_position))
    {
        print_error("position dump interval should be an integer number.\n");
    } 
    measure->dump_position = 1;
    printf
    ("INPUT: dump position every %d steps.\n", measure->sample_interval_position);
}




static void parse_dump_velocity(char **param,  int num_param, Measure *measure)
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
    printf
    ("INPUT: dump velocity every %d steps.\n", measure->sample_interval_velocity);
}




static void parse_dump_force(char **param,  int num_param, Measure *measure)
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
    printf("INPUT: dump force every %d steps.\n", measure->sample_interval_force);
}




static void parse_dump_potential(char **param,  int num_param, Measure *measure)
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
    printf
    (
        "INPUT: dump potential every %d steps.\n", 
        measure->sample_interval_potential
    );
}




static void parse_dump_virial(char **param,  int num_param, Measure *measure)
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
    printf
    ("INPUT: dump virial every %d steps.\n", measure->sample_interval_virial);
}




static void parse_dump_heat(char **param,  int num_param, Measure *measure)
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
    printf("INPUT: dump heat every %d steps.\n", measure->sample_interval_heat);
}




static void parse_compute_vac(char **param,  int num_param, Measure *measure)
{
    printf("INPUT: compute VAC.\n");
    measure->vac.compute = 1;

    if (num_param != 4)
    {
        print_error("compute_vac should have 3 parameters.\n");
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
    printf("       sample interval is %d.\n", measure->vac.sample_interval);

    // number of correlation steps
    if (!is_valid_int(param[2], &measure->vac.Nc))
    {
        print_error("Nc for VAC should be an integer number.\n");
    }
    if (measure->vac.Nc <= 0)
    {
        print_error("Nc for VAC should be positive.\n");
    }
    printf("       Nc is %d.\n", measure->vac.Nc);

    // maximal omega
    if (!is_valid_real(param[3], &measure->vac.omega_max))
    {
        print_error("omega_max should be a real number.\n");
    }
    if (measure->vac.omega_max <= 0)
    {
        print_error("omega_max should be positive.\n");
    }
    printf("       omega_max is %g THz.\n", measure->vac.omega_max);
}




static void parse_compute_hac(char **param,  int num_param, Measure* measure)
{
    measure->hac.compute = 1;

    printf("INPUT: compute HAC.\n");

    if (num_param != 4)
    {
        print_error("compute_hac should have 3 parameters.\n");
    }

    if (!is_valid_int(param[1], &measure->hac.sample_interval))
    {
        print_error("sample interval for HAC should be an integer number.\n");
    }
    printf("       sample interval is %d.\n", measure->hac.sample_interval);

    if (!is_valid_int(param[2], &measure->hac.Nc))
    {
        print_error("Nc for HAC should be an integer number.\n");
    }
    printf("       Nc is %d\n", measure->hac.Nc);

    if (!is_valid_int(param[3], &measure->hac.output_interval))
    {
        print_error("output_interval for HAC should be an integer number.\n");
    }
    printf("       output_interval is %d\n", measure->hac.output_interval);
}




static void parse_compute_hnemd(char **param, int num_param, Measure* measure)
{
    measure->hnemd.compute = 1;

    printf("INPUT: compute thermal conductivity using the HNEMD method.\n");

    if (num_param != 5)
    {
        print_error("compute_hnemd should have 4 parameters.\n");
    }

    if (!is_valid_int(param[1], &measure->hnemd.output_interval))
    {
        print_error("output_interval for HNEMD should be an integer number.\n");
    }
    printf("       output_interval = %d\n", measure->hnemd.output_interval);
    if (measure->hnemd.output_interval < 1)
    {
        print_error("output_interval for HNEMD should be larger than 0.\n");
    }
    if (!is_valid_real(param[2], &measure->hnemd.fe_x))
    {
        print_error("fe_x for HNEMD should be a real number.\n");
    }
    printf("       fe_x = %g /A\n", measure->hnemd.fe_x);
    if (!is_valid_real(param[3], &measure->hnemd.fe_y))
    {
        print_error("fe_y for HNEMD should be a real number.\n");
    }
    printf("       fe_y = %g /A\n", measure->hnemd.fe_y);
    if (!is_valid_real(param[4], &measure->hnemd.fe_z))
    {
        print_error("fe_z for HNEMD should be a real number.\n");
    }
    printf("       fe_z = %g /A\n", measure->hnemd.fe_z);

    // magnitude of the vector
    measure->hnemd.fe  = measure->hnemd.fe_x * measure->hnemd.fe_x;
    measure->hnemd.fe += measure->hnemd.fe_y * measure->hnemd.fe_y;
    measure->hnemd.fe += measure->hnemd.fe_z * measure->hnemd.fe_z;
    measure->hnemd.fe  = sqrt(measure->hnemd.fe);
}




static void parse_compute_shc(char **param,  int num_param, Measure* measure)
{
    printf("INPUT: compute SHC.\n");
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
    ("       sample interval for SHC is %d.\n", measure->shc.sample_interval);

    // number of correlation data
    if (!is_valid_int(param[2], &measure->shc.Nc))
    {
        print_error("Nc for SHC should be an integer.\n");
    }  
    printf("       number of correlation data is %d.\n", measure->shc.Nc);

    // number of time origins 
    if (!is_valid_int(param[3], &measure->shc.M))
    {
        print_error("M for SHC should be an integer.\n");
    }  
    printf("       number of time origions is %d.\n", measure->shc.M);

    // block A 
    if (!is_valid_int(param[4], &measure->shc.block_A))
    {
        print_error("block_A for SHC should be an integer.\n");
    }  
    printf
    ("       record the heat flowing from group %d.\n", measure->shc.block_A);
    
    // block B 
    if (!is_valid_int(param[5], &measure->shc.block_B))
    {
        print_error("block_B for SHC should be an integer.\n");
    }  
    printf
    ("       record the heat flowing into group %d.\n", measure->shc.block_B);
}




static void parse_deform(char **param,  int num_param, Atom* atom)
{
    print_error("the deform keyword is to be implemented.\n");
}




static void parse_compute_temp(char **param,  int num_param, Measure* measure)
{
    measure->heat.sample = 1;
    if (num_param != 2)
    {
        print_error("compute_temp should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &measure->heat.sample_interval))
    {
        print_error("temperature sampling interval should be an integer.\n");
    }  
    printf
    (
        "INPUT: sample block temperatures every %d steps.\n", 
        measure->heat.sample_interval
    );
}




static void parse_fix(char **param, int num_param, Atom *atom)
{
    if (num_param != 2)
    {
        print_error("fix should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &atom->fixed_group))
    {
        print_error("fixed_group should be an integer.\n");
    }  
    printf("INPUT: group %d will be fixed.\n", atom->fixed_group);
}




static void parse_run(char **param,  int num_param, Atom* atom)
{
    if (num_param != 2)
    {
        print_error("run should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &atom->number_of_steps))
    {
        print_error("number of steps should be an integer.\n");
    }
    printf("INPUT: run %d steps.\n", atom->number_of_steps);
}




void parse
(
    char **param, int num_param, Atom* atom,
    Force *force, Integrate *integrate, Measure *measure,
    int *is_potential,int *is_velocity,int *is_run
)
{
    if (strcmp(param[0], "potential") == 0)
    {
        *is_potential = 1;
        parse_potential(param, num_param, force);
    }
    else if (strcmp(param[0], "potentials") == 0)
    {
        *is_potential = 1;
        parse_potentials(param, num_param, force);
    }
    else if (strcmp(param[0], "velocity") == 0)
    {
        *is_velocity = 1;
        parse_velocity(param, num_param, atom);
    }
    else if (strcmp(param[0], "ensemble")       == 0) 
    {
        parse_ensemble(param, num_param, atom, integrate);
    }
    else if (strcmp(param[0], "time_step")      == 0) 
    {
        parse_time_step(param, num_param, atom);
    }
    else if (strcmp(param[0], "neighbor")       == 0) 
    {
        parse_neighbor(param, num_param, atom, force);
    }
    else if (strcmp(param[0], "dump_thermo")    == 0) 
    {
        parse_dump_thermo(param, num_param, measure);
    }
    else if (strcmp(param[0], "dump_position")  == 0) 
    {
        parse_dump_position(param, num_param, measure);
    }
    else if (strcmp(param[0], "dump_velocity")  == 0) 
    {
        parse_dump_velocity(param, num_param, measure);
    }
    else if (strcmp(param[0], "dump_force")     == 0) 
    {
        parse_dump_force(param, num_param, measure);
    }
    else if (strcmp(param[0], "dump_potential") == 0) 
    {
        parse_dump_potential(param, num_param, measure);
    }
    else if (strcmp(param[0], "dump_virial")    == 0) 
    {
        parse_dump_virial(param, num_param, measure);
    }
    else if (strcmp(param[0], "dump_heat")    == 0) 
    {
        parse_dump_heat(param, num_param, measure);
    }
    else if (strcmp(param[0], "compute_vac")    == 0) 
    {
        parse_compute_vac(param, num_param, measure);
    }
    else if (strcmp(param[0], "compute_hac")    == 0) 
    {
        parse_compute_hac(param, num_param, measure);
    }
    else if (strcmp(param[0], "compute_hnemd") == 0) 
    {
        parse_compute_hnemd(param, num_param, measure);
    }
    else if (strcmp(param[0], "compute_shc")    == 0) 
    {
        parse_compute_shc(param, num_param, measure);
    }
    else if (strcmp(param[0], "deform")         == 0) 
    {
        parse_deform(param, num_param, atom);
    }
    else if (strcmp(param[0], "compute_temp")   == 0) 
    {
        parse_compute_temp(param, num_param, measure);
    }
    else if (strcmp(param[0], "fix")            == 0) 
    {
        parse_fix(param, num_param, atom);
    }
    else if (strcmp(param[0], "run")            == 0)
    {
        *is_run = 1;
        parse_run(param, num_param, atom);
    }
    else
    {
        print_error("invalid keyword.\n");
    }
}

