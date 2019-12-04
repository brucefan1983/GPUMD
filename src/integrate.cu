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
The driver class for the various integrators.
------------------------------------------------------------------------------*/


#include "integrate.cuh"

#include "atom.cuh"
#include "ensemble.cuh"
#include "ensemble_nve.cuh"
#include "ensemble_ber.cuh"
#include "ensemble_nhc.cuh"
#include "ensemble_lan.cuh"
#include "ensemble_bdp.cuh"
#include "error.cuh"
#include "force.cuh"
#include "read_file.cuh"


Integrate::Integrate(void)
{
    ensemble = NULL;
}


Integrate::~Integrate(void)
{
    // nothing
}


void Integrate::finalize(void)
{
    delete ensemble;
    ensemble = NULL;
}


void Integrate::initialize(Atom* atom)
{
    // determine the integrator
    switch (type)
    {
        case 0: // NVE
            ensemble = new Ensemble_NVE(type, fixed_group);
            break;
        case 1: // NVT-Berendsen
            ensemble = new Ensemble_BER
            (type, fixed_group, temperature, temperature_coupling);
            break;
        case 2: // NVT-NHC
            ensemble = new Ensemble_NHC            
            (
                type, fixed_group, atom->N, temperature, temperature_coupling, 
                atom->time_step
            );
            break;
        case 3: // NVT-Langevin
            ensemble = new Ensemble_LAN
            (type, fixed_group, atom->N, temperature, temperature_coupling);
            break;
        case 4: // NVT-BDP
            ensemble = new Ensemble_BDP            
            (type, fixed_group, temperature, temperature_coupling);
            break;
        case 11: // NPT-Berendsen
            ensemble = new Ensemble_BER
            (
                type, fixed_group, temperature, temperature_coupling, 
                pressure_x, pressure_y, pressure_z, pressure_coupling,
                deform_x, deform_y, deform_z, deform_rate
            );
            break;
        case 21: // heat-NHC
            ensemble = new Ensemble_NHC
            (
                type, fixed_group, source, sink, 
                atom->group[0].cpu_size[source], atom->group[0].cpu_size[sink],
                temperature, temperature_coupling, delta_temperature, 
                atom->time_step
            );
            break;
        case 22: // heat-Langevin
            ensemble = new Ensemble_LAN
            (
                type, fixed_group, source, sink, 
                atom->group[0].cpu_size[source],
                atom->group[0].cpu_size[sink],
                atom->group[0].cpu_size_sum[source],
                atom->group[0].cpu_size_sum[sink],
                temperature, temperature_coupling, delta_temperature
            );
            break;
        case 23: // heat-BDP
            ensemble = new Ensemble_BDP
            (
                type, fixed_group, source, sink, temperature, 
                temperature_coupling, delta_temperature
            );
            break;
        default: 
            printf("Illegal integrator!\n");
            break;
    }
}


void Integrate::compute(Atom *atom, Force *force, Measure* measure)
{
    if (type >= 1 && type <= 20)
    {
        ensemble->temperature = temperature1 + (temperature2 - temperature1)
                              * real(atom->step) / atom->number_of_steps;
    }

    ensemble->compute(atom, force, measure);
}


// coding conventions:
//0:     NVE
//1-10:  NVT
//11-20: NPT
//21-30: heat (NEMD method for heat conductivity)
void Integrate::parse_ensemble(char **param, int num_param, Atom *atom)
{
    // 1. Determine the integration method
    if (strcmp(param[1], "nve") == 0)
    {
        type = 0;
        if (num_param != 2)
        {
            print_error("ensemble nve should have 0 parameter.\n");
        }
    }
    else if (strcmp(param[1], "nvt_ber") == 0)
    {
        type = 1;
        if (num_param != 5)
        {
            print_error("ensemble nvt_ber should have 3 parameters.\n");
        }
    }
    else if (strcmp(param[1], "nvt_nhc") == 0)
    {
        type = 2;
        if (num_param != 5)
        {
            print_error("ensemble nvt_nhc should have 3 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "nvt_lan") == 0)
    {
        type = 3;
        if (num_param != 5)
        {
            print_error("ensemble nvt_lan should have 3 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "nvt_bdp") == 0)
    {
        type = 4;
        if (num_param != 5)
        {
            print_error("ensemble nvt_bdp should have 3 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "npt_ber") == 0)
    {
        type = 11;
        if (num_param != 9)
        {
            print_error("ensemble npt_ber should have 7 parameters.\n"); 
        } 
    }
    else if (strcmp(param[1], "heat_nhc") == 0)
    {
        type = 21;
        if (num_param != 7)
        {
            print_error("ensemble heat_nhc should have 5 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "heat_lan") == 0)
    {
        type = 22;
        if (num_param != 7)
        {
            print_error("ensemble heat_lan should have 5 parameters.\n"); 
        }
    }
    else if (strcmp(param[1], "heat_bdp") == 0)
    {
        type = 23;
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
    if (type >= 1 && type <= 20)
    {
        // initial temperature
        if (!is_valid_real(param[2], &temperature1))
        {
            print_error("ensemble temperature should be a real number.\n");
        }
        if (temperature1 <= 0.0)
        {
            print_error("ensemble temperature should be a positive number.\n");
        }

        // final temperature
        if (!is_valid_real(param[3], &temperature2))
        {
            print_error("ensemble temperature should be a real number.\n");
        }
        if (temperature2 <= 0.0)
        {
            print_error("ensemble temperature should be a positive number.\n");
        }

        temperature = temperature1;

        // temperature_coupling
        if (!is_valid_real(param[4], &temperature_coupling))
        {
            print_error("temperature_coupling should be a real number.\n");
        }
        if (temperature_coupling <= 0.0)
        {
            print_error("temperature_coupling should be a positive number.\n");
        }
    }

    // 3. Pressures and pressure_coupling (NPT)
    real pressure[3];
    if (type >= 11 && type <= 20)
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
        pressure_x = pressure[0] / PRESSURE_UNIT_CONVERSION;
        pressure_y = pressure[1] / PRESSURE_UNIT_CONVERSION;
        pressure_z = pressure[2] / PRESSURE_UNIT_CONVERSION;

        // pressure_coupling:
        if (!is_valid_real(param[8], &pressure_coupling))
        {
            print_error("pressure_coupling should be a real number.\n");
        }
        if (pressure_coupling <= 0.0)
        {
            print_error("pressure_coupling should be a positive number.\n");
        }
    }

    // 4. heating and cooling wiht fixed temperatures
    if (type >= 21 && type <= 30)
    {
        // temperature
        if (!is_valid_real(param[2], &temperature))
        {
            print_error("ensemble temperature should be a real number.\n");
        }
        if (temperature <= 0.0)
        {
            print_error("ensemble temperature should be a positive number.\n");
        }

        // temperature_coupling
        if (!is_valid_real(param[3], &temperature_coupling))
        {
            print_error("temperature_coupling should be a real number.\n");
        }
        if (temperature_coupling <= 0.0)
        {
            print_error("temperature_coupling should be a positive number.\n");
        }

        // temperature difference
        if (!is_valid_real(param[4], &delta_temperature))
        {
            print_error("delta_temperature should be a real number.\n");
        }

        // group labels of heat source and sink
        if (!is_valid_int(param[5], &source))
        {
            print_error("heat.source should be an integer.\n");
        }
        if (!is_valid_int(param[6], &sink))
        {
            print_error("heat.sink should be an integer.\n");
        }
    }

    switch (type)
    {
        case 0:
            printf("Use NVE ensemble for this run.\n");
            break;
        case 1:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Berendsen method.\n"); 
            printf("    initial temperature is %g K.\n", temperature1);
            printf("    final temperature is %g K.\n", temperature2);
            printf("    T_coupling is %g.\n", temperature_coupling);
            break;
        case 2:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Nose-Hoover chain method.\n"); 
            printf("    initial temperature is %g K.\n", temperature1);
            printf("    final temperature is %g K.\n", temperature2);
            printf("    T_coupling is %g.\n", temperature_coupling);
            break;
        case 3:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Langevin method.\n"); 
            printf("    initial temperature is %g K.\n", temperature1);
            printf("    final temperature is %g K.\n", temperature2);
            printf("    T_coupling is %g.\n", temperature_coupling);
            break;
        case 4:
            printf("Use NVT ensemble for this run.\n");
            printf("    choose the Bussi-Donadio-Parrinello method.\n"); 
            printf("    initial temperature is %g K.\n", temperature1);
            printf("    final temperature is %g K.\n", temperature2);
            printf("    T_coupling is %g.\n", temperature_coupling);
            break;
        case 11:
            printf("Use NPT ensemble for this run.\n");
            printf("    choose the Berendsen method.\n");      
            printf("    initial temperature is %g K.\n", temperature1);
            printf("    final temperature is %g K.\n", temperature2);
            printf("    T_coupling is %g.\n", temperature_coupling);
            printf("    pressure_x is %g GPa.\n", pressure[0]);
            printf("    pressure_y is %g GPa.\n", pressure[1]);
            printf("    pressure_z is %g GPa.\n", pressure[2]);
            printf("    p_coupling is %g.\n", pressure_coupling);
            break;
        case 21:
            printf("Integrate with heating and cooling for this run.\n");
            printf("    choose the Nose-Hoover chain method.\n"); 
            printf("    temperature is %g K.\n", temperature);
            printf("    T_coupling is %g.\n", temperature_coupling);
            printf("    delta_T is %g K.\n", delta_temperature);
            printf("    heat source is group %d.\n", source);
            printf("    heat sink is group %d.\n", sink);
            break; 
        case 22:
            printf("Integrate with heating and cooling for this run.\n");
            printf("    choose the Langevin method.\n"); 
            printf("    temperature is %g K.\n", temperature);
            printf("    T_coupling is %g.\n", temperature_coupling);
            printf("    delta_T is %g K.\n", delta_temperature);
            printf("    heat source is group %d.\n", source);
            printf("    heat sink is group %d.\n", sink);
            break;
        case 23:
            printf("Integrate with heating and cooling for this run.\n");
            printf("    choose the Bussi-Donadio-Parrinello method.\n"); 
            printf("    temperature is %g K.\n", temperature);
            printf("    T_coupling is %g.\n", temperature_coupling);
            printf("    delta_T is %g K.\n", delta_temperature);
            printf("    heat source is group %d.\n", source);
            printf("    heat sink is group %d.\n", sink);
            break;
        default:
            print_error("invalid ensemble type.\n");
            break; 
    }
}


void Integrate::parse_fix(char **param, int num_param)
{
    if (num_param != 2)
    {
        print_error("fix should have 1 parameter.\n");
    }
    if (!is_valid_int(param[1], &fixed_group))
    {
        print_error("fixed_group should be an integer.\n");
    }
    printf("Group %d will be fixed.\n", fixed_group);
}


void Integrate::parse_deform(char **param, int num_param)
{
    printf("Deform the box.\n");

    if (num_param != 5)
    {
        print_error("deform should have 4 parameters.\n");
    }

    // strain rate
    if (!is_valid_real(param[1], &deform_rate))
    {
        print_error("defrom rate should be a number.\n");
    }
    printf("    strain rate is %g A / step.\n", deform_rate);

    // direction
    if (!is_valid_int(param[2], &deform_x))
    {
        print_error("deform_x should be integer.\n");
    }
    if (!is_valid_int(param[3], &deform_y))
    {
        print_error("deform_y should be integer.\n");
    }
    if (!is_valid_int(param[4], &deform_z))
    {
        print_error("deform_z should be integer.\n");
    }

    if (deform_x)
    {
        printf("    apply strain in x direction.\n");
    }
    if (deform_y)
    {
        printf("    apply strain in y direction.\n");
    }
    if (deform_z)
    {
        printf("    apply strain in z direction.\n");
    }
}


