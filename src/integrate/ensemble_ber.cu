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
The Berendsen thermostat:
[1] H. J. C. Berendsen et al. J. Chem. Phys. 81, 3684 (1984).
------------------------------------------------------------------------------*/


#include "ensemble_ber.cuh"


Ensemble_BER::Ensemble_BER(int t, int fg, double T, double Tc)
{
    type = t;
    fixed_group = fg;
    temperature = T;
    temperature_coupling = Tc;
}


Ensemble_BER::Ensemble_BER
(
    int t,
    int fg,
    double T,
    double Tc,
    double px,
    double py,
    double pz,
    double pc,
    int dx,
    int dy,
    int dz,
    double rate
)
{
    type = t;
    fixed_group = fg;
    temperature = T;
    temperature_coupling = Tc;
    pressure_x = px;
    pressure_y = py;
    pressure_z = pz;
    pressure_coupling = pc;
    deform_x = dx;
    deform_y = dy;
    deform_z = dz;
    deform_rate = rate;
}


Ensemble_BER::~Ensemble_BER(void)
{
    // nothing now
}


static __global__ void gpu_berendsen_temperature
(
    int N, double temperature, double coupling, double *g_prop, 
    double *g_vx, double *g_vy, double *g_vz
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {  
        double factor = sqrt(1.0 + coupling * (temperature / g_prop[0] - 1.0)); 
        g_vx[i] *= factor; 
        g_vy[i] *= factor; 
        g_vz[i] *= factor;
    }
}


static __global__ void gpu_berendsen_pressure
(
    int deform_x, int deform_y, int deform_z, double deform_rate,
    int number_of_particles, Box box,
    double p0x, double p0y, double p0z, double p_coupling, 
    double *g_prop, double *g_x, double *g_y, double *g_z
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_particles)
    {
        if (deform_x)
        {
            double scale_factor = box.cpu_h[0];
            scale_factor = (scale_factor + deform_rate) / scale_factor;
            g_x[i] *= scale_factor;
        }
        else if (box.pbc_x == 1)
        {
            double scale_factor = 1.0 - p_coupling * (p0x - g_prop[2]);
            g_x[i] *= scale_factor;
        }
        if (deform_y)
        {
            double scale_factor = box.cpu_h[1];
            scale_factor = (scale_factor + deform_rate) / scale_factor;
            g_y[i] *= scale_factor;
        }
        else if (box.pbc_y == 1)
        {
            double scale_factor = 1.0 - p_coupling * (p0y - g_prop[3]);
            g_y[i] *= scale_factor;
        }
        if (deform_z)
        {
            double scale_factor = box.cpu_h[2];
            scale_factor = (scale_factor + deform_rate) / scale_factor;
            g_z[i] *= scale_factor;
        }
        else if (box.pbc_z == 1)
        {
            double scale_factor = 1.0 - p_coupling * (p0z - g_prop[4]);
            g_z[i] *= scale_factor;
        }
    }
}


static void cpu_berendsen_pressure
(
    int deform_x, int deform_y, int deform_z, double deform_rate, Box& box,
    double p0x, double p0y, double p0z, double p_coupling, double *thermo
)
{
    double p[3];
    CHECK(cudaMemcpy(p, thermo+2, sizeof(double)*3, cudaMemcpyDeviceToHost));

    if (deform_x)
    {
        double scale_factor = box.cpu_h[0];
        scale_factor = (scale_factor + deform_rate) / scale_factor;
        box.cpu_h[0] *= scale_factor;
        box.cpu_h[3] = box.cpu_h[0] * 0.5;
    }
    else if (box.pbc_x == 1)
    {
        double scale_factor = 1.0 - p_coupling * (p0x - p[0]);
        box.cpu_h[0] *= scale_factor;
        box.cpu_h[3] = box.cpu_h[0] * 0.5;
    }

    if (deform_y)
    {
        double scale_factor = box.cpu_h[1];
        scale_factor = (scale_factor + deform_rate) / scale_factor;
        box.cpu_h[1] *= scale_factor;
        box.cpu_h[4] = box.cpu_h[1] * 0.5;
    }
    else if (box.pbc_y == 1)
    {
        double scale_factor = 1.0 - p_coupling * (p0y - p[1]);
        box.cpu_h[1] *= scale_factor;
        box.cpu_h[4] = box.cpu_h[1] * 0.5;
    }

    if (deform_z)
    {
        double scale_factor = box.cpu_h[2];
        scale_factor = (scale_factor + deform_rate) / scale_factor;
        box.cpu_h[2] *= scale_factor;
        box.cpu_h[5] = box.cpu_h[2] * 0.5;
    }
    else if (box.pbc_z == 1)
    {
        double scale_factor = 1.0 - p_coupling * (p0x - p[2]);
        box.cpu_h[2] *= scale_factor;
        box.cpu_h[5] = box.cpu_h[2] * 0.5;
    }
}


void Ensemble_BER::compute1
(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo
)
{
    velocity_verlet
    (
        true,
        time_step,
        group,
        mass,
        force_per_atom,
        position_per_atom,
        velocity_per_atom
     );
}


void Ensemble_BER::compute2
(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo
)
{
    const int number_of_atoms = mass.size();

    velocity_verlet
    (
        false,
        time_step,
        group,
        mass,
        force_per_atom,
        position_per_atom,
        velocity_per_atom
     );

    find_thermo
    (
        box.get_volume(),
        group,
        mass,
        potential_per_atom,
        velocity_per_atom,
        virial_per_atom,
        thermo
    );
    gpu_berendsen_temperature<<<(number_of_atoms - 1) / 128 + 1, 128>>>
    (
        number_of_atoms,
        temperature,
        temperature_coupling,
        thermo.data(),
        velocity_per_atom.data(),
        velocity_per_atom.data() + number_of_atoms,
        velocity_per_atom.data() + 2 * number_of_atoms
    );
    CUDA_CHECK_KERNEL
    if (type == 11)
    {
        gpu_berendsen_pressure<<<(number_of_atoms - 1) / 128 + 1, 128>>>
        (
            deform_x,
            deform_y,
            deform_z,
            deform_rate,
            number_of_atoms,
            box,
            pressure_x,
            pressure_y,
            pressure_z,
            pressure_coupling,
            thermo.data(),
            position_per_atom.data(),
            position_per_atom.data() + number_of_atoms,
            position_per_atom.data() + number_of_atoms * 2
        );
        CUDA_CHECK_KERNEL
        cpu_berendsen_pressure
        (
            deform_x,
            deform_y,
            deform_z,
            deform_rate,
            box,
            pressure_x,
            pressure_y,
            pressure_z,
            pressure_coupling,
            thermo.data()
        );
    }
}


