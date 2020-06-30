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
The SD (steepest decent) minimizer.
------------------------------------------------------------------------------*/


#include "minimizer_sd.cuh"
#include "force/force.cuh"

const double decreasing_factor = 0.2;
const double increasing_factor = 1.2;


namespace 
{

__global__ void update_positions
(
    const int number_of_atoms,
    const double position_step,
    const double* force_per_atom,
    const double* position_per_atom,
    double* position_per_atom_temp
)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number_of_atoms) 
    {
        const double position_change = force_per_atom[n] * position_step;
        position_per_atom_temp[n] = position_per_atom[n] + position_change;
    }
}


} // namespace


void Minimizer_SD::compute
(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    Neighbor& neighbor,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom
)
{
    const int number_of_atoms = type.size();

    force.compute
    (
        box,
        position_per_atom,
        type,
        group,
        neighbor,
        potential_per_atom,
        force_per_atom,
        virial_per_atom
    );

    double position_step = 0.1;

    for (int n = 0; n < number_of_steps_; ++n)
    {
        calculate_force_square_sum(force_per_atom);

        if (cpu_force_square_sum_[0] < force_tolerance_square_) break;

        update_positions<<<(number_of_atoms_ - 1) / 128 + 1 , 128>>>
        (
            number_of_atoms_,
            position_step,
            force_per_atom.data(),
            position_per_atom.data(),
            position_per_atom_temp_.data()
        );

        force.compute
        (
            box,
            position_per_atom_temp_,
            type,
            group,
            neighbor,
            potential_per_atom_temp_,
            force_per_atom_temp_,
            virial_per_atom
        );

        calculate_potential_difference(potential_per_atom);

        if (cpu_potential_difference_[0] > 0.0) 
        {
            position_step *= decreasing_factor;
        }
        else
        {
            position_per_atom_temp_.copy_to_device(position_per_atom.data());
            force_per_atom_temp_.copy_to_device(force_per_atom.data());
            potential_per_atom_temp_.copy_to_device(potential_per_atom.data());
            position_step *= increasing_factor;
        }
    }
}

