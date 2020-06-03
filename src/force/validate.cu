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
Use finite difference to validate the analytical force calculations.
------------------------------------------------------------------------------*/


#include "validate.cuh"
#include "force.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/error.cuh"


// This choice gives optimal accuracy for finite-difference calculations
#define DX1 1.0e-7
#define DX2 2.0e-7


// move one atom left or right
static __global__ void shift_atom
(
    const int d,
    const int n,
    const int direction,
    const double *x0,
    const double *y0,
    const double *z0,
    double *x,
    double *y,
    double *z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 == n)
    {
        if (d == 0)
        {
            if (direction == 1)
            {
                x[n] = x0[n] - DX1;
            }
            else
            {
                x[n] = x0[n] + DX1;
            }
        }
        else if (d == 1)
        {
            if (direction == 1)
            {
                y[n] = y0[n] - DX1;
            }
            else
            {
                y[n] = y0[n] + DX1;
            }
        }
        else
        {
            if (direction == 1)
            {
                z[n] = z0[n] - DX1;
            }
            else
            {
                z[n] = z0[n] + DX1;
            }
        } 
    }
}


// get the total potential form the per-atom potentials
static __global__ void sum_potential
(
    const int N,
    const int m,
    const double *p,
    double *p_sum
)
{
    int tid = threadIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1; 
    
    __shared__ double s_sum[1024];
    s_sum[tid] = 0;
    
    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int n = tid + patch * 1024;
        if (n < N)
        {        
            s_sum[tid] += p[n];
        }
    }
    
    __syncthreads();
    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_sum[tid] += s_sum[tid + offset]; }
        __syncthreads();
    } 

    if (tid ==  0) 
    {
        p_sum[m] = s_sum[0]; 
    }
}


// get the forces from the potential energies using finite difference
static __global__ void find_force_from_potential
(
    const int N,
    const double *p1,
    const double *p2,
    double *fx,
    double *fy,
    double *fz
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    int m;
    if (n1 < N)
    {
        m = n1; fx[n1] = (p1[m] - p2[m]) / DX2;
        m += N; fy[n1] = (p1[m] - p2[m]) / DX2;
        m += N; fz[n1] = (p1[m] - p2[m]) / DX2; 
    }
}


void validate_force
(
    const Box& box,
    GPU_Vector<double>& position_per_atom,
    std::vector<Group>& group,
    GPU_Vector<int>& type,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    Neighbor& neighbor,
    Force *force
)
{
    const int number_of_atoms = type.size();

    std::vector<double> cpu_force(number_of_atoms * 3);

    // first calculate the forces directly:
    force->compute
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

    // make a copy of the positions
    GPU_Vector<double> r0(number_of_atoms * 3);
    r0.copy_from_device(position_per_atom.data());

    // get the potentials
    GPU_Vector<double> p1(number_of_atoms * 3), p2(number_of_atoms * 3);
    for (int d = 0; d < 3; ++d)
    {
        for (int n = 0; n < number_of_atoms; ++n)
        {
            int m = d * number_of_atoms + n;

            // shift one atom to the left by a small amount
            shift_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>
            (
                d,
                n,
                1,
                r0.data(),
                r0.data() + number_of_atoms,
                r0.data() + number_of_atoms * 2,
                position_per_atom.data(),
                position_per_atom.data() + number_of_atoms,
                position_per_atom.data() + number_of_atoms * 2
            );
            CUDA_CHECK_KERNEL

            // get the potential energy
            force->compute
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

            // sum up the potential energy
            sum_potential<<<1, 1024>>>
            (
                number_of_atoms,
                m,
                potential_per_atom.data(),
                p1.data()
            );
            CUDA_CHECK_KERNEL

            // shift one atom to the right by a small amount
            shift_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>
            (
                d,
                n,
                2,
                r0.data(),
                r0.data() + number_of_atoms,
                r0.data() + number_of_atoms * 2,
                position_per_atom.data(),
                position_per_atom.data() + number_of_atoms,
                position_per_atom.data() + number_of_atoms * 2
            );
            CUDA_CHECK_KERNEL

            // get the potential energy
            force->compute
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

            // sum up the potential energy
            sum_potential<<<1, 1024>>>
            (
                number_of_atoms,
                m,
                potential_per_atom.data(),
                p2.data()
            );
            CUDA_CHECK_KERNEL
        }
    }

    // copy the positions back (as if nothing happens)
    r0.copy_to_device(position_per_atom.data());

    // get the forces from the potential energies using finite difference
    GPU_Vector<double> force_compare(number_of_atoms * 3);
    find_force_from_potential<<<(number_of_atoms - 1) / 128 + 1, 128>>>
    (
        number_of_atoms,
        p1.data(),
        p2.data(),
        force_compare.data(),
        force_compare.data() + number_of_atoms,
        force_compare.data() + number_of_atoms * 2
    );
    CUDA_CHECK_KERNEL

    // open file
    FILE *fid = my_fopen("f_compare.out", "w");
    
    // output the forces from direct calculations
    force_per_atom.copy_to_host(cpu_force.data());
    for (int n = 0; n < number_of_atoms; n++)
    {
        fprintf
        (
            fid, "%25.15e%25.15e%25.15e\n",
            cpu_force[n],
            cpu_force[n + number_of_atoms],
            cpu_force[n + number_of_atoms * 2]
        );
    }
 
    // output the forces from finite difference
    force_compare.copy_to_host(cpu_force.data());
    for (int n = 0; n < number_of_atoms; n++)
    {
        fprintf
        (
            fid, "%25.15e%25.15e%25.15e\n",
            cpu_force[n],
            cpu_force[n + number_of_atoms],
            cpu_force[n + number_of_atoms * 2]
        );
    }
    
    // close file
    fflush(fid);
    fclose(fid); 
}


