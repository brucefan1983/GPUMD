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
Use finite difference to calculate the hessian (force constants).
    H_ij^ab = [F_i^a(-) - F_i^a(+)] / [u_j^b(+) - u_j^b(-)]
------------------------------------------------------------------------------*/


#include "hessian.cuh"
#include "force.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE 128


Hessian::Hessian(Atom *atom, Force *force, Measure* measure)
{
    MY_MALLOC(H, real, atom->N * atom->N * 9);
    MY_MALLOC(f_negative, real, 3);
    MY_MALLOC(f_positive, real, 3);
    find_H(atom, force, measure);
}


Hessian::~Hessian(void)
{
    MY_FREE(H);
    MY_FREE(f_negative);
    MY_FREE(f_positive);
}


void Hessian::find_H(Atom *atom, Force *force, Measure* measure)
{
    int N = atom->N;
    for (int n1 = 0; n1 < N; ++n1)
    {
        for (int n2 = 0; n2 < N; ++n2)
        {
            int offset = (n1 * N + n2) * 9;
            find_H12(n1, n2, atom, force, measure, H + offset);
        }
    }

// test (gives identical results as those from my MATLAB code!)
    FILE* fid = fopen("H.txt", "w");
    for (int n1 = 0; n1 < N; ++n1)
    {
        for (int n2 = 0; n2 < N; ++n2)
        {
            int offset = (n1 * N + n2) * 9;
            for (int k = 0; k < 9; ++k)
            {
                fprintf(fid, "%g ", H[offset + k]);
            }
            fprintf(fid, "\n");
        }
    }
    fclose(fid);
}


void Hessian::find_H12
(int n1, int n2, Atom *atom, Force *force, Measure* measure, real* H12)
{
    for (int beta = 0; beta < 3; ++beta)
    {
        get_f(n1, n2, beta, -1, atom, force, measure, f_negative);
        get_f(n1, n2, beta, +1, atom, force, measure, f_positive);
        for (int alpha = 0; alpha < 3; ++alpha)
        {
            int index = alpha * 3 + beta;
            H12[index] = (f_negative[alpha] - f_positive[alpha]) / dx2;
        }
    }
}


void Hessian::get_f
(
    int n1, int n2, int beta, int direction, 
    Atom* atom, Force *force, Measure* measure, real* f
)
{
    shift_atom(n2, beta, +direction, atom);
    force->compute(atom, measure);
    int M = sizeof(real);
    CHECK(cudaMemcpy(f + 0, atom->fx + n1, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(f + 1, atom->fy + n1, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(f + 2, atom->fz + n1, M, cudaMemcpyDeviceToHost));
    shift_atom(n2, beta, -direction, atom);
}


static __global__ void gpu_shift_atom(real dx, real *x)
{
    x[0] += dx;
}


void Hessian::shift_atom(int n2, int beta, int direction, Atom* atom)
{
    if (beta == 0)
    {
        gpu_shift_atom<<<1, 1>>>(direction * dx, atom->x + n2);
        CUDA_CHECK_KERNEL
    }
    else if (beta == 1)
    {
        gpu_shift_atom<<<1, 1>>>(direction * dx, atom->y + n2);
        CUDA_CHECK_KERNEL
    }
    else
    {
        gpu_shift_atom<<<1, 1>>>(direction * dx, atom->z + n2);
        CUDA_CHECK_KERNEL
    }
}


