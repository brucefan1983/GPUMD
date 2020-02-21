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
Then calculate the dynamical matrices with different k points.
------------------------------------------------------------------------------*/


#include "hessian.cuh"
#include "force.cuh"
#include "atom.cuh"
#include "error.cuh"
#include "mic.cuh"
#include "cusolver_wrapper.cuh"
#include "read_file.cuh"

#define BLOCK_SIZE 128


void Hessian::compute
(char* input_dir, Atom* atom, Force* force, Measure* measure)
{
    initialize(input_dir, atom->N);
    find_H(atom, force, measure);
    find_D(input_dir, atom);

    // currently for Alex's GKMA calculations
    if (num_kpoints == 1)
    {
        find_eigenvectors(input_dir, atom);
    }

    finalize();
}


void Hessian::read_basis(char* input_dir, int N)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/basis.in");
    FILE *fid = fopen(file, "r");
    int count;
    count = fscanf(fid, "%d", &num_basis);
    PRINT_SCANF_ERROR(count, 1, "Reading error for basis.in.");

    MY_MALLOC(basis, int, num_basis);
    MY_MALLOC(mass, real, num_basis);
    for (int m = 0; m < num_basis; ++m)
    {
        count = fscanf(fid, "%d%lf", &basis[m], &mass[m]);
        PRINT_SCANF_ERROR(count, 2, "Reading error for basis.in.");
    }
    MY_MALLOC(label, int, N);
    for (int n = 0; n < N; ++n)
    {
        count = fscanf(fid, "%d", &label[n]);
        PRINT_SCANF_ERROR(count, 1, "Reading error for basis.in.");
    }
    fclose(fid);
}


void Hessian::read_kpoints(char* input_dir)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/kpoints.in");
    FILE *fid = fopen(file, "r");
    int count;
    count = fscanf(fid, "%d", &num_kpoints);
    PRINT_SCANF_ERROR(count, 1, "Reading error for kpoints.in.");

    MY_MALLOC(kpoints, real, num_kpoints * 3);
    for (int m = 0; m < num_kpoints; ++m)
    {
        count = fscanf(fid, "%lf%lf%lf", &kpoints[m * 3 + 0],
            &kpoints[m * 3 + 1], &kpoints[m * 3 + 2]);
        PRINT_SCANF_ERROR(count, 3, "Reading error for kpoints.in.");
    }
    fclose(fid);
}


void Hessian::initialize(char* input_dir, int N)
{
    read_basis(input_dir, N);
    read_kpoints(input_dir);
    int num_H = num_basis * N * 9;
    int num_D = num_basis * num_basis * 9 * num_kpoints;
    MY_MALLOC(H, real, num_H);
    MY_MALLOC(DR, real, num_D);
    MY_MALLOC(DI, real, num_D);
    for (int n = 0; n < num_H; ++n) { H[n] = 0; }
    for (int n = 0; n < num_D; ++n) { DR[n] = DI[n] = 0; }
}


void Hessian::finalize(void)
{
    MY_FREE(basis);
    MY_FREE(label);
    MY_FREE(mass);
    MY_FREE(kpoints);
    MY_FREE(H);
    MY_FREE(DR);
    MY_FREE(DI);
}


bool Hessian::is_too_far(int n1, int n2, Atom* atom)
{
    real x12 = atom->cpu_x[n2] - atom->cpu_x[n1];
    real y12 = atom->cpu_y[n2] - atom->cpu_y[n1];
    real z12 = atom->cpu_z[n2] - atom->cpu_z[n1];
    apply_mic
    (
        atom->box.triclinic, atom->box.pbc_x, atom->box.pbc_y,
        atom->box.pbc_z, atom->box.cpu_h, x12, y12, z12
    );
    real d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    return (d12_square > (cutoff * cutoff));
}


void Hessian::find_H(Atom* atom, Force* force, Measure* measure)
{
    int N = atom->N;
    for (int nb = 0; nb < num_basis; ++nb)
    {
        int n1 = basis[nb];
        for (int n2 = 0; n2 < N; ++n2)
        {
            if(is_too_far(n1, n2, atom)) continue;
            int offset = (nb * N + n2) * 9;
            find_H12(n1, n2, atom, force, measure, H + offset);
        }
    }
}


static void find_exp_ikr
(int n1, int n2, real* k, Atom* atom, real& cos_kr, real& sin_kr)
{
    real x12 = atom->cpu_x[n2] - atom->cpu_x[n1];
    real y12 = atom->cpu_y[n2] - atom->cpu_y[n1];
    real z12 = atom->cpu_z[n2] - atom->cpu_z[n1];
    apply_mic
    (
        atom->box.triclinic, atom->box.pbc_x, atom->box.pbc_y, 
        atom->box.pbc_z, atom->box.cpu_h, x12, y12, z12
    );
    real kr = k[0] * x12 + k[1] * y12 + k[2] * z12;
    cos_kr = cos(kr);
    sin_kr = sin(kr);
}


void Hessian::output_D(char* input_dir)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/D.out");
    FILE *fid = fopen(file, "w");
    for (int nk = 0; nk < num_kpoints; ++nk)
    {
        int offset = nk * num_basis * num_basis * 9;
        for (int n1 = 0; n1 < num_basis * 3; ++n1)
        {
            for (int n2 = 0; n2 < num_basis * 3; ++n2)
            {
                // cuSOLVER requires column-major
                fprintf(fid, "%g ", DR[offset + n1 + n2 * num_basis * 3]);
            }
            if (num_kpoints > 1)
            {
                for (int n2 = 0; n2 < num_basis * 3; ++n2)
                {
                    // cuSOLVER requires column-major
                    fprintf(fid, "%g ", DI[offset + n1 + n2 * num_basis * 3]);
                }
            }
            fprintf(fid, "\n");
        }
    }
    fclose(fid);
}


void Hessian::find_omega(FILE* fid, int offset)
{
    int dim = num_basis * 3;
    double* W; MY_MALLOC(W, double, dim);
    eig_hermitian_QR(dim, DR+offset, DI+offset, W);
    double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);
    for (int n = 0; n < dim; ++n)
    {
        fprintf(fid, "%g ", W[n] * natural_to_THz);
    }
    fprintf(fid, "\n");
    MY_FREE(W);
}


void Hessian::find_omega_batch(FILE* fid)
{
    int dim = num_basis * 3;
    double* W; MY_MALLOC(W, double, dim * num_kpoints);
    eig_hermitian_Jacobi_batch(dim, num_kpoints, DR, DI, W);
    double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);
    for (int nk = 0; nk < num_kpoints; ++nk)
    {
        int offset = nk * dim;
        for (int n = 0; n < dim; ++n)
        {
            fprintf(fid, "%g ", W[offset + n] * natural_to_THz);
        }
        fprintf(fid, "\n");
    }
    MY_FREE(W);
}


void Hessian::find_D(char* input_dir, Atom* atom)
{
    char file_omega2[200];
    strcpy(file_omega2, input_dir);
    strcat(file_omega2, "/omega2.out");
    FILE *fid_omega2 = fopen(file_omega2, "w");
    for (int nk = 0; nk < num_kpoints; ++nk)
    {
        int offset = nk * num_basis * num_basis * 9;
        for (int nb = 0; nb < num_basis; ++nb)
        {
            int n1 = basis[nb];
            int label_1 = label[n1];
            real mass_1 = mass[label_1];
            for (int n2 = 0; n2 < atom->N; ++n2)
            {
                if(is_too_far(n1, n2, atom)) continue;
                real cos_kr, sin_kr;
                find_exp_ikr(n1, n2, kpoints + nk * 3, atom, cos_kr, sin_kr);
                int label_2 = label[n2];
                real mass_2 = mass[label_2];
                real mass_factor = 1.0 / sqrt(mass_1 * mass_2);
                real* H12 = H + (nb * atom->N + n2) * 9;
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        int a3b = a * 3 + b;
                        int row = label_1 * 3 + a;
                        int col = label_2 * 3 + b;
                        // cuSOLVER requires column-major
                        int index = offset + col * num_basis * 3 + row;
                        DR[index] += H12[a3b] * cos_kr * mass_factor;
                        DI[index] += H12[a3b] * sin_kr * mass_factor;
                    }
                }
            }
        }
        if (num_basis > 10) { find_omega(fid_omega2, offset); } // > 32x32
    }
    output_D(input_dir);
    if (num_basis <= 10) { find_omega_batch(fid_omega2); } // <= 32x32
    fclose(fid_omega2);
}


void Hessian::find_H12
(int n1, int n2, Atom *atom, Force *force, Measure* measure, real* H12)
{
    real dx2 = displacement * 2;
    real f_positive[3];
    real f_negative[3];
    for (int beta = 0; beta < 3; ++beta)
    {
        get_f(-displacement, n1, n2, beta, atom, force, measure, f_negative);
        get_f(displacement, n1, n2, beta, atom, force, measure, f_positive);
        for (int alpha = 0; alpha < 3; ++alpha)
        {
            int index = alpha * 3 + beta;
            H12[index] = (f_negative[alpha] - f_positive[alpha]) / dx2;
        }
    }
}


void Hessian::find_eigenvectors(char* input_dir, Atom* atom)
{
    char file_eigenvectors[200];
    strcpy(file_eigenvectors, input_dir);
    strcat(file_eigenvectors, "/eigenvector.out");
    FILE *fid_eigenvectors = my_fopen(file_eigenvectors, "w");

    int dim = num_basis * 3;
    double* W; MY_MALLOC(W, double, dim);
    double* eigenvectors; MY_MALLOC(eigenvectors, double, dim * dim);
    eigenvectors_symmetric_Jacobi(dim, DR, W, eigenvectors);

    double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION);

    // output eigenvalues
    for(int n = 0; n < dim; n++)
    {
        fprintf(fid_eigenvectors, "%g ",  W[n] * natural_to_THz);
    }
    fprintf(fid_eigenvectors, "\n");

    // output eigenvectors
    for(int col = 0; col < dim; col++)
    {
        for (int a = 0; a < 3; a++)
        {
            for(int b = 0; b < num_basis; b++)
            {
                 int row = a + b * 3;
                 // column-major order from cuSolver
                 fprintf(fid_eigenvectors, "%g ",  eigenvectors[row+col*dim]);
            }
        }
        fprintf(fid_eigenvectors, "\n");
    }

    MY_FREE(W);
    MY_FREE(eigenvectors);
    fclose(fid_eigenvectors);
}


void Hessian::get_f
(
    real dx, int n1, int n2, int beta, 
    Atom* atom, Force *force, Measure* measure, real* f
)
{
    shift_atom(dx, n2, beta, atom);
    force->compute(atom, measure);
    int M = sizeof(real);
    CHECK(cudaMemcpy(f + 0, atom->fx + n1, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(f + 1, atom->fy + n1, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(f + 2, atom->fz + n1, M, cudaMemcpyDeviceToHost));
    shift_atom(-dx, n2, beta, atom);
}


static __global__ void gpu_shift_atom(real dx, real *x)
{
    x[0] += dx;
}


void Hessian::shift_atom(real dx, int n2, int beta, Atom* atom)
{
    if (beta == 0)
    {
        gpu_shift_atom<<<1, 1>>>(dx, atom->x + n2);
        CUDA_CHECK_KERNEL
    }
    else if (beta == 1)
    {
        gpu_shift_atom<<<1, 1>>>(dx, atom->y + n2);
        CUDA_CHECK_KERNEL
    }
    else
    {
        gpu_shift_atom<<<1, 1>>>(dx, atom->z + n2);
        CUDA_CHECK_KERNEL
    }
}


void Hessian::parse_cutoff(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("cutoff should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &cutoff))
    {
        PRINT_INPUT_ERROR("cutoff for hessian should be a number.\n");
    }
    if (cutoff <= 0)
    {
        PRINT_INPUT_ERROR("cutoff for hessian should be positive.\n");
    }
    printf("Cutoff distance for hessian = %g A.\n", cutoff);
}


void Hessian::parse_delta(char **param, int num_param)
{
    if (num_param != 2)
    {
        PRINT_INPUT_ERROR("delta should have 1 parameter.\n");
    }
    if (!is_valid_real(param[1], &displacement))
    {
        PRINT_INPUT_ERROR("delta for hessian should be a number.\n");
    }
    if (displacement <= 0)
    {
        PRINT_INPUT_ERROR("delta for hessian should be positive.\n");
    }
    printf("delta for hessian = %g A.\n", displacement);
}


