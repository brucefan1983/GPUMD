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
Green-Kubo Modal Analysis (GKMA) and
Homogenous Nonequilibrium Modal Analysis (HNEMA) implementations.

Original GMKA method is detailed in:
H.R. Seyf, K. Gordiz, F. DeAngelis, and A. Henry, "Using Green-Kubo modal
analysis (GKMA) and interface conductance modal analysis (ICMA) to study
phonon transport with molecular dynamics," J. Appl. Phys., 125, 081101 (2019).

The code here is inspired by the LAMMPS implementation provided by the Henry
group at MIT. This code can be found:
https://drive.google.com/open?id=1IHJ7x-bLZISX3I090dW_Y_y-Mqkn07zg

GPUMD Contributing author: Alexander Gabourie (Stanford University)
------------------------------------------------------------------------------*/

#include "modal_analysis.cuh"
#include "error.cuh"
#include <cublas_v2.h>


#define NUM_OF_HEAT_COMPONENTS 5
#define BLOCK_SIZE 128
#define ACCUM_BLOCK 1024
#define BIN_BLOCK 128
#define BLOCK_SIZE_FORCE 64
#define BLOCK_SIZE_GK 16

#define ACCUMULATE 0
#define SET 1

static __global__ void gpu_reset_data
(
        int num_elements, float* data
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        data[n] = 0.0f;
    }
}

static __global__ void gpu_scale_jm
(
        int num_elements, float factor, float* jm
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        jm[n]*=factor;
    }
}


static __device__ void gpu_bin_reduce
(
       int num_modes, int bin_size, int shift, int num_bins,
       int tid, int bid, int number_of_patches,
       const float* __restrict__ g_jm,
       float* bin_out
)
{
    __shared__ float s_data_xin[BIN_BLOCK];
    __shared__ float s_data_xout[BIN_BLOCK];
    __shared__ float s_data_yin[BIN_BLOCK];
    __shared__ float s_data_yout[BIN_BLOCK];
    __shared__ float s_data_z[BIN_BLOCK];
    s_data_xin[tid] = 0.0f;
    s_data_xout[tid] = 0.0f;
    s_data_yin[tid] = 0.0f;
    s_data_yout[tid] = 0.0f;
    s_data_z[tid] = 0.0f;

    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * BIN_BLOCK;
        if (n < bin_size)
        {
            s_data_xin[tid] += g_jm[n + shift];
            s_data_xout[tid] += g_jm[n + shift + num_modes];
            s_data_yin[tid] += g_jm[n + shift + 2*num_modes];
            s_data_yout[tid] += g_jm[n + shift + 3*num_modes];
            s_data_z[tid] += g_jm[n + shift + 4*num_modes];
        }
    }

    __syncthreads();
    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_data_xin[tid] += s_data_xin[tid + offset];
            s_data_xout[tid] += s_data_xout[tid + offset];
            s_data_yin[tid] += s_data_yin[tid + offset];
            s_data_yout[tid] += s_data_yout[tid + offset];
            s_data_z[tid] += s_data_z[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        bin_out[bid] = s_data_xin[0];
        bin_out[bid + num_bins] = s_data_xout[0];
        bin_out[bid + 2*num_bins] = s_data_yin[0];
        bin_out[bid + 3*num_bins] = s_data_yout[0];
        bin_out[bid + 4*num_bins] = s_data_z[0];
    }
}

static __global__ void gpu_bin_modes
(
       int num_modes,
       const int* __restrict__ bin_count,
       const int* __restrict__ bin_sum,
       int num_bins,
       const float* __restrict__ g_jm,
       float* bin_out
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bin_size = bin_count[bid];
    int shift = bin_sum[bid];
    int number_of_patches = (bin_size - 1) / BIN_BLOCK + 1;

    gpu_bin_reduce
    (
           num_modes, bin_size, shift, num_bins,
           tid, bid, number_of_patches, g_jm, bin_out
    );

}

static __global__ void elemwise_mass_scale
(
        int num_participating, int N1,
        const float* __restrict__ g_sqrtmass,
        const double* __restrict__ g_vx,
        const double* __restrict__ g_vy,
        const double* __restrict__ g_vz,
        float* g_mv_x, float* g_mv_y, float* g_mv_z
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nglobal = i + N1;
    if (i < num_participating)
    {
        float sqrtmass = g_sqrtmass[i];
        float vx, vy, vz;
        vx = __double2float_rn(LDG(g_vx, nglobal));
        vy = __double2float_rn(LDG(g_vy, nglobal));
        vz = __double2float_rn(LDG(g_vz, nglobal));
        g_mv_x[i] = sqrtmass*vx;
        g_mv_y[i] = sqrtmass*vy;
        g_mv_z[i] = sqrtmass*vz;
    }
}

static __global__ void gpu_set_mass_terms
(
    int num_participating, int N1,
    const double* __restrict__ g_mass,
    float* sqrtmass,
    float* rsqrtmass
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nglobal = i + N1;
    if (i < num_participating)
    {
        float mass = __double2float_rn(LDG(g_mass, nglobal));
        sqrtmass[i] = sqrt(mass);
        rsqrtmass[i] = rsqrt(mass);
    }
}

static __global__ void prepare_sm
(
        int num_participating, int N1,
        const double* __restrict__ sxx,
        const double* __restrict__ sxy,
        const double* __restrict__ sxz,
        const double* __restrict__ syx,
        const double* __restrict__ syy,
        const double* __restrict__ syz,
        const double* __restrict__ szx,
        const double* __restrict__ szy,
        const double* __restrict__ szz,
        const float* __restrict__ rsqrtmass,
        float* smx, float* smy, float* smz
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nglobal = i + N1;
    if (i < num_participating)
    {
        float invmass = rsqrtmass[i];
        // x's
        smx[i] = __double2float_rn(sxx[nglobal])*invmass;
        smx[i + num_participating] =
                __double2float_rn(syx[nglobal])*invmass;
        smx[i + 2*num_participating] =
                        __double2float_rn(szx[nglobal])*invmass;

        // y's
        smy[i] = __double2float_rn(sxy[nglobal])*invmass;
        smy[i + num_participating] =
                        __double2float_rn(syy[nglobal])*invmass;
        smy[i + 2*num_participating] =
                        __double2float_rn(szy[nglobal])*invmass;

        // z's
        smz[i] = __double2float_rn(sxz[nglobal])*invmass;
        smz[i + num_participating] =
                        __double2float_rn(syz[nglobal])*invmass;
        smz[i + 2*num_participating] =
                        __double2float_rn(szz[nglobal])*invmass;

    }
}

template <int operate>
static __global__ void gpu_update_jm
(
    int num_modes,
    const float* __restrict__ jmx,
    const float* __restrict__ jmy,
    const float* __restrict__ jmz,
    float* jm
)
{
    int mode = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode < num_modes)
    {
        int yidx = mode + num_modes;
        int zidx = mode + 2*num_modes;

        if (operate == SET)
        {
            jm[mode] = jmx[mode] + jmy[mode]; // jxi
            jm[mode + num_modes] = jmz[mode]; // jxo
            jm[mode + 2*num_modes] = jmx[yidx] + jmy[yidx]; // jyi
            jm[mode + 3*num_modes] = jmz[yidx];             // jyo
            jm[mode + 4*num_modes] = jmx[zidx]+jmy[zidx]+jmz[zidx]; // jz
        }
        if (operate == ACCUMULATE)
        {
            jm[mode] += jmx[mode] + jmy[mode]; // jxi
            jm[mode + num_modes] += jmz[mode]; // jxo
            jm[mode + 2*num_modes] += jmx[yidx] + jmy[yidx]; // jyi
            jm[mode + 3*num_modes] += jmz[yidx];             // jyo
            jm[mode + 4*num_modes] += jmx[zidx]+jmy[zidx]+jmz[zidx]; // jz
        }

    }
}


void MODAL_ANALYSIS::compute_heat(Atom *atom)
{

    int grid_size = (num_participating - 1) / BLOCK_SIZE + 1;
    elemwise_mass_scale<<<grid_size, BLOCK_SIZE>>>
    (
          num_participating, N1, sqrtmass,
          atom->vx, atom->vy, atom->vz,
          mv_x, mv_y, mv_z
    );
    CUDA_CHECK_KERNEL

    prepare_sm<<<grid_size, BLOCK_SIZE>>>
    (
           num_participating, N1,
           atom->virial_per_atom,
           atom->virial_per_atom + atom->N * 3,
           atom->virial_per_atom + atom->N * 4,
           atom->virial_per_atom + atom->N * 6,
           atom->virial_per_atom + atom->N * 1,
           atom->virial_per_atom + atom->N * 5,
           atom->virial_per_atom + atom->N * 7,
           atom->virial_per_atom + atom->N * 8,
           atom->virial_per_atom + atom->N * 2,
           rsqrtmass, smx, smy, smz
    );
    CUDA_CHECK_KERNEL

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0;
    float beta = 0.0;
    int stride = 1;

    cublasSgemv(handle, CUBLAS_OP_N, num_modes, num_participating,
            &alpha, eig_x, num_modes, mv_x, stride, &beta, xdot_x, stride);
    cublasSgemv(handle, CUBLAS_OP_N, num_modes, num_participating,
            &alpha, eig_y, num_modes, mv_y, stride, &beta, xdot_y, stride);
    cublasSgemv(handle, CUBLAS_OP_N, num_modes, num_participating,
            &alpha, eig_z, num_modes, mv_z, stride, &beta, xdot_z, stride);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_modes, 3, num_participating,
        &alpha, eig_x, num_modes, smx, num_participating, &beta, jmx, num_modes);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_modes, 3, num_participating,
        &alpha, eig_y, num_modes, smy, num_participating, &beta, jmy, num_modes);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_modes, 3, num_participating,
        &alpha, eig_z, num_modes, smz, num_participating, &beta, jmz, num_modes);

    cublasSdgmm(handle, CUBLAS_SIDE_LEFT, num_modes, 3, jmx, num_modes,
            xdot_x, stride, jmx, num_modes);
    cublasSdgmm(handle, CUBLAS_SIDE_LEFT, num_modes, 3, jmy, num_modes,
            xdot_y, stride, jmy, num_modes);
    cublasSdgmm(handle, CUBLAS_SIDE_LEFT, num_modes, 3, jmz, num_modes,
            xdot_z, stride, jmz, num_modes);

    cublasDestroy(handle);

//    cudaDeviceSynchronize();
//    float jxi = 0;
//    float jxo = 0;
//    for (int i = 0; i < num_modes; i++)
//    {
//        jxi += jmx[i] + jmy[i];
//        jxo += jmz[i];
//    }
//    printf("jxi = %g, jxo = %g\n", jxi, jxo);


    grid_size = (num_modes - 1) / BLOCK_SIZE + 1;
    if (method == GKMA_METHOD)
    {
        gpu_update_jm<SET><<<grid_size, BLOCK_SIZE>>>
        (
            num_modes, jmx, jmy, jmz, jm
        );
    }
    else if (method == HNEMA_METHOD)
    {
        gpu_update_jm<ACCUMULATE><<<grid_size, BLOCK_SIZE>>>
        (
            num_modes, jmx, jmy, jmz, jm
        );
    }
    CUDA_CHECK_KERNEL

}

void MODAL_ANALYSIS::setN(Atom *atom)
{
    N1 = 0;
    N2 = 0;
    for (int n = 0; n < atom_begin; ++n)
    {
        N1 += atom->cpu_type_size[n];
    }
    for (int n = atom_begin; n <= atom_end; ++n)
    {
        N2 += atom->cpu_type_size[n];
    }

    num_participating = N2 - N1;
}

void MODAL_ANALYSIS::set_eigmode(int mode, std::ifstream &eigfile, float* eig)
{
    float floatval;
    for (int i=0; i<num_participating; i++)
    {
        eigfile >> floatval;
        // column major ordering for cuBLAS
        eig[mode + i*num_modes] = floatval;
    }
}

void MODAL_ANALYSIS::preprocess(char *input_dir, Atom *atom)
{
    if (!compute) return;
        num_modes = last_mode-first_mode+1;
        samples_per_output = output_interval/sample_interval;
        setN(atom);

        strcpy(output_file_position, input_dir);
        if (method == GKMA_METHOD)
        {
            strcat(output_file_position, "/heatmode.out");
        }
        else if (method == HNEMA_METHOD)
        {
            strcat(output_file_position, "/kappamode.out");
        }

        size_t eig_size = sizeof(float) * num_participating * num_modes;
        CHECK(cudaMallocManaged((void **)&eig_x,eig_size));
        CHECK(cudaMallocManaged((void **)&eig_y,eig_size));
        CHECK(cudaMallocManaged((void **)&eig_z,eig_size));

        // initialize eigenvector data structures
        strcpy(eig_file_position, input_dir);
        strcat(eig_file_position, "/eigenvector.out"); //TODO change to .in
        std::ifstream eigfile;
        eigfile.open(eig_file_position);
        if (!eigfile)
        {
            PRINT_INPUT_ERROR("Cannot open eigenvector.out file.");
        }

        // GPU phonon code output format
        std::string val;

        // Setup binning
        if (f_flag)
        {
            double *f;
            CHECK(cudaMallocManaged((void **)&f, sizeof(double)*num_modes));
            getline(eigfile, val);
            std::stringstream ss(val);
            for (int i=0; i<first_mode-1; i++) { ss >> f[0]; }
            double temp;
            for (int i=0; i<num_modes; i++)
            {
                ss >> temp;
                f[i] = copysign(sqrt(abs(temp))/(2.0*PI), temp);
            }
            double fmax, fmin; // freq are in ascending order in file
            int shift;
            fmax = (floor(abs(f[num_modes-1])/f_bin_size)+1)*f_bin_size;
            fmin = floor(abs(f[0])/f_bin_size)*f_bin_size;
            shift = floor(abs(fmin)/f_bin_size);
            num_bins = floor((fmax-fmin)/f_bin_size);

            size_t bin_count_size = sizeof(int)*num_bins;
            CHECK(cudaMallocManaged((void **)&bin_count, bin_count_size));
            for(int i_=0; i_<num_bins;i_++){bin_count[i_]=(int)0;}

            for (int i = 0; i< num_modes; i++)
                bin_count[int(abs(f[i]/f_bin_size))-shift]++;

            size_t bin_sum_size = sizeof(int)*num_bins;
            CHECK(cudaMallocManaged((void **)&bin_sum, bin_sum_size));
            for(int i_=0; i_<num_bins;i_++){bin_sum[i_]=(int)0;}

            for (int i = 1; i < num_bins; i++)
                bin_sum[i] = bin_sum[i-1] + bin_count[i-1];

            CHECK(cudaFree(f));
        }
        else
        {
            // TODO validate this section
            num_bins = (int)ceil((double)num_modes/(double)bin_size);
            size_t bin_count_size = sizeof(int)*num_bins;
            CHECK(cudaMallocManaged((void **)&bin_count, bin_count_size));
            for(int i_=0; i_<num_bins-1;i_++){bin_count[i_]=(int)bin_size;}
            bin_count[num_bins-1] = num_modes%bin_size;

            size_t bin_sum_size = sizeof(int)*num_bins;
            CHECK(cudaMallocManaged((void **)&bin_sum, bin_sum_size));
            for(int i_=0; i_<num_bins;i_++){bin_sum[i_]=(int)0;}

            for (int i = 1; i < num_bins; i++)
                bin_sum[i] = bin_sum[i-1] + bin_count[i-1];

            getline(eigfile,val);
        }

        // skips modes up to first_mode
        for (int i=1; i<first_mode; i++) { getline(eigfile,val); }
        for (int j=0; j<num_modes; j++) //modes
        {
            set_eigmode(j, eigfile, eig_x);
            set_eigmode(j, eigfile, eig_y);
            set_eigmode(j, eigfile, eig_z);
        }
        eigfile.close();

        // Allocate intermediate vector
        size_t mv_n_size = sizeof(float) * num_participating;
        CHECK(cudaMallocManaged((void **)&mv_x, mv_n_size));
        CHECK(cudaMallocManaged((void **)&mv_y, mv_n_size));
        CHECK(cudaMallocManaged((void **)&mv_z, mv_n_size));

        // Allocate modal velocities
        size_t xdot_size = sizeof(float) * num_modes;
        CHECK(cudaMallocManaged((void **)&xdot_x, xdot_size));
        CHECK(cudaMallocManaged((void **)&xdot_y, xdot_size));
        CHECK(cudaMallocManaged((void **)&xdot_z, xdot_size));

        // Allocate modal measured quantities
        size_t jmxyz_size = sizeof(float) * num_modes*3;
        CHECK(cudaMallocManaged((void **)&jmx,jmxyz_size));
        CHECK(cudaMallocManaged((void **)&jmy,jmxyz_size));
        CHECK(cudaMallocManaged((void **)&jmz,jmxyz_size));

        num_heat_stored = num_modes*NUM_OF_HEAT_COMPONENTS;
        size_t jm_size = sizeof(float) * num_heat_stored;
        CHECK(cudaMallocManaged((void **)&jm,jm_size));

        if (method == HNEMA_METHOD)
        {
            int grid_size = (num_heat_stored-1)/BLOCK_SIZE+1;
            gpu_reset_data<<<grid_size,BLOCK_SIZE>>>(num_heat_stored, jm);
            CUDA_CHECK_KERNEL
        }

        size_t bin_out_size = sizeof(float) * num_bins * NUM_OF_HEAT_COMPONENTS;
        CHECK(cudaMallocManaged((void **)&bin_out, bin_out_size));

        size_t sm_size = sizeof(float) * num_modes * 3;
        CHECK(cudaMallocManaged((void **)&smx, sm_size));
        CHECK(cudaMallocManaged((void **)&smy, sm_size));
        CHECK(cudaMallocManaged((void **)&smz, sm_size));

        // prepare masses
        size_t mass_size = sizeof(float) * num_participating;
        CHECK(cudaMallocManaged((void **)&sqrtmass, mass_size));
        CHECK(cudaMallocManaged((void **)&rsqrtmass, mass_size));
        gpu_set_mass_terms
        <<<(num_participating-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
        (
            num_participating, N1, atom->mass, sqrtmass, rsqrtmass
        );
        CUDA_CHECK_KERNEL
}

void MODAL_ANALYSIS::process(int step, Atom *atom, Integrate *integrate, double fe)
{
    if (!compute) return;
    if (!((step+1) % sample_interval == 0)) return;

    compute_heat(atom);

    if (method == HNEMA_METHOD &&
            !((step+1) % output_interval == 0)) return;


    gpu_bin_modes<<<num_bins, BIN_BLOCK>>>
    (
           num_modes, bin_count, bin_sum, num_bins,
           jm, bin_out
    );
    CUDA_CHECK_KERNEL

    if (method == HNEMA_METHOD)
    {
        float volume = atom->box.get_volume();
        float factor = KAPPA_UNIT_CONVERSION/
            (volume * integrate->ensemble->temperature
                    * fe * (float)samples_per_output);
        int num_bins_stored = num_bins * NUM_OF_HEAT_COMPONENTS;
        gpu_scale_jm<<<(num_bins_stored-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
        (
            num_bins_stored, factor, bin_out
        );
        CUDA_CHECK_KERNEL
    }

    // Compute thermal conductivity and output
    cudaDeviceSynchronize(); // ensure GPU ready to move data to CPU
    FILE *fid = fopen(output_file_position, "a");
    for (int i = 0; i < num_bins; i++)
    {
        fprintf(fid, "%g %g %g %g %g\n",
                bin_out[i], bin_out[i+num_bins], bin_out[i+2*num_bins],
                         bin_out[i+3*num_bins], bin_out[i+4*num_bins]);
    }
    fflush(fid);
    fclose(fid);

    if (method == HNEMA_METHOD)
    {
        int grid_size = (num_heat_stored-1)/BLOCK_SIZE+1;
        gpu_reset_data<<<grid_size,BLOCK_SIZE>>>(num_heat_stored, jm);
        CUDA_CHECK_KERNEL
    }

}


void MODAL_ANALYSIS::postprocess()
{
    if (!compute) return;
    CHECK(cudaFree(eig_x));
    CHECK(cudaFree(eig_y));
    CHECK(cudaFree(eig_z));

    CHECK(cudaFree(xdot_x));
    CHECK(cudaFree(xdot_y));
    CHECK(cudaFree(xdot_z));

    CHECK(cudaFree(jmx));
    CHECK(cudaFree(jmy));
    CHECK(cudaFree(jmz));
    CHECK(cudaFree(jm));

    CHECK(cudaFree(smx));
    CHECK(cudaFree(smy));
    CHECK(cudaFree(smz));

    CHECK(cudaFree(mv_x));
    CHECK(cudaFree(mv_y));
    CHECK(cudaFree(mv_z));

    CHECK(cudaFree(bin_out));
    CHECK(cudaFree(bin_count));
    CHECK(cudaFree(bin_sum));
}
