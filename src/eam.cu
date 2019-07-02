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
The EAM potential. Currently two analytical versions:
[1] X. W. Zhou et al. Phys. Rev. B 69, 144113 (2004).
[2] X. D. Dai et al. JPCM 18, 4527 (2006).
------------------------------------------------------------------------------*/


#include "eam.cuh"
#include "mic.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE_FORCE 64
#define FIND_FORCE_EAM_STEP2(A, B, C, D)                                       \
    find_force_eam_step2<A, B, C, D><<<grid_size, BLOCK_SIZE_FORCE>>>          \
    (                                                                          \
        measure->hnemd.fe_x, measure->hnemd.fe_y, measure->hnemd.fe_z,         \
        eam2004zhou, eam2006dai, atom->N, N1, N2, atom->box.triclinic,         \
        atom->box.pbc_x, atom->box.pbc_y, atom->box.pbc_z, atom->NN_local,     \
        atom->NL_local, eam_data.Fp, atom->x, atom->y, atom->z, atom->vx,      \
        atom->vy, atom->vz, atom->box.h, atom->fx, atom->fy, atom->fz,         \
        atom->virial_per_atom_x, atom->virial_per_atom_y,                      \
        atom->virial_per_atom_z, atom->potential_per_atom,                     \
        atom->heat_per_atom, atom->group[0].label, measure->shc.fv_index,      \
        measure->shc.fv, measure->shc.a_map, measure->shc.b_map,               \
        measure->shc.count_b                                                   \
    ) 


EAM::EAM(FILE *fid, Atom* atom, char *name)
{

    if (strcmp(name, "eam_zhou_2004") == 0)  initialize_eam2004zhou(fid);
    if (strcmp(name, "eam_dai_2006") == 0)    initialize_eam2006dai(fid);

    // memory for the derivative of the density functional 
    CHECK(cudaMalloc((void**)&eam_data.Fp, sizeof(real) * atom->N));
}


void EAM::initialize_eam2004zhou(FILE *fid)
{
    printf("Use the EAM-type potential in the following reference:\n");
    printf("    X. W. Zhou et al., PRB 69, 144113 (2004).\n");
    potential_model = 0;

    double x[21];
    for (int n = 0; n < 21; n++)
    {
        int count = fscanf(fid, "%lf", &x[n]);
        if (count != 1)
        {print_error("reading error for potential.in.\n"); exit(1);}
    }
    eam2004zhou.re     = x[0];
    eam2004zhou.fe     = x[1];
    eam2004zhou.rho_e  = x[2];
    eam2004zhou.rho_s  = x[3];
    eam2004zhou.alpha  = x[4];
    eam2004zhou.beta   = x[5];
    eam2004zhou.A      = x[6];
    eam2004zhou.B      = x[7];
    eam2004zhou.kappa  = x[8];
    eam2004zhou.lambda = x[9];
    eam2004zhou.Fn0    = x[10];
    eam2004zhou.Fn1    = x[11];
    eam2004zhou.Fn2    = x[12];
    eam2004zhou.Fn3    = x[13];
    eam2004zhou.F0     = x[14];
    eam2004zhou.F1     = x[15];
    eam2004zhou.F2     = x[16];
    eam2004zhou.F3     = x[17];
    eam2004zhou.eta    = x[18];
    eam2004zhou.Fe     = x[19];
    eam2004zhou.rc     = x[20];
    eam2004zhou.rho_n  = eam2004zhou.rho_e * 0.85;
    eam2004zhou.rho_0  = eam2004zhou.rho_e * 1.15;
    rc                 = eam2004zhou.rc;
}


void EAM::initialize_eam2006dai(FILE *fid)
{
    printf("Use the EAM-type potential in the following reference:\n");
    printf("    X. D. Dai et al., JPCM 18, 4527 (2006).\n");
    potential_model = 1;

    double x[9];
    for (int n = 0; n < 9; n++)
    {
        int count = fscanf(fid, "%lf", &x[n]);
        if (count != 1)
        {print_error("reading error for potential.in.\n"); exit(1);}
    }
    eam2006dai.A  = x[0];
    eam2006dai.d  = x[1];
    eam2006dai.c  = x[2];
    eam2006dai.c0 = x[3];
    eam2006dai.c1 = x[4];
    eam2006dai.c2 = x[5];
    eam2006dai.c3 = x[6];
    eam2006dai.c4 = x[7];
    eam2006dai.B  = x[8];
    eam2006dai.rc = (eam2006dai.c>eam2006dai.d) ? eam2006dai.c : eam2006dai.d;
    rc            = eam2006dai.rc;
}


EAM::~EAM(void)
{
    CHECK(cudaFree(eam_data.Fp));
}


// pair function (phi and phip have been intentionally halved here)
static __device__ void find_phi
(EAM2004Zhou eam, real d12, real &phi, real &phip)
{
    real r_ratio = d12 / eam.re;
    real tmp1 = (r_ratio - eam.kappa) * (r_ratio - eam.kappa); // 2
    tmp1 *= tmp1; // 4
    tmp1 *= tmp1 * tmp1 * tmp1 * tmp1; // 20
    real tmp2 = (r_ratio - eam.lambda) * (r_ratio - eam.lambda); // 2
    tmp2 *= tmp2; // 4
    tmp2 *= tmp2 * tmp2 * tmp2 * tmp2; // 20    
    real phi1 = HALF * eam.A * exp(-eam.alpha * (r_ratio - ONE)) / (ONE + tmp1);
    real phi2 = HALF * eam.B * exp( -eam.beta * (r_ratio - ONE)) / (ONE + tmp2);
    phi = phi1 - phi2;
    phip = (phi2/eam.re)*(eam.beta+20.0*tmp2/(r_ratio-eam.lambda)/(ONE+tmp2))
         - (phi1/eam.re)*(eam.alpha+20.0*tmp1/(r_ratio-eam.kappa)/(ONE+tmp1));
}


// density function f(r)
static __device__ void find_f(EAM2004Zhou eam, real d12, real &f)
{
    real r_ratio = d12 / eam.re;
    real tmp = (r_ratio - eam.lambda) * (r_ratio - eam.lambda); // 2
    tmp *= tmp; // 4
    tmp *= tmp * tmp * tmp * tmp; // 20  
    f = eam.fe * exp(-eam.beta * (r_ratio - ONE)) / (ONE + tmp);
}


// derivative of the density function f'(r)
static __device__ void find_fp(EAM2004Zhou eam, real d12, real &fp)
{
    real r_ratio = d12 / eam.re; 
    real tmp = (r_ratio - eam.lambda) * (r_ratio - eam.lambda); // 2
    tmp *= tmp; // 4
    tmp *= tmp * tmp * tmp * tmp; // 20  
    real f = eam.fe * exp(-eam.beta * (r_ratio - ONE)) / (ONE + tmp);
    fp = -(f/eam.re)*(eam.beta+20.0*tmp/(r_ratio-eam.lambda)/(ONE+tmp));
}


// embedding function
static __device__ void find_F(EAM2004Zhou eam, real rho, real &F, real &Fp)
{      
    if (rho < eam.rho_n)
    {
        real x = rho / eam.rho_n - ONE;
        F = ((eam.Fn3 * x + eam.Fn2) * x + eam.Fn1) * x + eam.Fn0;
        Fp = ((THREE * eam.Fn3 * x + TWO * eam.Fn2) * x + eam.Fn1) / eam.rho_n;
    }
    else if (rho < eam.rho_0)
    {
        real x = rho / eam.rho_e - ONE;
        F = ((eam.F3 * x + eam.F2) * x + eam.F1) * x + eam.F0;
        Fp = ((THREE * eam.F3 * x + TWO * eam.F2) * x + eam.F1) / eam.rho_e;
    }
    else
    {
        real x = rho / eam.rho_s;
        real x_eta = pow(x, eam.eta);
        F = eam.Fe * (ONE - eam.eta * log(x)) * x_eta;
        Fp = (eam.eta / rho) * (F - eam.Fe * x_eta);
    }
}


// pair function (phi and phip have been intentionally halved here)
static __device__ void find_phi(EAM2006Dai fs, real d12, real &phi, real &phip)
{
    if (d12 > fs.c)
    {
        phi = ZERO;
        phip = ZERO;
    }
    else
    {
        real tmp=((((fs.c4*d12 + fs.c3)*d12 + fs.c2)*d12 + fs.c1)*d12 + fs.c0);
        
        phi = HALF * (d12 - fs.c) * (d12 - fs.c) * tmp;
        
        phip = TWO * (d12 - fs.c) * tmp;
        phip += (((FOUR*fs.c4*d12 + THREE*fs.c3)*d12 + TWO*fs.c2)*d12 + fs.c1)
              * (d12 - fs.c) * (d12 - fs.c);
        phip *= HALF;
    }
}


// density function f(r)
static __device__ void find_f(EAM2006Dai fs, real d12, real &f)
{
    if (d12 > fs.d)
    {
        f = ZERO;
    }
    else
    {
        real tmp = (d12 - fs.d) * (d12 - fs.d);
        f = tmp  + fs.B * fs.B * tmp * tmp;
    }
}


// derivative of the density function f'(r)
static __device__ void find_fp(EAM2006Dai fs, real d12, real &fp)
{
    if (d12 > fs.d)
    {
        fp = ZERO;
    }
    else 
    {
        real tmp = TWO * (d12 - fs.d);
        fp = tmp * (ONE + fs.B * fs.B * tmp * (d12 - fs.d));
    }
}


// embedding function
static __device__ void find_F(EAM2006Dai fs, real rho, real &F, real &Fp)
{      
    real sqrt_rho = sqrt(rho);
    F = -fs.A * sqrt_rho;
    Fp = -fs.A * HALF / sqrt_rho;
}


// Calculate the embedding energy and its derivative
template <int potential_model>
static __global__ void find_force_eam_step1
(
    EAM2004Zhou  eam2004zhou, EAM2006Dai eam2006dai, 
    int N, int N1, int N2, int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    int* g_NN, int* g_NL,
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z,
    const real* __restrict__ g_box, 
    real* g_Fp, real* g_pe 
)
{ 
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    
    if (n1 >= N1 && n1 < N2)
    {
        int NN = g_NN[n1];
           
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
          
        // Calculate the density
        real rho = ZERO;
        for (int i1 = 0; i1 < NN; ++i1)
        {      
            int n2 = g_NL[n1 + N * i1];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12); 
            real rho12 = ZERO;
            if (potential_model == 0) 
            {
                find_f(eam2004zhou, d12, rho12);
            }
            if (potential_model == 1) 
            {
                find_f(eam2006dai, d12, rho12);
            }
            rho += rho12;
        }
        
        // Calculate the embedding energy F and its derivative Fp
        real F, Fp;
        if (potential_model == 0) find_F(eam2004zhou, rho, F, Fp);
        if (potential_model == 1) find_F(eam2006dai, rho, F, Fp);

        g_pe[n1] += F; // many-body potential energy      
        g_Fp[n1] = Fp;   
    }
}


// Force evaluation kernel
template <int potential_model, int cal_j, int cal_q, int cal_k>
static __global__ void find_force_eam_step2
(
    real fe_x, real fe_y, real fe_z,
    EAM2004Zhou  eam2004zhou, EAM2006Dai eam2006dai,
    int N, int N1, int N2, int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    int *g_NN, int *g_NL,
    const real* __restrict__ g_Fp, 
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
    const real* __restrict__ g_vx, 
    const real* __restrict__ g_vy, 
    const real* __restrict__ g_vz,
    const real* __restrict__ g_box,
    real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz, real *g_pe, 
    real *g_h, int *g_label, int *g_fv_index, real *g_fv,
    int *g_a_map, int *g_b_map, int g_count_b
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    real s_fx = ZERO; // force_x
    real s_fy = ZERO; // force_y
    real s_fz = ZERO; // force_z
    real s_pe = ZERO; // potential energy
    real s_sx = ZERO; // virial_stress_x
    real s_sy = ZERO; // virial_stress_y
    real s_sz = ZERO; // virial_stress_z
    real s_h1 = ZERO; // heat_x_in
    real s_h2 = ZERO; // heat_x_out
    real s_h3 = ZERO; // heat_y_in
    real s_h4 = ZERO; // heat_y_out
    real s_h5 = ZERO; // heat_z

    // driving force 
    real fx_driving = ZERO;
    real fy_driving = ZERO;
    real fz_driving = ZERO;

    if (n1 >= N1 && n1 < N2)
    {  
        int NN = g_NN[n1];        
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real vx1, vy1, vz1;
        if (cal_j || cal_q || cal_k)
        {
            vx1 = LDG(g_vx, n1);
            vy1 = LDG(g_vy, n1); 
            vz1 = LDG(g_vz, n1);
        }
        real Fp1 = LDG(g_Fp, n1);

        for (int i1 = 0; i1 < NN; ++i1)
        {   
            int n2 = g_NL[n1 + N * i1];
            real Fp2 = LDG(g_Fp, n2);
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        
            real phi, phip, fp;
            if (potential_model == 0) 
            {
                find_phi(eam2004zhou, d12, phi, phip);
                find_fp(eam2004zhou, d12, fp);
            }
            if (potential_model == 1) 
            {
                find_phi(eam2006dai, d12, phi, phip);
                find_fp(eam2006dai, d12, fp);
            }
            phip /= d12;
            fp   /= d12;
            real f12x =  x12 * (phip + Fp1 * fp); 
            real f12y =  y12 * (phip + Fp1 * fp); 
            real f12z =  z12 * (phip + Fp1 * fp); 
            real f21x = -x12 * (phip + Fp2 * fp); 
            real f21y = -y12 * (phip + Fp2 * fp); 
            real f21z = -z12 * (phip + Fp2 * fp); 
            
            // two-body potential energy
            s_pe += phi;
 
            // per atom force
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z;  

            // driving force
            if (cal_k)
            { 
                fx_driving += f21x * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
                fy_driving += f21y * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
                fz_driving += f21z * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
            } 

            // per-atom virial stress
            s_sx -= x12 * (f12x - f21x) * HALF; 
            s_sy -= y12 * (f12y - f21y) * HALF; 
            s_sz -= z12 * (f12z - f21z) * HALF;

            // per-atom heat current
            if (cal_j || cal_k)
            {
                s_h1 += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s_h2 += (f21z * vz1) * x12;               // x-out
                s_h3 += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s_h4 += (f21z * vz1) * y12;               // y-out
                s_h5 += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            }
 
            // accumulate heat across some sections (for NEMD)
            //        check if AB pair possible & exists
            if (cal_q && g_a_map[n1] != -1 && g_b_map[n2] != -1 &&
                    g_fv_index[g_a_map[n1] * g_count_b + g_b_map[n2]] != -1)
            {
                int index_12 =
                    g_fv_index[g_a_map[n1] * g_count_b + g_b_map[n2]] * 12;
                g_fv[index_12 + 0]  += f12x;
                g_fv[index_12 + 1]  += f12y;
                g_fv[index_12 + 2]  += f12z;
                g_fv[index_12 + 3]  += f21x;
                g_fv[index_12 + 4]  += f21y;
                g_fv[index_12 + 5]  += f21z;
                g_fv[index_12 + 6]  = vx1;
                g_fv[index_12 + 7]  = vy1;
                g_fv[index_12 + 8]  = vz1;
                g_fv[index_12 + 9]  = LDG(g_vx, n2);
                g_fv[index_12 + 10] = LDG(g_vy, n2);
                g_fv[index_12 + 11] = LDG(g_vz, n2);
            }
        }

        // add driving force
        if (cal_k)
        { 
            s_fx += fx_driving;
            s_fy += fy_driving;
            s_fz += fz_driving;
        }

        // save force
        g_fx[n1] += s_fx; 
        g_fy[n1] += s_fy; 
        g_fz[n1] += s_fz;

        // accumulate virial and potential energy
        g_sx[n1] += s_sx;
        g_sy[n1] += s_sy;
        g_sz[n1] += s_sz;
        g_pe[n1] += s_pe;

        if (cal_j || cal_k) // save heat current
        {
            g_h[n1 + 0 * N] += s_h1;
            g_h[n1 + 1 * N] += s_h2;
            g_h[n1 + 2 * N] += s_h3;
            g_h[n1 + 3 * N] += s_h4;
            g_h[n1 + 4 * N] += s_h5;
        }
    }
}   


// Force evaluation wrapper
void EAM::compute(Atom *atom, Measure *measure)
{
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    find_measurement_flags(atom, measure);
    if (potential_model == 0)
    {
        find_force_eam_step1<0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            eam2004zhou, eam2006dai, atom->N, N1, N2, atom->box.triclinic, 
            atom->box.pbc_x, atom->box.pbc_y, atom->box.pbc_z, atom->NN_local, 
            atom->NL_local, atom->x, atom->y, atom->z, atom->box.h, 
            eam_data.Fp, atom->potential_per_atom
        );
        CUDA_CHECK_KERNEL
        if (compute_j)
        {
            FIND_FORCE_EAM_STEP2(0, 1, 0, 0);
        }
        else if (compute_shc && !measure->hnemd.compute)
        {
            FIND_FORCE_EAM_STEP2(0, 0, 1, 0);
        }
        else if (measure->hnemd.compute && !compute_shc)
        {
            FIND_FORCE_EAM_STEP2(0, 0, 0, 1);
        }
        else if (measure->hnemd.compute && compute_shc)
        {
            FIND_FORCE_EAM_STEP2(0, 0, 1, 1);
        }
        else
        {
            FIND_FORCE_EAM_STEP2(0, 0, 0, 0);
        }
        CUDA_CHECK_KERNEL
    }

    if (potential_model == 1)
    {
        find_force_eam_step1<1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            eam2004zhou, eam2006dai, atom->N, N1, N2, atom->box.triclinic, 
            atom->box.pbc_x, atom->box.pbc_y, atom->box.pbc_z, atom->NN_local, 
            atom->NL_local, atom->x, atom->y, atom->z, atom->box.h, 
            eam_data.Fp, atom->potential_per_atom
        );
        CUDA_CHECK_KERNEL
        if (compute_j)
        {
            FIND_FORCE_EAM_STEP2(1, 1, 0, 0);
        }
        else if (compute_shc && !measure->hnemd.compute)
        {
            FIND_FORCE_EAM_STEP2(1, 0, 1, 0);
        }
        else if (measure->hnemd.compute && !compute_shc)
        {
            FIND_FORCE_EAM_STEP2(1, 0, 0, 1);
        }
        else if (measure->hnemd.compute && compute_shc)
        {
            FIND_FORCE_EAM_STEP2(1, 0, 1, 1);
        }
        else
        {
            FIND_FORCE_EAM_STEP2(1, 0, 0, 0);
        }
        CUDA_CHECK_KERNEL
    }
}


