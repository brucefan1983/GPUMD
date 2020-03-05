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
#define FIND_FORCE_EAM_STEP2(A)                                                \
    find_force_eam_step2<A><<<grid_size, BLOCK_SIZE_FORCE>>>                   \
    (                                                                          \
        eam2004zhou, eam2006dai, atom->N, N1, N2, atom->box,                   \
        atom->NN_local,                                                        \
        atom->NL_local, eam_data.Fp, atom->x, atom->y, atom->z, atom->vx,      \
        atom->vy, atom->vz, atom->fx, atom->fy, atom->fz,                      \
        atom->virial_per_atom, atom->potential_per_atom                        \
    ) 


EAM::EAM(FILE *fid, Atom* atom, char *name)
{

    if (strcmp(name, "eam_zhou_2004") == 0)  initialize_eam2004zhou(fid);
    if (strcmp(name, "eam_dai_2006") == 0)    initialize_eam2006dai(fid);

    // memory for the derivative of the density functional 
    CHECK(cudaMalloc((void**)&eam_data.Fp, sizeof(double) * atom->N));
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
        PRINT_SCANF_ERROR(count, 1, "Reading error for EAM potential.");
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
        PRINT_SCANF_ERROR(count, 1, "Reading error for EAM potential.");
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
(EAM2004Zhou eam, double d12, double &phi, double &phip)
{
    double r_ratio = d12 / eam.re;
    double tmp1 = (r_ratio - eam.kappa) * (r_ratio - eam.kappa); // 2
    tmp1 *= tmp1; // 4
    tmp1 *= tmp1 * tmp1 * tmp1 * tmp1; // 20
    double tmp2 = (r_ratio - eam.lambda) * (r_ratio - eam.lambda); // 2
    tmp2 *= tmp2; // 4
    tmp2 *= tmp2 * tmp2 * tmp2 * tmp2; // 20    
    double phi1 = HALF * eam.A * exp(-eam.alpha * (r_ratio - ONE)) / (ONE + tmp1);
    double phi2 = HALF * eam.B * exp( -eam.beta * (r_ratio - ONE)) / (ONE + tmp2);
    phi = phi1 - phi2;
    phip = (phi2/eam.re)*(eam.beta+20.0*tmp2/(r_ratio-eam.lambda)/(ONE+tmp2))
         - (phi1/eam.re)*(eam.alpha+20.0*tmp1/(r_ratio-eam.kappa)/(ONE+tmp1));
}


// density function f(r)
static __device__ void find_f(EAM2004Zhou eam, double d12, double &f)
{
    double r_ratio = d12 / eam.re;
    double tmp = (r_ratio - eam.lambda) * (r_ratio - eam.lambda); // 2
    tmp *= tmp; // 4
    tmp *= tmp * tmp * tmp * tmp; // 20  
    f = eam.fe * exp(-eam.beta * (r_ratio - ONE)) / (ONE + tmp);
}


// derivative of the density function f'(r)
static __device__ void find_fp(EAM2004Zhou eam, double d12, double &fp)
{
    double r_ratio = d12 / eam.re; 
    double tmp = (r_ratio - eam.lambda) * (r_ratio - eam.lambda); // 2
    tmp *= tmp; // 4
    tmp *= tmp * tmp * tmp * tmp; // 20  
    double f = eam.fe * exp(-eam.beta * (r_ratio - ONE)) / (ONE + tmp);
    fp = -(f/eam.re)*(eam.beta+20.0*tmp/(r_ratio-eam.lambda)/(ONE+tmp));
}


// embedding function
static __device__ void find_F(EAM2004Zhou eam, double rho, double &F, double &Fp)
{      
    if (rho < eam.rho_n)
    {
        double x = rho / eam.rho_n - ONE;
        F = ((eam.Fn3 * x + eam.Fn2) * x + eam.Fn1) * x + eam.Fn0;
        Fp = ((THREE * eam.Fn3 * x + TWO * eam.Fn2) * x + eam.Fn1) / eam.rho_n;
    }
    else if (rho < eam.rho_0)
    {
        double x = rho / eam.rho_e - ONE;
        F = ((eam.F3 * x + eam.F2) * x + eam.F1) * x + eam.F0;
        Fp = ((THREE * eam.F3 * x + TWO * eam.F2) * x + eam.F1) / eam.rho_e;
    }
    else
    {
        double x = rho / eam.rho_s;
        double x_eta = pow(x, eam.eta);
        F = eam.Fe * (ONE - eam.eta * log(x)) * x_eta;
        Fp = (eam.eta / rho) * (F - eam.Fe * x_eta);
    }
}


// pair function (phi and phip have been intentionally halved here)
static __device__ void find_phi(EAM2006Dai fs, double d12, double &phi, double &phip)
{
    if (d12 > fs.c)
    {
        phi = ZERO;
        phip = ZERO;
    }
    else
    {
        double tmp=((((fs.c4*d12 + fs.c3)*d12 + fs.c2)*d12 + fs.c1)*d12 + fs.c0);
        
        phi = HALF * (d12 - fs.c) * (d12 - fs.c) * tmp;
        
        phip = TWO * (d12 - fs.c) * tmp;
        phip += (((FOUR*fs.c4*d12 + THREE*fs.c3)*d12 + TWO*fs.c2)*d12 + fs.c1)
              * (d12 - fs.c) * (d12 - fs.c);
        phip *= HALF;
    }
}


// density function f(r)
static __device__ void find_f(EAM2006Dai fs, double d12, double &f)
{
    if (d12 > fs.d)
    {
        f = ZERO;
    }
    else
    {
        double tmp = (d12 - fs.d) * (d12 - fs.d);
        f = tmp  + fs.B * fs.B * tmp * tmp;
    }
}


// derivative of the density function f'(r)
static __device__ void find_fp(EAM2006Dai fs, double d12, double &fp)
{
    if (d12 > fs.d)
    {
        fp = ZERO;
    }
    else 
    {
        double tmp = TWO * (d12 - fs.d);
        fp = tmp * (ONE + fs.B * fs.B * tmp * (d12 - fs.d));
    }
}


// embedding function
static __device__ void find_F(EAM2006Dai fs, double rho, double &F, double &Fp)
{      
    double sqrt_rho = sqrt(rho);
    F = -fs.A * sqrt_rho;
    Fp = -fs.A * HALF / sqrt_rho;
}


// Calculate the embedding energy and its derivative
template <int potential_model>
static __global__ void find_force_eam_step1
(
    EAM2004Zhou  eam2004zhou, EAM2006Dai eam2006dai, 
    int N, int N1, int N2, Box box, 
    int* g_NN, int* g_NL,
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z, 
    double* g_Fp, double* g_pe 
)
{ 
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    
    if (n1 >= N1 && n1 < N2)
    {
        int NN = g_NN[n1];
           
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
          
        // Calculate the density
        double rho = ZERO;
        for (int i1 = 0; i1 < NN; ++i1)
        {      
            int n2 = g_NL[n1 + N * i1];
            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12); 
            double rho12 = ZERO;
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
        double F, Fp;
        if (potential_model == 0) find_F(eam2004zhou, rho, F, Fp);
        if (potential_model == 1) find_F(eam2006dai, rho, F, Fp);

        g_pe[n1] += F; // many-body potential energy      
        g_Fp[n1] = Fp;   
    }
}


// Force evaluation kernel
template <int potential_model>
static __global__ void find_force_eam_step2
(
    EAM2004Zhou  eam2004zhou, EAM2006Dai eam2006dai,
    int N, int N1, int N2, Box box, 
    int *g_NN, int *g_NL,
    const double* __restrict__ g_Fp, 
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z, 
    const double* __restrict__ g_vx, 
    const double* __restrict__ g_vy, 
    const double* __restrict__ g_vz,
    double *g_fx, double *g_fy, double *g_fz,
    double *g_virial, double *g_pe
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    double s_fx = ZERO; // force_x
    double s_fy = ZERO; // force_y
    double s_fz = ZERO; // force_z
    double s_pe = ZERO; // potential energy
    double s_sxx = ZERO; // virial_stress_xx
    double s_sxy = ZERO; // virial_stress_xy
    double s_sxz = ZERO; // virial_stress_xz
    double s_syx = ZERO; // virial_stress_yx
    double s_syy = ZERO; // virial_stress_yy
    double s_syz = ZERO; // virial_stress_yz
    double s_szx = ZERO; // virial_stress_zx
    double s_szy = ZERO; // virial_stress_zy
    double s_szz = ZERO; // virial_stress_zz

    if (n1 >= N1 && n1 < N2)
    {  
        int NN = g_NN[n1];        
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        double Fp1 = LDG(g_Fp, n1);

        for (int i1 = 0; i1 < NN; ++i1)
        {   
            int n2 = g_NL[n1 + N * i1];
            double Fp2 = LDG(g_Fp, n2);
            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        
            double phi, phip, fp;
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
            double f12x =  x12 * (phip + Fp1 * fp); 
            double f12y =  y12 * (phip + Fp1 * fp); 
            double f12z =  z12 * (phip + Fp1 * fp); 
            double f21x = -x12 * (phip + Fp2 * fp); 
            double f21y = -y12 * (phip + Fp2 * fp); 
            double f21z = -z12 * (phip + Fp2 * fp); 
            
            // two-body potential energy
            s_pe += phi;
 
            // per atom force
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z;  

            // per-atom virial
            s_sxx += x12 * f21x;
            s_sxy += x12 * f21y;
            s_sxz += x12 * f21z;
            s_syx += y12 * f21x;
            s_syy += y12 * f21y;
            s_syz += y12 * f21z;
            s_szx += z12 * f21x;
            s_szy += z12 * f21y;
            s_szz += z12 * f21z;
        }

        // save force
        g_fx[n1] += s_fx; 
        g_fy[n1] += s_fy; 
        g_fz[n1] += s_fz;

        // save virial
        // xx xy xz    0 3 4
        // yx yy yz    6 1 5
        // zx zy zz    7 8 2
        g_virial[n1 + 0 * N] += s_sxx;
        g_virial[n1 + 1 * N] += s_syy;
        g_virial[n1 + 2 * N] += s_szz;
        g_virial[n1 + 3 * N] += s_sxy;
        g_virial[n1 + 4 * N] += s_sxz;
        g_virial[n1 + 5 * N] += s_syz;
        g_virial[n1 + 6 * N] += s_syx;
        g_virial[n1 + 7 * N] += s_szx;
        g_virial[n1 + 8 * N] += s_szy;

        // save potential energy
        g_pe[n1] += s_pe;
    }
}   


// Force evaluation wrapper
void EAM::compute(Atom *atom, Measure *measure, int potential_number)
{
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

    if (potential_model == 0)
    {
        find_force_eam_step1<0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            eam2004zhou, eam2006dai, atom->N, N1, N2, atom->box, 
            atom->NN_local, atom->NL_local, atom->x, atom->y, atom->z, 
            eam_data.Fp, atom->potential_per_atom
        );
        CUDA_CHECK_KERNEL

        FIND_FORCE_EAM_STEP2(0);
        CUDA_CHECK_KERNEL
    }

    if (potential_model == 1)
    {
        find_force_eam_step1<1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            eam2004zhou, eam2006dai, atom->N, N1, N2, atom->box, 
            atom->NN_local, atom->NL_local, atom->x, atom->y, atom->z, 
            eam_data.Fp, atom->potential_per_atom
        );
        CUDA_CHECK_KERNEL
        
        FIND_FORCE_EAM_STEP2(1);
        CUDA_CHECK_KERNEL
    }
}


