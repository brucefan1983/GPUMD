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


#include "vashishta.cuh"
#include "mic.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE_VASHISHTA 64


/*----------------------------------------------------------------------------80
    Reference: 
        P. Vashishta et al., J. Appl. Phys. 101, 103515 (2007).
*-----------------------------------------------------------------------------*/


void Vashishta::initialize_para(FILE *fid)
{
    printf("Use Vashishta potential.\n");
    int count;

    double B_0, B_1, cos0_0, cos0_1, C, r0, cut;
    count = fscanf
    (fid, "%lf%lf%lf%lf%lf%lf%lf", &B_0, &B_1, &cos0_0, &cos0_1, &C, &r0, &cut);
    PRINT_SCANF_ERROR(count, 7, "Reading error for Vashishta potential.");
    vashishta_para.B[0] = B_0;
    vashishta_para.B[1] = B_1;
    vashishta_para.cos0[0] = cos0_0;
    vashishta_para.cos0[1] = cos0_1;
    vashishta_para.C = C;
    vashishta_para.r0 = r0;
    vashishta_para.rc = cut;
    rc = cut;

    double H[3], qq[3], lambda_inv[3], D[3], xi_inv[3], W[3];
    int eta[3];
    for (int n = 0; n < 3; n++)
    {
        count = fscanf
        (
            fid, "%lf%d%lf%lf%lf%lf%lf", 
            &H[n], &eta[n], &qq[n], &lambda_inv[n], &D[n], &xi_inv[n], &W[n]
        );
        PRINT_SCANF_ERROR(count, 7, "Reading error for Vashishta potential.");
        qq[n] *= K_C;         // Gauss -> SI
        D[n] *= (K_C * 0.5); // Gauss -> SI and D -> D/2
        lambda_inv[n] = 1.0 / lambda_inv[n];
        xi_inv[n] = 1.0 / xi_inv[n];

        vashishta_para.H[n] = H[n];
        vashishta_para.eta[n] = eta[n];
        vashishta_para.qq[n] = qq[n];
        vashishta_para.lambda_inv[n] = lambda_inv[n];
        vashishta_para.D[n] = D[n];
        vashishta_para.xi_inv[n] = xi_inv[n];
        vashishta_para.W[n] = W[n];

        double rci = 1.0 / rc;
        double rci4 = rci * rci * rci * rci;
        double rci6 = rci4 * rci * rci;
        double p2_steric = H[n] * pow(rci, double(eta[n]));
        double p2_charge = qq[n] * rci * exp(-rc*lambda_inv[n]);
        double p2_dipole = D[n] * rci4 * exp(-rc*xi_inv[n]);
        double p2_vander = W[n] * rci6;
        vashishta_para.v_rc[n] = p2_steric+p2_charge-p2_dipole-p2_vander;
        vashishta_para.dv_rc[n] = p2_dipole * (xi_inv[n] + 4.0 * rci) 
                                + p2_vander * (6.0 * rci)
                                - p2_charge * (lambda_inv[n] + rci)
                                - p2_steric * (eta[n] * rci);
    }
}


Vashishta::Vashishta(FILE *fid, Atom* atom)
{
    initialize_para(fid);
    int num = (atom->neighbor.MN < 100) ? atom->neighbor.MN : 100;
    vashishta_data.f12x.resize(atom->N * num);
    vashishta_data.f12y.resize(atom->N * num);
    vashishta_data.f12z.resize(atom->N * num);
    vashishta_data.NN_short.resize(atom->N);
    vashishta_data.NL_short.resize(atom->N * num);
}


Vashishta::~Vashishta(void)
{
    // nothing
}


// eta is always an integer and we don't need the very slow pow()
static __device__ double my_pow(double x, int n) 
{
    if (n == 7) 
    { 
        double y = x;
        x *= x;
        y *= x; // x^3
        x *= x; // x^4
        return y * x;
    }
    else if (n == 9) 
    { 
        double y = x;
        x *= x; // x^2
        x *= x; // x^4
        y *= x; // x^5
        return y * x; 
    }
    else // n = 11
    { 
        double y = x;
        x *= x; // x^2
        y *= x; // x^3
        x *= x; // x^4
        x *= x; // x^8
        return y * x; 
    }
}


// get U_ij and (d U_ij / d r_ij) / r_ij for the 2-body part
static __device__ void find_p2_and_f2
(
    double H, int eta, double qq, double lambda_inv, double D, double xi_inv, double W, 
    double v_rc, double dv_rc, double rc, double d12, double &p2, double &f2
)
{
    double d12inv = 1.0 / d12;
    double d12inv2 = d12inv * d12inv;
    double p2_steric = H * my_pow(d12inv, eta);
    double p2_charge = qq * d12inv * exp(-d12 * lambda_inv);
    double p2_dipole = D * (d12inv2 * d12inv2) * exp(-d12 * xi_inv);
    double p2_vander = W * (d12inv2 * d12inv2 * d12inv2);
    p2 = p2_steric + p2_charge - p2_dipole - p2_vander; 
    p2 -= v_rc + (d12 - rc) * dv_rc; // shifted potential
    f2 = p2_dipole * (xi_inv + 4.0*d12inv) + p2_vander * (6.0 * d12inv);
    f2 -= p2_charge * (lambda_inv + d12inv) + p2_steric * (eta * d12inv);
    f2 = (f2 - dv_rc) * d12inv;      // shifted force
}


// 2-body part of the Vashishta potential (kernel)
static __global__ void gpu_find_force_vashishta_2body
(
    int number_of_particles, int N1, int N2, Box box, 
    Vashishta_Para vas,
    int *g_NN, int *g_NL, int *g_NN_local, int *g_NL_local,
    int *g_type, int shift,
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z, 
    double *g_fx, double *g_fy, double *g_fz,
    double *g_virial, double *g_potential
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    double s_fx = 0.0; // force_x
    double s_fy = 0.0; // force_y
    double s_fz = 0.0; // force_z
    double s_pe = 0.0; // potential energy
    double s_sxx = 0.0; // virial_stress_xx
    double s_sxy = 0.0; // virial_stress_xy
    double s_sxz = 0.0; // virial_stress_xz
    double s_syx = 0.0; // virial_stress_yx
    double s_syy = 0.0; // virial_stress_yy
    double s_syz = 0.0; // virial_stress_yz
    double s_szx = 0.0; // virial_stress_zx
    double s_szy = 0.0; // virial_stress_zy
    double s_szz = 0.0; // virial_stress_zz

    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1] - shift;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        
        int count = 0; // initialize g_NN_local[n1] to 0

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = g_NL[n1 + number_of_particles * i1];
            
            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            if (d12 >= vas.rc) { continue; }
            if (d12 < vas.r0) // r0 is much smaller than rc
            {                    
                g_NL_local[n1 + number_of_particles * (count++)] = n2;
            }
            int type2 = g_type[n2] - shift;
            int type12 = type1 + type2; // 0 = AA; 1 = AB or BA; 2 = BB
            double p2, f2;

            find_p2_and_f2
            (
                vas.H[type12], vas.eta[type12], vas.qq[type12], 
                vas.lambda_inv[type12], vas.D[type12], vas.xi_inv[type12],
                vas.W[type12], vas.v_rc[type12], vas.dv_rc[type12], 
                vas.rc, d12, p2, f2
            );

            // treat two-body potential in the same way as many-body potential
            double f12x = f2 * x12 * 0.5; 
            double f12y = f2 * y12 * 0.5; 
            double f12z = f2 * z12 * 0.5; 
            double f21x = -f12x; 
            double f21y = -f12y; 
            double f21z = -f12z; 
       
            // accumulate force
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z;
            
            // accumulate potential energy and virial
            s_pe += p2 * 0.5; // two-body potential
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

        g_NN_local[n1] = count; // now the local neighbor list has been built

        g_fx[n1] += s_fx; // save force
        g_fy[n1] += s_fy;
        g_fz[n1] += s_fz;

        // save virial
        // xx xy xz    0 3 4
        // yx yy yz    6 1 5
        // zx zy zz    7 8 2
        g_virial[n1 + 0 * number_of_particles] += s_sxx;
        g_virial[n1 + 1 * number_of_particles] += s_syy;
        g_virial[n1 + 2 * number_of_particles] += s_szz;
        g_virial[n1 + 3 * number_of_particles] += s_sxy;
        g_virial[n1 + 4 * number_of_particles] += s_sxz;
        g_virial[n1 + 5 * number_of_particles] += s_syz;
        g_virial[n1 + 6 * number_of_particles] += s_syx;
        g_virial[n1 + 7 * number_of_particles] += s_szx;
        g_virial[n1 + 8 * number_of_particles] += s_szy;

        // save potential
        g_potential[n1] += s_pe;
    }
}


// calculate the partial forces dU_i/dr_ij
static __global__ void gpu_find_force_vashishta_partial
(
    int number_of_particles, int N1, int N2, Box box, 
    Vashishta_Para vas,
    int *g_neighbor_number, int *g_neighbor_list,
    int *g_type, int shift,
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z, 
    double *g_potential, double *g_f12x, double *g_f12y, double *g_f12z  
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1] - shift;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        double potential_energy = 0.0;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2] - shift;

            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = 1.0 / d12;
          
            double f12x = 0.0; double f12y = 0.0; double f12z = 0.0;
            double gamma2 = 1.0 / ((d12 - vas.r0) * (d12 - vas.r0)); // gamma=1
             
            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];  
                if (n3 == n2) { continue; }
                int type3 = g_type[n3] - shift;   // only consider ABB and BAA
                if (type3 != type2) { continue; } // exclude AAB, BBA, ABA, BAB
                if (type3 == type1) { continue; } // exclude AAA, BBB

                double x13 = g_x[n3] - x1;
                double y13 = g_y[n3] - y1;
                double z13 = g_z[n3] - z1;
                dev_apply_mic(box, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                double exp123 = exp(1.0 / (d12 - vas.r0) + 1.0 / (d13 - vas.r0));
                double one_over_d12d13 = 1.0 / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13) * one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double cos_inv = cos123 - vas.cos0[type1];
                cos_inv = 1.0 / (1.0 + vas.C * cos_inv * cos_inv);

                // accumulate potential energy
                potential_energy += (cos123 - vas.cos0[type1])
                                  * (cos123 - vas.cos0[type1])
                                  * cos_inv*0.5*vas.B[type1]*exp123;

                double tmp1=vas.B[type1]*exp123*cos_inv*(cos123-vas.cos0[type1]);
                double tmp2=gamma2 * (cos123 - vas.cos0[type1]) * d12inv;

                double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += tmp1*(2.0*cos_d*cos_inv-tmp2*x12);
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += tmp1*(2.0*cos_d*cos_inv-tmp2*y12);
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += tmp1*(2.0*cos_d*cos_inv-tmp2*z12);
            }
            g_f12x[index] = f12x;
            g_f12y[index] = f12y;
            g_f12z[index] = f12z;
        }
        // save potential
        g_potential[n1] += potential_energy;
    }
}


// Find force and related quantities for the Vashishta potential (A wrapper)
void Vashishta::compute(Atom *atom, int potential_number)
{
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_VASHISHTA + 1;
    int shift = atom->shift[potential_number];

    // 2-body part
    gpu_find_force_vashishta_2body<<<grid_size, BLOCK_SIZE_VASHISHTA>>>
    (
        atom->N,
        N1,
        N2,
        atom->box,
        vashishta_para,
        atom->neighbor.NN_local.data(),
        atom->neighbor.NL_local.data(),
        vashishta_data.NN_short.data(),
        vashishta_data.NL_short.data(),
        atom->type.data(),
        shift,
        atom->position_per_atom.data(),
        atom->position_per_atom.data() + atom->N,
        atom->position_per_atom.data() + atom->N * 2,
        atom->force_per_atom.data(),
        atom->force_per_atom.data() + atom->N,
        atom->force_per_atom.data() + 2 * atom->N,
        atom->virial_per_atom.data(),
        atom->potential_per_atom.data()
    );
    CUDA_CHECK_KERNEL

    // 3-body part
    gpu_find_force_vashishta_partial<<<grid_size, BLOCK_SIZE_VASHISHTA>>>
    (
        atom->N,
        N1,
        N2,
        atom->box,
        vashishta_para,
        vashishta_data.NN_short.data(),
        vashishta_data.NL_short.data(),
        atom->type.data(),
        shift,
        atom->position_per_atom.data(),
        atom->position_per_atom.data() + atom->N,
        atom->position_per_atom.data() + atom->N * 2,
        atom->potential_per_atom.data(),
        vashishta_data.f12x.data(),
        vashishta_data.f12y.data(),
        vashishta_data.f12z.data()
    );
    CUDA_CHECK_KERNEL
    find_properties_many_body
    (
        atom->box,
        vashishta_data.NN_short.data(),
        vashishta_data.NL_short.data(),
        vashishta_data.f12x.data(),
        vashishta_data.f12y.data(),
        vashishta_data.f12z.data(),
        atom->position_per_atom,
        atom->force_per_atom,
        atom->virial_per_atom
    );
}


