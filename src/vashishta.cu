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
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE_VASHISHTA 64
#define GPU_FIND_FORCE_VASHISHTA_2BODY(A, B, C, D)                             \
    gpu_find_force_vashishta_2body<A, B, C, D>                                 \
    <<<grid_size, BLOCK_SIZE_VASHISHTA>>>                                      \
    (                                                                          \
        measure->hnemd.fe_x, measure->hnemd.fe_y, measure->hnemd.fe_z,         \
        atom->N, N1, N2, atom->box.triclinic, atom->box.pbc_x,                 \
        atom->box.pbc_y, atom->box.pbc_z, vashishta_para, atom->NN_local,      \
        atom->NL_local, vashishta_data.NN_short, vashishta_data.NL_short,      \
        atom->type, shift, vashishta_data.table, atom->x, atom->y, atom->z,    \
        atom->vx, atom->vy, atom->vz, atom->box.h, atom->fx, atom->fy,         \
        atom->fz, atom->virial_per_atom, atom->potential_per_atom,             \
        atom->heat_per_atom, atom->group[0].label, measure->shc.fv_index,      \
        measure->shc.fv, measure->shc.a_map, measure->shc.b_map,               \
        measure->shc.count_b                                                   \
    )


/*----------------------------------------------------------------------------80
    Reference: 
        P. Vashishta et al., J. Appl. Phys. 101, 103515 (2007).
*-----------------------------------------------------------------------------*/


void Vashishta::initialize_0(FILE *fid)
{
    printf("Use Vashishta potential.\n");
    int count;

    double B_0, B_1, cos0_0, cos0_1, C, r0, cut;
    count = fscanf
    (fid, "%lf%lf%lf%lf%lf%lf%lf", &B_0, &B_1, &cos0_0, &cos0_1, &C, &r0, &cut);
    if (count != 7) print_error("reading error for Vashishta potential.\n");
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
        if (count != 7) print_error("reading error for Vashishta potential.\n");
        qq[n] *= K_C;         // Gauss -> SI
        D[n] *= (K_C * HALF); // Gauss -> SI and D -> D/2
        lambda_inv[n] = ONE / lambda_inv[n];
        xi_inv[n] = ONE / xi_inv[n];

        vashishta_para.H[n] = H[n];
        vashishta_para.eta[n] = eta[n];
        vashishta_para.qq[n] = qq[n];
        vashishta_para.lambda_inv[n] = lambda_inv[n];
        vashishta_para.D[n] = D[n];
        vashishta_para.xi_inv[n] = xi_inv[n];
        vashishta_para.W[n] = W[n];

        real rci = ONE / rc;
        real rci4 = rci * rci * rci * rci;
        real rci6 = rci4 * rci * rci;
        real p2_steric = H[n] * pow(rci, real(eta[n]));
        real p2_charge = qq[n] * rci * exp(-rc*lambda_inv[n]);
        real p2_dipole = D[n] * rci4 * exp(-rc*xi_inv[n]);
        real p2_vander = W[n] * rci6;
        vashishta_para.v_rc[n] = p2_steric+p2_charge-p2_dipole-p2_vander;
        vashishta_para.dv_rc[n] = p2_dipole * (xi_inv[n] + FOUR * rci) 
                                + p2_vander * (SIX * rci)
                                - p2_charge * (lambda_inv[n] + rci)
                                - p2_steric * (eta[n] * rci);
    }
}


// get U_ij and (d U_ij / d r_ij) / r_ij for the 2-body part
static void find_p2_and_f2_host
(
    real H, int eta, real qq, real lambda_inv, real D, real xi_inv, real W, 
    real v_rc, real dv_rc, real rc, real d12, real &p2, real &f2
)
{
    real d12inv = ONE / d12;
    real d12inv2 = d12inv * d12inv;
    real p2_steric = eta; p2_steric = H * pow(d12inv, eta);
    real p2_charge = qq * d12inv * exp(-d12 * lambda_inv);
    real p2_dipole = D * (d12inv2 * d12inv2) * exp(-d12 * xi_inv);
    real p2_vander = W * (d12inv2 * d12inv2 * d12inv2);
    p2 = p2_steric + p2_charge - p2_dipole - p2_vander; 
    p2 -= v_rc + (d12 - rc) * dv_rc; // shifted potential
    f2 = p2_dipole * (xi_inv + FOUR*d12inv) + p2_vander * (SIX * d12inv);
    f2 -= p2_charge * (lambda_inv + d12inv) + p2_steric * (eta * d12inv);
    f2 = (f2 - dv_rc) * d12inv;      // shifted force
}


void Vashishta::initialize_1(FILE *fid)
{
    printf("Use tabulated Vashishta potential.\n");
    int count;

    int N; double rmin;
    count = fscanf(fid, "%d%lf", &N, &rmin);
    if (count != 2) 
    {
        print_error("reading error for Vashishta potential.\n");
        exit(1);
    }
    vashishta_para.N = N;
    vashishta_para.rmin = rmin;

    real *table;
    MY_MALLOC(table, real, N * 6);
    
    double B_0, B_1, cos0_0, cos0_1, C, r0, cut;
    count = fscanf
    (fid, "%lf%lf%lf%lf%lf%lf%lf", &B_0, &B_1, &cos0_0, &cos0_1, &C, &r0, &cut);
    if (count != 7) print_error("reading error for Vashishta potential.\n");

    vashishta_para.B[0] = B_0;
    vashishta_para.B[1] = B_1;
    vashishta_para.cos0[0] = cos0_0;
    vashishta_para.cos0[1] = cos0_1;
    vashishta_para.C = C;
    vashishta_para.r0 = r0;
    vashishta_para.rc = cut;
    vashishta_para.scale = (N-ONE)/(cut-rmin);
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
        if (count != 7) print_error("reading error for Vashishta potential.\n");

        qq[n] *= K_C;         // Gauss -> SI
        D[n] *= (K_C * HALF); // Gauss -> SI and D -> D/2
        lambda_inv[n] = ONE / lambda_inv[n];
        xi_inv[n] = ONE / xi_inv[n];

        vashishta_para.H[n] = H[n];
        vashishta_para.eta[n] = eta[n];
        vashishta_para.qq[n] = qq[n];
        vashishta_para.lambda_inv[n] = lambda_inv[n];
        vashishta_para.D[n] = D[n];
        vashishta_para.xi_inv[n] = xi_inv[n];
        vashishta_para.W[n] = W[n];

        real rci = ONE / rc;
        real rci4 = rci * rci * rci * rci;
        real rci6 = rci4 * rci * rci;
        real p2_steric = H[n] * pow(rci, real(eta[n]));
        real p2_charge = qq[n] * rci * exp(-rc*lambda_inv[n]);
        real p2_dipole = D[n] * rci4 * exp(-rc*xi_inv[n]);
        real p2_vander = W[n] * rci6;
        vashishta_para.v_rc[n] = p2_steric+p2_charge-p2_dipole-p2_vander;
        vashishta_para.dv_rc[n] = p2_dipole * (xi_inv[n] + FOUR * rci) 
                                + p2_vander * (SIX * rci)
                                - p2_charge * (lambda_inv[n] + rci)
                                - p2_steric * (eta[n] * rci);

        // build the table
        for (int m = 0; m < N; m++) 
        {
            real d12 = rmin + m * (cut - rmin) / (N-ONE);
            real p2, f2;
            find_p2_and_f2_host
            (
                H[n], eta[n], qq[n], lambda_inv[n], D[n], xi_inv[n], W[n], 
                vashishta_para.v_rc[n], 
                vashishta_para.dv_rc[n], 
                rc, d12, p2, f2
            );
            int index_p = m + N * n;
            int index_f = m + N * (n + 3);
            table[index_p] = p2;
            table[index_f] = f2;
        }
    }

    int memory = sizeof(real) * N * 6;
    CHECK(cudaMalloc((void**)&vashishta_data.table, memory));
    CHECK(cudaMemcpy(vashishta_data.table, table, memory,
        cudaMemcpyHostToDevice));
    MY_FREE(table);
}


Vashishta::Vashishta(FILE *fid, Atom* atom, int use_table_input)
{
    use_table = use_table_input;
    if (use_table == 0) initialize_0(fid);
    if (use_table == 1) initialize_1(fid);

    int num = ((atom->neighbor.MN<20) ? atom->neighbor.MN : 20);
    int memory = sizeof(real) * atom->N * num;
    CHECK(cudaMalloc((void**)&vashishta_data.f12x, memory));
    CHECK(cudaMalloc((void**)&vashishta_data.f12y, memory));
    CHECK(cudaMalloc((void**)&vashishta_data.f12z, memory));
    memory = sizeof(int) * atom->N;
    CHECK(cudaMalloc((void**)&vashishta_data.NN_short, memory));
    memory = sizeof(int) * atom->N * num;
    CHECK(cudaMalloc((void**)&vashishta_data.NL_short, memory));
}




Vashishta::~Vashishta(void)
{
    if (use_table) { CHECK(cudaFree(vashishta_data.table)); }
    CHECK(cudaFree(vashishta_data.f12x));
    CHECK(cudaFree(vashishta_data.f12y));
    CHECK(cudaFree(vashishta_data.f12z));
    CHECK(cudaFree(vashishta_data.NN_short));
    CHECK(cudaFree(vashishta_data.NL_short));
}


// eta is always an integer and we don't need the very slow pow()
static __device__ real my_pow(real x, int n) 
{
    if (n == 7) 
    { 
        real y = x;
        x *= x;
        y *= x; // x^3
        x *= x; // x^4
        return y * x;
    }
    else if (n == 9) 
    { 
        real y = x;
        x *= x; // x^2
        x *= x; // x^4
        y *= x; // x^5
        return y * x; 
    }
    else // n = 11
    { 
        real y = x;
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
    real H, int eta, real qq, real lambda_inv, real D, real xi_inv, real W, 
    real v_rc, real dv_rc, real rc, real d12, real &p2, real &f2
)
{
    real d12inv = ONE / d12;
    real d12inv2 = d12inv * d12inv;
    real p2_steric = H * my_pow(d12inv, eta);
    real p2_charge = qq * d12inv * exp(-d12 * lambda_inv);
    real p2_dipole = D * (d12inv2 * d12inv2) * exp(-d12 * xi_inv);
    real p2_vander = W * (d12inv2 * d12inv2 * d12inv2);
    p2 = p2_steric + p2_charge - p2_dipole - p2_vander; 
    p2 -= v_rc + (d12 - rc) * dv_rc; // shifted potential
    f2 = p2_dipole * (xi_inv + FOUR*d12inv) + p2_vander * (SIX * d12inv);
    f2 -= p2_charge * (lambda_inv + d12inv) + p2_steric * (eta * d12inv);
    f2 = (f2 - dv_rc) * d12inv;      // shifted force
}


// 2-body part of the Vashishta potential (kernel)
template <int use_table, int cal_j, int cal_q, int cal_k>
static __global__ void gpu_find_force_vashishta_2body
(
    real fe_x, real fe_y, real fe_z,
    int number_of_particles, int N1, int N2, 
    int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    Vashishta_Para vas,
    int *g_NN, int *g_NL, int *g_NN_local, int *g_NL_local,
    int *g_type, int shift,
    const real* __restrict__ g_table,
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
    const real* __restrict__ g_vx, 
    const real* __restrict__ g_vy, 
    const real* __restrict__ g_vz,
    const real* __restrict__ g_box, real *g_fx, real *g_fy, real *g_fz,
    real *g_virial, real *g_potential, 
    real *g_h, int *g_label, int *g_fv_index, real *g_fv,
    int *g_a_map, int *g_b_map, int g_count_b
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    real s_fx = ZERO; // force_x
    real s_fy = ZERO; // force_y
    real s_fz = ZERO; // force_z
    real s_pe = ZERO; // potential energy
    real s_sxx = ZERO; // virial_stress_xx
    real s_syy = ZERO; // virial_stress_yy
    real s_szz = ZERO; // virial_stress_zz
    real s_sxy = ZERO; // virial_stress_xy
    real s_sxz = ZERO; // virial_stress_xz
    real s_syz = ZERO; // virial_stress_yz
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
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1] - shift;
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
        
        int count = 0; // initialize g_NN_local[n1] to 0

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = g_NL[n1 + number_of_particles * i1];
            
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            if (d12 >= vas.rc) { continue; }
            if (d12 < vas.r0) // r0 is much smaller than rc
            {                    
                g_NL_local[n1 + number_of_particles * (count++)] = n2;
            }
            int type2 = g_type[n2] - shift;
            int type12 = type1 + type2; // 0 = AA; 1 = AB or BA; 2 = BB
            real p2, f2;

            if (use_table == 1)
            {
                if (d12 > vas.rmin)
                {
                    real tmp = (d12 - vas.rmin) * vas.scale;
                    int index = tmp; // 0 <= index < N-1
                    real x = tmp - index; // 0 <= x < 1
                    index += type12 * vas.N;
                    p2 = (ONE-x)*LDG(g_table, index) + x*LDG(g_table, index+1);
                    index += vas.N * 3;    
                    f2 = (ONE-x)*LDG(g_table, index) + x*LDG(g_table, index+1);
                }
                else
                {
                    find_p2_and_f2
                    (
                        vas.H[type12], vas.eta[type12], vas.qq[type12], 
                        vas.lambda_inv[type12], vas.D[type12], 
                        vas.xi_inv[type12], vas.W[type12], vas.v_rc[type12], 
                        vas.dv_rc[type12], vas.rc, d12, p2, f2
                    );
                }
            }

            if (use_table == 0)
            {
                find_p2_and_f2
                (
                    vas.H[type12], vas.eta[type12], vas.qq[type12], 
                    vas.lambda_inv[type12], vas.D[type12], vas.xi_inv[type12],
                    vas.W[type12], vas.v_rc[type12], vas.dv_rc[type12], 
                    vas.rc, d12, p2, f2
                );
            }

            // treat two-body potential in the same way as many-body potential
            real f12x = f2 * x12 * HALF; 
            real f12y = f2 * y12 * HALF; 
            real f12z = f2 * z12 * HALF; 
            real f21x = -f12x; 
            real f21y = -f12y; 
            real f21z = -f12z; 
       
            // accumulate force
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
            
            // accumulate potential energy and virial
            s_pe += p2 * HALF; // two-body potential
            s_sxx += x12 * f21x;
            s_syy += y12 * f21y;
            s_szz += z12 * f21z;
            s_sxy += x12 * f21y;
            s_sxz += x12 * f21z;
            s_syz += y12 * f21z;
            
            if (cal_j || cal_k) // heat current (EMD)
            {
                s_h1 += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s_h2 += (f21z * vz1) * x12;               // x-out
                s_h3 += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s_h4 += (f21z * vz1) * y12;               // y-out
                s_h5 += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            }

            // accumulate heat across some sections (for NEMD)
            // check if AB pair possible & exists
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

        g_NN_local[n1] = count; // now the local neighbor list has been built

        // add driving force
        if (cal_k)
        {
            s_fx += fx_driving;
            s_fy += fy_driving;
            s_fz += fz_driving;
        }

        g_fx[n1] += s_fx; // save force
        g_fy[n1] += s_fy;
        g_fz[n1] += s_fz;

        // save stress and potential
        g_virial[n1 + 0 * number_of_particles] += s_sxx;
        g_virial[n1 + 1 * number_of_particles] += s_syy;
        g_virial[n1 + 2 * number_of_particles] += s_szz;
        g_virial[n1 + 3 * number_of_particles] += s_sxy;
        g_virial[n1 + 4 * number_of_particles] += s_sxz;
        g_virial[n1 + 5 * number_of_particles] += s_syz;
        g_potential[n1] += s_pe;
        if (cal_j || cal_k) // save heat current
        {
            g_h[n1 + 0 * number_of_particles] += s_h1;
            g_h[n1 + 1 * number_of_particles] += s_h2;
            g_h[n1 + 2 * number_of_particles] += s_h3;
            g_h[n1 + 3 * number_of_particles] += s_h4;
            g_h[n1 + 4 * number_of_particles] += s_h5;
        }
    }
}


// calculate the partial forces dU_i/dr_ij
static __global__ void gpu_find_force_vashishta_partial
(
    int number_of_particles, int N1, int N2, 
    int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    Vashishta_Para vas,
    int *g_neighbor_number, int *g_neighbor_list,
    int *g_type, int shift,
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
    const real* __restrict__ g_box,
    real *g_potential, real *g_f12x, real *g_f12y, real *g_f12z  
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1] - shift;
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real potential_energy = ZERO;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2] - shift;

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real d12inv = ONE / d12;
          
            real f12x = ZERO; real f12y = ZERO; real f12z = ZERO;
            real gamma2 = ONE / ((d12 - vas.r0) * (d12 - vas.r0)); // gamma=1
             
            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];  
                if (n3 == n2) { continue; }
                int type3 = g_type[n3] - shift;   // only consider ABB and BAA
                if (type3 != type2) { continue; } // exclude AAB, BBA, ABA, BAB
                if (type3 == type1) { continue; } // exclude AAA, BBB

                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box,
                    x13, y13, z13);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                real exp123 = exp(ONE / (d12 - vas.r0) + ONE / (d13 - vas.r0));
                real one_over_d12d13 = ONE / (d12 * d13);
                real cos123 = (x12*x13 + y12*y13 + z12*z13) * one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv;
                real cos_inv = cos123 - vas.cos0[type1];
                cos_inv = ONE / (ONE + vas.C * cos_inv * cos_inv);

                // accumulate potential energy
                potential_energy += (cos123 - vas.cos0[type1])
                                  * (cos123 - vas.cos0[type1])
                                  * cos_inv*HALF*vas.B[type1]*exp123;

                real tmp1=vas.B[type1]*exp123*cos_inv*(cos123-vas.cos0[type1]);
                real tmp2=gamma2 * (cos123 - vas.cos0[type1]) * d12inv;

                real cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += tmp1*(TWO*cos_d*cos_inv-tmp2*x12);
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += tmp1*(TWO*cos_d*cos_inv-tmp2*y12);
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += tmp1*(TWO*cos_d*cos_inv-tmp2*z12);
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
void Vashishta::compute(Atom *atom, Measure *measure, int potential_number)
{
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_VASHISHTA + 1;
    int shift = atom->shift[potential_number];
    find_measurement_flags(atom, measure);
    // 2-body part
    if (compute_j)
    {
        if (use_table == 0)
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(0, 1, 0, 0);
        }
        else
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(1, 1, 0, 0);
        }
        CUDA_CHECK_KERNEL
    }
    else if (compute_shc && !measure->hnemd.compute)
    {
        if (use_table == 0)
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(0, 0, 1, 0);
        }
        else
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(1, 0, 1, 0);
        }
        CUDA_CHECK_KERNEL
    }
    else if (measure->hnemd.compute && !compute_shc)
    {
        if (use_table == 0)
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(0, 0, 0, 1);
        }
        else
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(1, 0, 0, 1);
        }
        CUDA_CHECK_KERNEL
    }
    else if (measure->hnemd.compute && compute_shc)
    {
        if (use_table == 0)
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(0, 0, 1, 1);
        }
        else
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(1, 0, 1, 1);
        }
        CUDA_CHECK_KERNEL
    }
    else
    {
        if (use_table == 0)
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(0, 0, 0, 0);
        }
        else
        {
            GPU_FIND_FORCE_VASHISHTA_2BODY(1, 0, 0, 0);
        }
        CUDA_CHECK_KERNEL
    }
    // 3-body part
    gpu_find_force_vashishta_partial<<<grid_size, BLOCK_SIZE_VASHISHTA>>>
    (
        atom->N, N1, N2, atom->box.triclinic, atom->box.pbc_x, atom->box.pbc_y,
        atom->box.pbc_z, vashishta_para, vashishta_data.NN_short,
        vashishta_data.NL_short, atom->type, shift, atom->x, atom->y, atom->z,
        atom->box.h, atom->potential_per_atom, vashishta_data.f12x, 
        vashishta_data.f12y, vashishta_data.f12z
    );
    CUDA_CHECK_KERNEL
    find_properties_many_body
    (
        atom, measure, vashishta_data.NN_short, vashishta_data.NL_short,
        vashishta_data.f12x, vashishta_data.f12y, vashishta_data.f12z
    );
}


