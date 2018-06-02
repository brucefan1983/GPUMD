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




#include "common.cuh"
#include "potential.cuh"

   

static void initialize_tersoff_1989_1(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use Tersoff-1989 (single-element) potential.\n");

    int count;
#ifdef USE_DP
    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", 
        &force_model->ters0.a,
        &force_model->ters0.b,
        &force_model->ters0.lambda,
        &force_model->ters0.mu,
        &force_model->ters0.beta,
        &force_model->ters0.n,
        &force_model->ters0.c,
        &force_model->ters0.d,
        &force_model->ters0.h,
        &force_model->ters0.r1,
        &force_model->ters0.r2
    );
#else
    count = fscanf
    (
        fid, "%f%f%f%f%f%f%f%f%f%f%f", 
        &force_model->ters0.a,
        &force_model->ters0.b,
        &force_model->ters0.lambda,
        &force_model->ters0.mu,
        &force_model->ters0.beta,
        &force_model->ters0.n,
        &force_model->ters0.c,
        &force_model->ters0.d,
        &force_model->ters0.h,
        &force_model->ters0.r1,
        &force_model->ters0.r2
    );
#endif

    if (count != 11) 
    {
        printf("Error: reading error for potential.in.\n");
        exit(1);
    }

    force_model->ters0.c2 = force_model->ters0.c * force_model->ters0.c;
    force_model->ters0.d2 = force_model->ters0.d * force_model->ters0.d;
    force_model->ters0.one_plus_c2overd2 = 1.0 + force_model->ters0.c2 
                                               / force_model->ters0.d2;
    force_model->ters0.pi_factor 
        = PI / (force_model->ters0.r2 - force_model->ters0.r1);
    force_model->ters0.minus_half_over_n = - 0.5 / force_model->ters0.n;

    force_model->rc = force_model->ters0.r2;
}



static void initialize_tersoff_1989_2(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use Tersoff-1989 (double-element) potential.\n");

    int count;
#ifdef USE_DP
    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", 
        &force_model->ters0.a,
        &force_model->ters0.b,
        &force_model->ters0.lambda,
        &force_model->ters0.mu,
        &force_model->ters0.beta,
        &force_model->ters0.n,
        &force_model->ters0.c,
        &force_model->ters0.d,
        &force_model->ters0.h,
        &force_model->ters0.r1,
        &force_model->ters0.r2
    );
#else
    count = fscanf
    (
        fid, "%f%f%f%f%f%f%f%f%f%f%f", 
        &force_model->ters0.a,
        &force_model->ters0.b,
        &force_model->ters0.lambda,
        &force_model->ters0.mu,
        &force_model->ters0.beta,
        &force_model->ters0.n,
        &force_model->ters0.c,
        &force_model->ters0.d,
        &force_model->ters0.h,
        &force_model->ters0.r1,
        &force_model->ters0.r2
    );
#endif

    if (count != 11) 
    {
        printf("Error: reading error for potential.in.\n");
        exit(1);
    }

#ifdef USE_DP
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", 
            &force_model->ters1.a,
            &force_model->ters1.b,
            &force_model->ters1.lambda,
            &force_model->ters1.mu,
            &force_model->ters1.beta,
            &force_model->ters1.n,
            &force_model->ters1.c,
            &force_model->ters1.d,
            &force_model->ters1.h,
            &force_model->ters1.r1,
            &force_model->ters1.r2
        );
#else
        count = fscanf
        (
            fid, "%f%f%f%f%f%f%f%f%f%f%f", 
            &force_model->ters1.a,
            &force_model->ters1.b,
            &force_model->ters1.lambda,
            &force_model->ters1.mu,
            &force_model->ters1.beta,
            &force_model->ters1.n,
            &force_model->ters1.c,
            &force_model->ters1.d,
            &force_model->ters1.h,
            &force_model->ters1.r1,
            &force_model->ters1.r2
        );
#endif
        if (count != 11) 
        {
            printf("Error: reading error for potential.in.\n");
            exit(1);
        }
    
    // type 0
    force_model->ters0.c2 = force_model->ters0.c * force_model->ters0.c;
    force_model->ters0.d2 = force_model->ters0.d * force_model->ters0.d;
    force_model->ters0.one_plus_c2overd2 = 1.0 + force_model->ters0.c2 
                                               / force_model->ters0.d2;
    force_model->ters0.pi_factor 
        = PI / (force_model->ters0.r2 - force_model->ters0.r1);
    force_model->ters0.minus_half_over_n = - 0.5 / force_model->ters0.n;

    // type 1
    force_model->ters1.c2 = force_model->ters1.c * force_model->ters1.c;
    force_model->ters1.d2 = force_model->ters1.d * force_model->ters1.d;
    force_model->ters1.one_plus_c2overd2 = 1.0 + force_model->ters1.c2 
                                               / force_model->ters1.d2;
    force_model->ters1.pi_factor 
        = PI / (force_model->ters1.r2 - force_model->ters1.r1);
    force_model->ters1.minus_half_over_n = - 0.5 / force_model->ters1.n;

    real chi;
#ifdef USE_DP
    count = fscanf(fid, "%lf", &chi);
#else
    count = fscanf(fid, "%f", &chi); 
#endif       
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}

    // mixing type 0 and type 1
    force_model->ters2.a = sqrt(force_model->ters0.a * force_model->ters1.a);
    force_model->ters2.b = sqrt(force_model->ters0.b * force_model->ters1.b);  
    force_model->ters2.b *= chi;
    force_model->ters2.lambda 
        = 0.5 * (force_model->ters0.lambda + force_model->ters1.lambda);
    force_model->ters2.mu     
        = 0.5 * (force_model->ters0.mu     + force_model->ters1.mu);
    force_model->ters2.beta = 0.0; // not used
    force_model->ters2.n = 0.0;    // not used
    force_model->ters2.c2 = 0.0;   // not used
    force_model->ters2.d2 = 0.0;   // not used
    force_model->ters2.h = 0.0;    // not used
    force_model->ters2.r1 = sqrt(force_model->ters0.r1 * force_model->ters1.r1);
    force_model->ters2.r2 = sqrt(force_model->ters0.r2 * force_model->ters1.r2);
    force_model->ters2.one_plus_c2overd2 = 0.0; // not used
    force_model->ters2.pi_factor 
        = PI / (force_model->ters2.r2 - force_model->ters2.r1);
    force_model->ters2.minus_half_over_n = 0.0; // not used

    force_model->rc = (force_model->ters0.r2 > force_model->ters1.r2) 
                    ?  force_model->ters0.r2 : force_model->ters1.r2;

}




static void initialize_sw_1985(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use Stillinger-Weber potential.\n");
    int count;

#ifdef USE_DP
    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf%lf", &force_model->sw.epsilon, 
        &force_model->sw.lambda, &force_model->sw.A, &force_model->sw.B,
        &force_model->sw.a, &force_model->sw.gamma,
        &force_model->sw.sigma, &force_model->sw.cos0
    );
#else
    count = fscanf
    (
        fid, "%f%f%f%f%f%f%f%f", &force_model->sw.epsilon,
        &force_model->sw.lambda, &force_model->sw.A, &force_model->sw.B,
        &force_model->sw.a, &force_model->sw.gamma,
        &force_model->sw.sigma, &force_model->sw.cos0
    );
#endif
    if (count != 8) 
    {
        print_error("reading error for potential.in.\n");
        exit(1);
    }
    force_model->sw.epsilon_times_A = force_model->sw.epsilon
                                    * force_model->sw.A;
    force_model->sw.epsilon_times_lambda = force_model->sw.epsilon
                                         * force_model->sw.lambda;
    force_model->sw.sigma_times_a = force_model->sw.sigma
                                  * force_model->sw.a;

    force_model->rc = force_model->sw.sigma_times_a;

}  




static void initialize_vashishta(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use Vashishta potential.\n");
    int count;
    
    double B_0, B_1, cos0_0, cos0_1, C, r0, rc;
    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf", &B_0, &B_1, &cos0_0, &cos0_1, &C, &r0, &rc
    );
    if (count != 7) print_error("reading error for Vashishta potential.\n");
    force_model->vas.B[0] = B_0;
    force_model->vas.B[1] = B_1;
    force_model->vas.cos0[0] = cos0_0;
    force_model->vas.cos0[1] = cos0_1;
    force_model->vas.C = C;
    force_model->vas.r0 = r0;
    force_model->vas.rc = rc;
    force_model->rc = rc;
    
    double H[3], qq[3], lambda_inv[3], D[3], xi_inv[3], W[3];
    int eta[3];
    for (int n = 0; n < 3; n++)
    {  
        count = fscanf
        (
            fid, "%lf%d%lf%lf%lf%lf%lf", 
		    &H[n], &eta[n], &qq[n], &lambda_inv[n], &D[n], &xi_inv[n], &W[n]
        );
        if (count != 7) 
		    print_error("reading error for Vashishta potential.\n");
		qq[n] *= K_C;         // Gauss -> SI
		D[n] *= (K_C * HALF); // Gauss -> SI and D -> D/2
		lambda_inv[n] = ONE / lambda_inv[n];
		xi_inv[n] = ONE / xi_inv[n];
		
		force_model->vas.H[n] = H[n];
		force_model->vas.eta[n] = eta[n];
		force_model->vas.qq[n] = qq[n];
		force_model->vas.lambda_inv[n] = lambda_inv[n];
		force_model->vas.D[n] = D[n];
		force_model->vas.xi_inv[n] = xi_inv[n];
		force_model->vas.W[n] = W[n];
			
        real rci = ONE / rc;
        real rci4 = rci * rci * rci * rci;
        real rci6 = rci4 * rci * rci;
        real p2_steric = H[n] * pow(rci, real(eta[n]));
	    real p2_charge = qq[n] * rci * exp(-rc*lambda_inv[n]);
        real p2_dipole = D[n] * rci4 * exp(-rc*xi_inv[n]);
	    real p2_vander = W[n] * rci6;
	    force_model->vas.v_rc[n] = p2_steric+p2_charge-p2_dipole-p2_vander;
        force_model->vas.dv_rc[n] = p2_dipole * (xi_inv[n] + FOUR * rci) 
	                              + p2_vander * (SIX * rci)
                                  - p2_charge * (lambda_inv[n] + rci)      
						          - p2_steric * (eta[n] * rci);
    }
}  





// get U_ij and (d U_ij / d r_ij) / r_ij for the 2-body part
static void find_p2_and_f2
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




static void initialize_vashishta_table(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use tabulated Vashishta potential.\n");
    int count;

    int N; double rmin;
    count = fscanf(fid, "%d%lf", &N, &rmin);
    if (count != 2) print_error("reading error for Vashishta potential.\n");
    force_model->vas_table.N = N;
    force_model->vas_table.rmin = rmin;

    real *cpu_table;
    MY_MALLOC(cpu_table, real, N * 6);
    
    double B_0, B_1, cos0_0, cos0_1, C, r0, rc;
    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf", &B_0, &B_1, &cos0_0, &cos0_1, &C, &r0, &rc
    );
    if (count != 7) print_error("reading error for Vashishta potential.\n");
    force_model->vas_table.B[0] = B_0;
    force_model->vas_table.B[1] = B_1;
    force_model->vas_table.cos0[0] = cos0_0;
    force_model->vas_table.cos0[1] = cos0_1;
    force_model->vas_table.C = C;
    force_model->vas_table.r0 = r0;
    force_model->vas_table.rc = rc;
    force_model->vas_table.scale = (N-ONE)/(rc-rmin);
    force_model->rc = rc;
    
    double H[3], qq[3], lambda_inv[3], D[3], xi_inv[3], W[3];
    int eta[3];
    for (int n = 0; n < 3; n++)
    {  
        count = fscanf
        (
            fid, "%lf%d%lf%lf%lf%lf%lf", 
		          &H[n], &eta[n], &qq[n], &lambda_inv[n], &D[n], &xi_inv[n], &W[n]
        );
        if (count != 7) 
		      print_error("reading error for Vashishta potential.\n");
		      qq[n] *= K_C;         // Gauss -> SI
		      D[n] *= (K_C * HALF); // Gauss -> SI and D -> D/2
		      lambda_inv[n] = ONE / lambda_inv[n];
		      xi_inv[n] = ONE / xi_inv[n];
		
		      force_model->vas_table.H[n] = H[n];
		      force_model->vas_table.eta[n] = eta[n];
		      force_model->vas_table.qq[n] = qq[n];
		      force_model->vas_table.lambda_inv[n] = lambda_inv[n];
		      force_model->vas_table.D[n] = D[n];
		      force_model->vas_table.xi_inv[n] = xi_inv[n];
		      force_model->vas_table.W[n] = W[n];
			
        real rci = ONE / rc;
        real rci4 = rci * rci * rci * rci;
        real rci6 = rci4 * rci * rci;
        real p2_steric = H[n] * pow(rci, real(eta[n]));
	       real p2_charge = qq[n] * rci * exp(-rc*lambda_inv[n]);
        real p2_dipole = D[n] * rci4 * exp(-rc*xi_inv[n]);
	       real p2_vander = W[n] * rci6;
	       force_model->vas_table.v_rc[n] 
            = p2_steric+p2_charge-p2_dipole-p2_vander;
        force_model->vas_table.dv_rc[n] = p2_dipole * (xi_inv[n] + FOUR * rci) 
	                                 + p2_vander * (SIX * rci)
                                  - p2_charge * (lambda_inv[n] + rci)      
						                            - p2_steric * (eta[n] * rci);

        // build the table
        for (int m = 0; m < N; m++) 
        {
            real d12 = rmin + m * (rc - rmin) / (N-ONE);
            real p2, f2;
            find_p2_and_f2
            (
                H[n], eta[n], qq[n], lambda_inv[n], D[n], xi_inv[n], W[n], 
                force_model->vas_table.v_rc[n], 
                force_model->vas_table.dv_rc[n], 
                rc, d12, p2, f2
            );
            int index_p = m + N * n;
            int index_f = m + N * (n + 3);
            cpu_table[index_p] = p2;
            cpu_table[index_f] = f2;
        }
    }

    int memory = sizeof(real) * N * 6;
    CHECK(cudaMalloc((void**)&force_model->vas_table.table, memory));
    cudaMemcpy
    (force_model->vas_table.table, cpu_table, memory, cudaMemcpyHostToDevice);
    MY_FREE(cpu_table);
}  



static void initialize_sw_1985_2(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use two-element Stillinger-Weber potential.\n");
    int count;

    /* format for the potential file (assuming types 0 and 1)
    A[00] B[00] a[00] sigma[00] gamma[00]
    A[01] B[01] a[01] sigma[01] gamma[01]
    A[11] B[11] a[11] sigma[11] gamma[11]
    lambda[000] cos0[000]
    lambda[001] cos0[001]
    lambda[010] cos0[010]
    lambda[011] cos0[011]
    lambda[100] cos0[100]
    lambda[101] cos0[101]
    lambda[110] cos0[110]
    lambda[111] cos0[111]
    */

    // 2-body parameters and the force cutoff
    double A[3], B[3], a[3], sigma[3], gamma[3];
    force_model->rc = 0.0;
    for (int n = 0; n < 3; n++)
    {  
        count = fscanf
        (fid, "%lf%lf%lf%lf%lf", &A[n], &B[n], &a[n], &sigma[n], &gamma[n]);
        if (count != 5) print_error("reading error for potential file.\n");
        force_model->sw2.A[n] = A[n];
        force_model->sw2.B[n] = B[n];
        force_model->sw2.a[n] = a[n];
        force_model->sw2.sigma[n] = sigma[n];
        force_model->sw2.gamma[n] = gamma[n];
        force_model->sw2.rc[n] = sigma[n] * a[n];
        if (force_model->rc < force_model->sw2.rc[n])
            force_model->rc = force_model->sw2.rc[n]; // force cutoff
    }

    // 3-body parameters
    double lambda[8], cos0[8];
    for (int n = 0; n < 8; n++)
    {  
        count = fscanf
        (fid, "%lf%lf", &lambda[n], &cos0[n]);
        if (count != 2) print_error("reading error for potential file.\n");
        force_model->sw2.lambda[n] = lambda[n];
        force_model->sw2.cos0[n] = cos0[n];
    }
}  




static void initialize_sw_1985_3(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use three-element Stillinger-Weber potential.\n");
    int count;

    // 2-body parameters and the force cutoff
    double A[3][3], B[3][3], a[3][3], sigma[3][3], gamma[3][3];
    force_model->rc = 0.0;
    for (int n1 = 0; n1 < 3; n1++)
    for (int n2 = 0; n2 < 3; n2++)
    {  
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf", &A[n1][n2], &B[n1][n2], &a[n1][n2], 
            &sigma[n1][n2], &gamma[n1][n2]
        );
        if (count != 5) print_error("reading error for potential file.\n");
        force_model->sw3.A[n1][n2] = A[n1][n2];
        force_model->sw3.B[n1][n2] = B[n1][n2];
        force_model->sw3.a[n1][n2] = a[n1][n2];
        force_model->sw3.sigma[n1][n2] = sigma[n1][n2];
        force_model->sw3.gamma[n1][n2] = gamma[n1][n2];
        force_model->sw3.rc[n1][n2] = sigma[n1][n2] * a[n1][n2];
        if (force_model->rc < force_model->sw3.rc[n1][n2])
            force_model->rc = force_model->sw3.rc[n1][n2]; // force cutoff
    }

    // 3-body parameters
    double lambda[3][3][3], cos0[3][3][3];
    for (int n1 = 0; n1 < 3; n1++)
    for (int n2 = 0; n2 < 3; n2++)
    for (int n3 = 0; n3 < 3; n3++)
    {  
        count = fscanf
        (fid, "%lf%lf", &lambda[n1][n2][n3], &cos0[n1][n2][n3]);
        if (count != 2) print_error("reading error for potential file.\n");
        force_model->sw3.lambda[n1][n2][n3] = lambda[n1][n2][n3];
        force_model->sw3.cos0[n1][n2][n3] = cos0[n1][n2][n3];
    }
}  




static void initialize_rebo_mos2(Force_Model *force_model)
{
    printf("INPUT: use the potential in [PRB 79, 245110 (2009)].\n");
    force_model->rc = 10.5;
}




// todo: write my_fscanf or change the input methods
static void initialize_eam_zhou_2004_1(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use single-element analytical EAM potential.\n");
    int count;
#ifdef USE_DP
    count = fscanf(fid, "%lf", &force_model->eam1.re);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.fe);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.rho_e);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.rho_s);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.alpha);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.beta);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.A);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.B);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.kappa);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.lambda);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.Fn0);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.Fn1);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.Fn2);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.Fn3);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.F0);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.F1);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.F2);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.F3);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.eta);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.Fe);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->eam1.rc);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
#else
    count = fscanf(fid, "%f", &force_model->eam1.re);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.fe);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.rho_e);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.rho_s);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.alpha);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.beta);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.A);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.B);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.kappa);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.lambda);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.Fn0);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.Fn1);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.Fn2);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.Fn3);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.F0);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.F1);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.F2);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.F3);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.eta);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.Fe);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->eam1.rc);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
#endif
    force_model->eam1.rho_n = force_model->eam1.rho_e * 0.85;
    force_model->eam1.rho_0 = force_model->eam1.rho_e * 1.15;

    force_model->rc = force_model->eam1.rc;


}  




static void initialize_eam_dai_2006(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use single-element analytical FS potential.\n");
    int count;
#ifdef USE_DP
    count = fscanf(fid, "%lf", &force_model->fs.A);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.d);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.c);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.c0);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.c1);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.c2);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.c3);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.c4);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%lf", &force_model->fs.B);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
#else
    count = fscanf(fid, "%f", &force_model->fs.A);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.d);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.c);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.c0);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.c1);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.c2);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.c3);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.c4);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
    count = fscanf(fid, "%f", &force_model->fs.B);
    if (count != 1){print_error("reading error for potential.in.\n"); exit(1);}
#endif

    force_model->rc = (force_model->fs.c > force_model->fs.d) 
                    ?  force_model->fs.c : force_model->fs.d;

}  




static void initialize_lj1(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use single-element LJ potential.\n");
    int count;
    real epsilon, sigma, cutoff;
#ifdef USE_DP
    count = fscanf(fid, "%lf%lf%lf", &epsilon, &sigma, &cutoff);
#else
    count = fscanf(fid, "%f%f%f", &epsilon, &sigma, &cutoff);
#endif
    if (count != 3) 
    {
        print_error("reading error for potential.in.\n");
        exit(1);
    }

#ifdef USE_DP
    force_model->lj1.s6e24  = pow(sigma, 6.0)  * epsilon * 24.0;
    force_model->lj1.s12e24 = pow(sigma, 12.0) * epsilon * 24.0;
    force_model->lj1.s6e4   = pow(sigma, 6.0)  * epsilon * 4.0;
    force_model->lj1.s12e4  = pow(sigma, 12.0) * epsilon * 4.0;
#else
    force_model->lj1.s6e24  = pow(sigma, 6.0f)  * epsilon * 24.0f;
    force_model->lj1.s12e24 = pow(sigma, 12.0f) * epsilon * 24.0f;
    force_model->lj1.s6e4   = pow(sigma, 6.0f)  * epsilon * 4.0f;
    force_model->lj1.s12e4  = pow(sigma, 12.0f) * epsilon * 4.0f;
#endif
    force_model->lj1.cutoff_square = cutoff * cutoff;

    force_model->rc = cutoff;
}  




static void initialize_ri(FILE *fid, Force_Model *force_model)
{
    printf("INPUT: use the rigid-ion potential.\n");
    int count;
    real q1, q2;
    real b11, b22, b12;
#ifdef USE_DP
    count = fscanf(fid, "%lf%lf%lf", &q1, &q2, &force_model->ri.cutoff);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
    count = 
    fscanf(fid, "%lf%lf%lf", &force_model->ri.a11, &b11, &force_model->ri.c11);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
    count = 
    fscanf(fid, "%lf%lf%lf", &force_model->ri.a22, &b22, &force_model->ri.c22);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
    count = 
    fscanf(fid, "%lf%lf%lf", &force_model->ri.a12, &b12, &force_model->ri.c12);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
#else
    count = fscanf(fid, "%f%f%f", &q1, &q2, &force_model->ri.cutoff);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
    count = 
    fscanf(fid, "%f%f%f", &force_model->ri.a11, &b11, &force_model->ri.c11);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
    count = 
    fscanf(fid, "%f%f%f", &force_model->ri.a22, &b22, &force_model->ri.c22);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
    count = 
    fscanf(fid, "%f%f%f", &force_model->ri.a12, &b12, &force_model->ri.c12);
    if (count != 3){print_error("reading error for potential.in.\n"); exit(1);}
#endif
    force_model->ri.qq11 = q1 * q1 * K_C;
    force_model->ri.qq22 = q2 * q2 * K_C;
    force_model->ri.qq12 = q1 * q2 * K_C;
    force_model->ri.b11 = ONE / b11;
    force_model->ri.b22 = ONE / b22;
    force_model->ri.b12 = ONE / b12;

    force_model->rc = force_model->ri.cutoff;
}  




/* Read in potential paramters.
  My indexing conventions:
  LJ:      0-9
  RI:      10-19
  EAM:     20-29
  SW:      30-39
  Tersoff: 40-49
*/
static void initialize_force_model(Files *files, Force_Model *force_model)
{
    printf("INFO:  read in potential parameters.\n");
    FILE *fid_potential = my_fopen(files->potential_in, "r");
    char force_name[20];
    int count = fscanf(fid_potential, "%s", force_name);
    if (count != 1) 
    {
        print_error("reading error for potential.in.\n");
        exit(1);
    }
    else if (strcmp(force_name, "lj1") == 0) 
    { 
        force_model->type = 0; 
        initialize_lj1(fid_potential, force_model);
    }
    else if (strcmp(force_name, "ri") == 0)
    { 
        force_model->type = 10; 
        initialize_ri(fid_potential, force_model);
    }
    else if (strcmp(force_name, "eam_zhou_2004_1") == 0) 
    { 
        force_model->type = 20; 
        initialize_eam_zhou_2004_1(fid_potential, force_model);
    }
    else if (strcmp(force_name, "eam_dai_2006") == 0) 
    { 
        force_model->type = 21; 
        initialize_eam_dai_2006(fid_potential, force_model);
    }
    else if (strcmp(force_name, "sw_1985") == 0) 
    { 
        force_model->type = 30; 
        initialize_sw_1985(fid_potential, force_model);
    }
    else if (strcmp(force_name, "vashishta") == 0) 
    { 
        force_model->type = 32; 
        initialize_vashishta(fid_potential, force_model);
    }
    else if (strcmp(force_name, "sw_1985_2") == 0) 
    { 
        force_model->type = 33; 
        initialize_sw_1985_2(fid_potential, force_model);
    }
    else if (strcmp(force_name, "vashishta_table") == 0) 
    { 
        force_model->type = 34; 
        initialize_vashishta_table(fid_potential, force_model);
    }
    else if (strcmp(force_name, "sw_1985_3") == 0) 
    { 
        force_model->type = 35; 
        initialize_sw_1985_3(fid_potential, force_model);
    }
    else if (strcmp(force_name, "tersoff_1989_1") == 0) 
    { 
        force_model->type = 40; 
        initialize_tersoff_1989_1(fid_potential, force_model);
    }
    else if (strcmp(force_name, "tersoff_1989_2") == 0) 
    { 
        force_model->type = 41; 
        initialize_tersoff_1989_2(fid_potential, force_model);
    }
    else if (strcmp(force_name, "rebo_mos2") == 0) 
    { 
        force_model->type = 42; 
        initialize_rebo_mos2(force_model);
    }
    else    
    { 
        print_error("illegal force model.\n"); 
        exit(1); 
    }
    fclose(fid_potential);
    printf("INFO:  potential parameters initialized.\n\n");
}




//read in potential parameters and then initialize the neighbor list and force
void process_potential
(
    Files *files, Parameters *para, Force_Model *force_model, 
    GPU_Data *gpu_data
)
{    
    initialize_force_model(files, force_model);

    // only for Tersoff-type potentials
    if (force_model->type >= 40 && force_model->type < 50)
    {
        int memory = sizeof(real) * para->N * para->neighbor.MN;
        CHECK(cudaMalloc((void**)&gpu_data->b,  memory)); 
        CHECK(cudaMalloc((void**)&gpu_data->bp, memory)); 
    }

    // for SW and Tersoff type potentials
    if (force_model->type >= 30)
    {
        // Assume that there are at most 20 neighbors for the many-body part;
        // This should be more than enough
        // I do not change 20 to para->neighbor.MN because MN can be very large
        // in some cases
        int memory = sizeof(real) * para->N * 20; 
        CHECK(cudaMalloc((void**)&gpu_data->f12x, memory));
        CHECK(cudaMalloc((void**)&gpu_data->f12y, memory));
        CHECK(cudaMalloc((void**)&gpu_data->f12z, memory));
    }

}



