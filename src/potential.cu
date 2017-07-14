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




#include "common.h"
#include "potential.h"

   

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
    int number_of_types = 0;
    count = fscanf(fid, "%d", &number_of_types); 
    if (count != 1) 
    {
        print_error("reading error for potential.in.\n");
        exit(1);
    }
    if (number_of_types != 1) 
    {
        print_error("number of atom types should be 1 for SW.\n");
        exit(1);
    }
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
}



