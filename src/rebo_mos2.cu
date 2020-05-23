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
The REBO potential for Mo-S systems
References: 
MoS:  T. Liang et al., PRB 79, 245110 (2009).
MoS:  T. Liang et al., PRB 85, 199903(E) (2012).
MoS:  J. A. Stewart et al., MSMSE 21, 045003 (2013).
------------------------------------------------------------------------------*/


#include "rebo_mos2.cuh"
#include "mic.cuh"
#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE_FORCE 64

#define TWOPI 6.283185307179586

#define REBO_MOS2_Q_MM     3.41912939000591
#define REBO_MOS2_A_MM     179.008013654688
#define REBO_MOS2_B_MM     706.247903589221
#define REBO_MOS2_a_MM     1.0750071299934
#define REBO_MOS2_b_MM     1.16100322369589
#define REBO_MOS2_r1_MM    3.5
#define REBO_MOS2_r2_MM    3.8

#define REBO_MOS2_Q_SS     0.254959104053671
#define REBO_MOS2_A_SS     1228.43233679426
#define REBO_MOS2_B_SS     1498.64815404145
#define REBO_MOS2_a_SS     1.10775022439715
#define REBO_MOS2_b_SS     1.1267362361032
#define REBO_MOS2_r1_SS    2.3
#define REBO_MOS2_r2_SS    3.0

#define REBO_MOS2_Q_MS    1.50553783915379
#define REBO_MOS2_A_MS    575.509677721866
#define REBO_MOS2_B_MS    1344.46820036159
#define REBO_MOS2_a_MS    1.1926790221882
#define REBO_MOS2_b_MS    1.2697375220429
#define REBO_MOS2_r1_MS   2.75
#define REBO_MOS2_r2_MS   3.05

#define REBO_MOS2_pi_factor_MM    10.471975511965978
#define REBO_MOS2_pi_factor_SS    4.487989505128276
#define REBO_MOS2_pi_factor_MS    10.471975511965978

// From Stewart
#define REBO_MOS2_a0_M   0.138040769883614
#define REBO_MOS2_a1_M   0.803625443023934
#define REBO_MOS2_a2_M   0.292412960851064
#define REBO_MOS2_a3_M   0.640588078946224

#define REBO_MOS2_a0_S   0.062978539843324
#define REBO_MOS2_a1_S   2.478617619878250
#define REBO_MOS2_a2_S   0.036666243238154
#define REBO_MOS2_a3_S   2.386431372486710

// G
#define REBO_MOS2_b0_M   0.132684255066327
#define REBO_MOS2_b1_M  -0.007642788338017
#define REBO_MOS2_b2_M   0.034139577505937
#define REBO_MOS2_b3_M   0.252305097138087
#define REBO_MOS2_b4_M   0.122728737222567
#define REBO_MOS2_b5_M  -0.361387798398897
#define REBO_MOS2_b6_M  -0.282577591351457

#define REBO_MOS2_b0_S   0.00684876159675
#define REBO_MOS2_b1_S  -0.02389964401024
#define REBO_MOS2_b2_S   0.13745735331117
#define REBO_MOS2_b3_S   0.03301646749774
#define REBO_MOS2_b4_S  -0.3106429154485
#define REBO_MOS2_b5_S  -0.08550273135791
#define REBO_MOS2_b6_S   0.14925279030688

// gamma - G
#define REBO_MOS2_c0_M  -0.012489954031047
#define REBO_MOS2_c1_M   0.052881075696207
#define REBO_MOS2_c2_M   0.033783229738093
#define REBO_MOS2_c3_M  -0.289030210924907
#define REBO_MOS2_c4_M  -0.015212259708707
#define REBO_MOS2_c5_M   0.366352510383837
#define REBO_MOS2_c6_M   0.152601607764937

#define REBO_MOS2_c0_S  -0.291933961596750
#define REBO_MOS2_c1_S   1.694924444010240
#define REBO_MOS2_c2_S  -3.705308953311170
#define REBO_MOS2_c3_S   3.417533432502260
#define REBO_MOS2_c4_S  -0.907985984551500
#define REBO_MOS2_c5_S   0.085502731357910
#define REBO_MOS2_c6_S  -0.149252790306880

// LJ (From Liang)
#define REBO_MOS2_EPSILON_MM  0.00058595
#define REBO_MOS2_EPSILON_SS  0.02
#define REBO_MOS2_EPSILON_MS  0.003423302499050

#define REBO_MOS2_SIGMA_MM    4.2
#define REBO_MOS2_SIGMA_SS    3.13
#define REBO_MOS2_SIGMA_MS    3.665

// 0.95 * sigma
#define REBO_MOS2_LJCUT1_MM   3.99
#define REBO_MOS2_LJCUT1_SS   2.9735
#define REBO_MOS2_LJCUT1_MS   3.48175

// 2.5 * sigma
#define REBO_MOS2_LJCUT2_MM   10.5
#define REBO_MOS2_LJCUT2_SS   7.825
#define REBO_MOS2_LJCUT2_MS   9.1625

// 4 * epsilon * sigma^12
#define REBO_MOS2_s12e4_MM    70617.45058278613
#define REBO_MOS2_s12e4_SS    70732.99337720051
#define REBO_MOS2_s12e4_MS    80425.36048432751

// 4 * epsilon * sigma^6
#define REBO_MOS2_s6e4_MM     12.865192601587204
#define REBO_MOS2_s6e4_SS     75.223928840336711
#define REBO_MOS2_s6e4_MS     33.185559361443481

// 48 * epsilon * sigma^12
#define REBO_MOS2_s12e48_MM   847409.4069934335
#define REBO_MOS2_s12e48_SS   848795.9205264060
#define REBO_MOS2_s12e48_MS   965104.3258119302

// 24 * epsilon * sigma^6
#define REBO_MOS2_s6e24_MM    77.19115560952322
#define REBO_MOS2_s6e24_SS    451.3435730420202
#define REBO_MOS2_s6e24_MS    199.1133561686609

// pre-computed coefficient of the (r - r1)^2 term
#define REBO_MOS2_D2_MM      0.031194467724753
#define REBO_MOS2_D2_SS      0.820449609102021
#define REBO_MOS2_D2_MS      0.113097798217445

// pre-computed coefficient of the (r - r1)^3 term
#define REBO_MOS2_D3_MM     -0.053895558827613
#define REBO_MOS2_D3_SS     -1.089810409215252
#define REBO_MOS2_D3_MS     -0.137425146625715


REBO_MOS::REBO_MOS(Atom* atom)
{
    int num = (atom->neighbor.MN < 50) ? atom->neighbor.MN : 50;
    rebo_mos_data.p.resize(atom->N);
    rebo_mos_data.pp.resize(atom->N);
    rebo_mos_data.b.resize(atom->N * num);
    rebo_mos_data.bp.resize(atom->N * num);
    rebo_mos_data.f12x.resize(atom->N * num);
    rebo_mos_data.f12y.resize(atom->N * num);
    rebo_mos_data.f12z.resize(atom->N * num);
    rebo_mos_data.NN_short.resize(atom->N);
    rebo_mos_data.NL_short.resize(atom->N * num);

    printf("Use the potential in [PRB 79, 245110 (2009)].\n");
    rc = 10.5;
}


REBO_MOS::~REBO_MOS(void)
{
    // nothing
}


// The repulsive function and its derivative
static __device__ void find_fr_and_frp
(int type12, double d12, double &fr, double &frp)
{     
    if (type12 == 0)
    {   
        fr  = (1.0 + REBO_MOS2_Q_MM / d12) * REBO_MOS2_A_MM 
            * exp(-REBO_MOS2_a_MM * d12);  
        frp = REBO_MOS2_a_MM + REBO_MOS2_Q_MM / (d12 * (d12 + REBO_MOS2_Q_MM));
        frp *= -fr;
    }
    else if (type12 == 2)
    {
        fr  = (1.0 + REBO_MOS2_Q_SS / d12) * REBO_MOS2_A_SS 
            * exp(-REBO_MOS2_a_SS * d12);  
        frp = REBO_MOS2_a_SS + REBO_MOS2_Q_SS / (d12 * (d12 + REBO_MOS2_Q_SS));
        frp *= -fr;
    }  
    else
    {
        fr  = (1.0 + REBO_MOS2_Q_MS / d12) * REBO_MOS2_A_MS 
            * exp(-REBO_MOS2_a_MS * d12);  
        frp = REBO_MOS2_a_MS + REBO_MOS2_Q_MS / (d12 * (d12 + REBO_MOS2_Q_MS));
        frp *= -fr;
    }
}


// The attractive function and its derivative
static __device__ void find_fa_and_fap
(int type12, double d12, double &fa, double &fap)
{  
    if (type12 == 0)
    {   
        fa  = REBO_MOS2_B_MM * exp(- REBO_MOS2_b_MM * d12); 
        fap = - REBO_MOS2_b_MM * fa;
    }
    else if (type12 == 2)
    {    
        fa  = REBO_MOS2_B_SS * exp(- REBO_MOS2_b_SS * d12); 
        fap = - REBO_MOS2_b_SS * fa;
    }
    else
    {
        fa  = REBO_MOS2_B_MS * exp(- REBO_MOS2_b_MS * d12); 
        fap = - REBO_MOS2_b_MS * fa;
    }
}


// The attractive function
static __device__ void find_fa
(int type12, double d12, double &fa)
{
    if (type12 == 0)
    {
        fa  = REBO_MOS2_B_MM * exp(- REBO_MOS2_b_MM * d12); 
    }
    else if (type12 == 2)
    {
        fa  = REBO_MOS2_B_SS * exp(- REBO_MOS2_b_SS * d12); 
    }
    else
    {
        fa  = REBO_MOS2_B_MS * exp(- REBO_MOS2_b_MS * d12); 
    }
}


// The cutoff function and its derivative
static __device__ void find_fc_and_fcp
(int type12, double d12, double &fc, double &fcp)
{
    if (type12 == 0)
    {
        if (d12 < REBO_MOS2_r1_MM) {fc = 1.0; fcp = 0.0;}
        else if (d12 < REBO_MOS2_r2_MM)
        {              
            fc  = cos(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM)) 
                * 0.5 + 0.5;
            fcp = -sin(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM))
                * REBO_MOS2_pi_factor_MM * 0.5;
        }
        else {fc  = 0.0; fcp = 0.0;}
    }
    else if (type12 == 2)
    { 
        if (d12 < REBO_MOS2_r1_SS) {fc = 1.0; fcp = 0.0;}
        else if (d12 < REBO_MOS2_r2_SS)
        {      
            fc  = cos(REBO_MOS2_pi_factor_SS * (d12 - REBO_MOS2_r1_SS)) 
                * 0.5 + 0.5;
            fcp = -sin(REBO_MOS2_pi_factor_SS * (d12 - REBO_MOS2_r1_SS))
                * REBO_MOS2_pi_factor_SS * 0.5;
        }
        else {fc  = 0.0; fcp = 0.0;}
    }
    else  
    {
        if (d12 < REBO_MOS2_r1_MS) {fc = 1.0; fcp = 0.0;}
        else if (d12 < REBO_MOS2_r2_MS)
        {              
            fc  = cos(REBO_MOS2_pi_factor_MS * (d12 - REBO_MOS2_r1_MS)) 
                * 0.5 + 0.5;
            fcp = -sin(REBO_MOS2_pi_factor_MS * (d12 - REBO_MOS2_r1_MS))
                * REBO_MOS2_pi_factor_MS * 0.5;
        }
        else {fc  = 0.0; fcp = 0.0;}
    }
}


// The cutoff function
static __device__ void find_fc(int type12, double d12, double &fc)
{
    if (type12 == 0)
    {
        if (d12 < REBO_MOS2_r1_MM) {fc = 1.0;}
        else if (d12 < REBO_MOS2_r2_MM)
        {
            fc  = cos(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM)) 
                * 0.5 + 0.5;
        }
        else {fc  = 0.0;}
    }
    else if (type12 == 2)
    {
        if (d12 < REBO_MOS2_r1_SS) {fc = 1.0;}
        else if (d12 < REBO_MOS2_r2_SS)
        {
            fc  = cos(REBO_MOS2_pi_factor_SS * (d12 - REBO_MOS2_r1_SS)) 
                * 0.5 + 0.5;
        }
        else {fc  = 0.0;}
    }
    else  
    {
        if (d12 < REBO_MOS2_r1_MS) {fc = 1.0;}
        else if (d12 < REBO_MOS2_r2_MS)
        {
            fc  = cos(REBO_MOS2_pi_factor_MS * (d12 - REBO_MOS2_r1_MS)) 
                * 0.5 + 0.5;
        }
        else {fc  = 0.0;}
    }
}


// The angular function and its derivative
static __device__ void find_g_and_gp(int type1, double x, double &g, double &gp)
{
    if (type1 == 0) // Mo
    {
        g =         REBO_MOS2_b6_M; 
        g = g * x + REBO_MOS2_b5_M;
        g = g * x + REBO_MOS2_b4_M;
        g = g * x + REBO_MOS2_b3_M;
        g = g * x + REBO_MOS2_b2_M;
        g = g * x + REBO_MOS2_b1_M;
        g = g * x + REBO_MOS2_b0_M;

        gp =          6.0   * REBO_MOS2_b6_M;
        gp = gp * x + 5.0  * REBO_MOS2_b5_M;
        gp = gp * x + 4.0  * REBO_MOS2_b4_M;
        gp = gp * x + 3.0 * REBO_MOS2_b3_M;
        gp = gp * x + 2.0   * REBO_MOS2_b2_M;
        gp = gp * x +         REBO_MOS2_b1_M;

        if (x > 0.5)
        {
            // tmp = (gamma - G)
            double tmp =      REBO_MOS2_c6_M;
            tmp = tmp * x + REBO_MOS2_c5_M;
            tmp = tmp * x + REBO_MOS2_c4_M;
            tmp = tmp * x + REBO_MOS2_c3_M;
            tmp = tmp * x + REBO_MOS2_c2_M;
            tmp = tmp * x + REBO_MOS2_c1_M;
            tmp = tmp * x + REBO_MOS2_c0_M;

            // psi
            double psi = 0.5 * ( 1.0 - cos( TWOPI * (x-0.5) ) );

            // g = G + psi * (gamma - G)
            g += psi * tmp;

            // gp = G' + psi' * (gamma - G) now
            gp += PI * sin( TWOPI * (x-0.5) ) * tmp;

            // tmp = (gamma - G)'
            tmp =           6.0   * REBO_MOS2_c6_M;
            tmp = tmp * x + 5.0  * REBO_MOS2_c5_M;
            tmp = tmp * x + 4.0  * REBO_MOS2_c4_M;
            tmp = tmp * x + 3.0 * REBO_MOS2_c3_M;
            tmp = tmp * x + 2.0   * REBO_MOS2_c2_M;
            tmp = tmp * x +         REBO_MOS2_c1_M;

            // gp = G' + psi' * (gamma - G) + psi * (gamma - G)' now
            gp += psi * tmp;
        }
    }
    else // S
    {
        g =         REBO_MOS2_b6_S;
        g = g * x + REBO_MOS2_b5_S;
        g = g * x + REBO_MOS2_b4_S;
        g = g * x + REBO_MOS2_b3_S;
        g = g * x + REBO_MOS2_b2_S;
        g = g * x + REBO_MOS2_b1_S;
        g = g * x + REBO_MOS2_b0_S;

        gp =          6.0   * REBO_MOS2_b6_S;
        gp = gp * x + 5.0  * REBO_MOS2_b5_S;
        gp = gp * x + 4.0  * REBO_MOS2_b4_S;
        gp = gp * x + 3.0 * REBO_MOS2_b3_S;
        gp = gp * x + 2.0   * REBO_MOS2_b2_S;
        gp = gp * x +         REBO_MOS2_b1_S;

        if (x > 0.5)
        {
            // tmp = (gamma - G)
            double tmp =      REBO_MOS2_c6_S;
            tmp = tmp * x + REBO_MOS2_c5_S;
            tmp = tmp * x + REBO_MOS2_c4_S;
            tmp = tmp * x + REBO_MOS2_c3_S;
            tmp = tmp * x + REBO_MOS2_c2_S;
            tmp = tmp * x + REBO_MOS2_c1_S;
            tmp = tmp * x + REBO_MOS2_c0_S;

            // psi
            double psi = 0.5 * ( 1.0 - cos( TWOPI * (x-0.5) ) );

            // g = G + psi * (gamma - G)
            g += psi * tmp;

            // gp = G' + psi' * (gamma - G) now
            gp += PI * sin( TWOPI * (x-0.5) ) * tmp;

            // tmp = (gamma - G)'
            tmp =           6.0   * REBO_MOS2_c6_S;
            tmp = tmp * x + 5.0  * REBO_MOS2_c5_S;
            tmp = tmp * x + 4.0  * REBO_MOS2_c4_S;
            tmp = tmp * x + 3.0 * REBO_MOS2_c3_S;
            tmp = tmp * x + 2.0   * REBO_MOS2_c2_S;
            tmp = tmp * x +         REBO_MOS2_c1_S;

            // gp = G' + psi' * (gamma - G) + psi * (gamma - G)' now
            gp += psi * tmp;
        }
    }
}


// The angular function
static __device__ void find_g(int type1, double x, double &g)
{
    if (type1 == 0) // Mo
    {
        g =         REBO_MOS2_b6_M;
        g = g * x + REBO_MOS2_b5_M;
        g = g * x + REBO_MOS2_b4_M;
        g = g * x + REBO_MOS2_b3_M;
        g = g * x + REBO_MOS2_b2_M;
        g = g * x + REBO_MOS2_b1_M;
        g = g * x + REBO_MOS2_b0_M;

        if (x > 0.5)
        {
            // tmp = (gamma - G)
            double tmp =      REBO_MOS2_c6_M;
            tmp = tmp * x + REBO_MOS2_c5_M;
            tmp = tmp * x + REBO_MOS2_c4_M;
            tmp = tmp * x + REBO_MOS2_c3_M;
            tmp = tmp * x + REBO_MOS2_c2_M;
            tmp = tmp * x + REBO_MOS2_c1_M;
            tmp = tmp * x + REBO_MOS2_c0_M;
            
            tmp *= 0.5 * ( 1.0 - cos( TWOPI * (x-0.5) ) );
            g += tmp;
        }
    }
    else // S
    {
        g =         REBO_MOS2_b6_S;
        g = g * x + REBO_MOS2_b5_S;
        g = g * x + REBO_MOS2_b4_S;
        g = g * x + REBO_MOS2_b3_S;
        g = g * x + REBO_MOS2_b2_S;
        g = g * x + REBO_MOS2_b1_S;
        g = g * x + REBO_MOS2_b0_S;
        if (x > 0.5)
        {
            // tmp = (gamma - G)
            double tmp =      REBO_MOS2_c6_S;
            tmp = tmp * x + REBO_MOS2_c5_S;
            tmp = tmp * x + REBO_MOS2_c4_S;
            tmp = tmp * x + REBO_MOS2_c3_S;
            tmp = tmp * x + REBO_MOS2_c2_S;
            tmp = tmp * x + REBO_MOS2_c1_S;
            tmp = tmp * x + REBO_MOS2_c0_S;
            
            tmp *= 0.5 * ( 1.0 - cos( TWOPI * (x-0.5) ) );
            g += tmp;
        }
    }
}


// The coordination function and its derivative
static __device__ void find_p_and_pp(int type1, double x, double &p, double &pp)
{
    if (type1 == 0)
    {
        p = REBO_MOS2_a1_M * exp(- REBO_MOS2_a2_M * x);
        pp = p * REBO_MOS2_a2_M - REBO_MOS2_a0_M;
        p = REBO_MOS2_a3_M - REBO_MOS2_a0_M * (x - 1.0) - p;
    }
    else
    {
        p = REBO_MOS2_a1_S * exp(- REBO_MOS2_a2_S * x);
        pp = p * REBO_MOS2_a2_S - REBO_MOS2_a0_S;
        p = REBO_MOS2_a3_S - REBO_MOS2_a0_S * (x - 1.0) - p;
    }
}


// get U_ij and (d U_ij / d r_ij) / r_ij for the 2-body part
static __device__ void find_p2_and_f2(int type12, double d12, double &p2, double &f2)
{
    if (type12 == 0) // Mo-Mo
    {
        if      (d12 >= REBO_MOS2_LJCUT2_MM) { p2 = 0.0; f2 = 0.0; }
        else if (d12 >  REBO_MOS2_LJCUT1_MM)
        {
            double d12inv2 = 1.0 / (d12 * d12);
            double d12inv6 = d12inv2 * d12inv2 * d12inv2;
            p2  = REBO_MOS2_s12e4_MM * d12inv6 * d12inv6;
            p2 -= REBO_MOS2_s6e4_MM * d12inv6;
            f2  = REBO_MOS2_s6e24_MM * d12inv6;
            f2 -= REBO_MOS2_s12e48_MM * d12inv6 * d12inv6;
            f2 *= d12inv2;
        }
        else if (d12 > REBO_MOS2_r1_MM)
        {
            double dr = d12 - REBO_MOS2_r1_MM;
            p2 = (REBO_MOS2_D2_MM + REBO_MOS2_D3_MM * dr) * dr * dr;
            f2 = (REBO_MOS2_D2_MM * 2.0 + REBO_MOS2_D3_MM * 3.0 * dr) * dr;
            f2 /= d12;
        }
        else { p2 = 0.0; f2 = 0.0; }
    }
    else if (type12 == 1) // Mo-S
    {
        if      (d12 >= REBO_MOS2_LJCUT2_MS) { p2 = 0.0; f2 = 0.0; }
        else if (d12 >  REBO_MOS2_LJCUT1_MS)
        {
            double d12inv2 = 1.0 / (d12 * d12);
            double d12inv6 = d12inv2 * d12inv2 * d12inv2;
            p2  = REBO_MOS2_s12e4_MS * d12inv6 * d12inv6;
            p2 -= REBO_MOS2_s6e4_MS * d12inv6;
            f2  = REBO_MOS2_s6e24_MS * d12inv6;
            f2 -= REBO_MOS2_s12e48_MS * d12inv6 * d12inv6;
            f2 *= d12inv2;
        }
        else if (d12 > REBO_MOS2_r1_MS)
        {
            double dr = d12 - REBO_MOS2_r1_MS;
            p2 = (REBO_MOS2_D2_MS + REBO_MOS2_D3_MS * dr) * dr * dr;
            f2 = (REBO_MOS2_D2_MS * 2.0 + REBO_MOS2_D3_MS * 3.0 * dr) * dr;
            f2 /= d12;
        }
        else { p2 = 0.0; f2 = 0.0; }
    }
    else // S-S
    {
        if      (d12 >= REBO_MOS2_LJCUT2_SS) { p2 = 0.0; f2 = 0.0; }
        else if (d12 >  REBO_MOS2_LJCUT1_SS)
        {
            double d12inv2 = 1.0 / (d12 * d12);
            double d12inv6 = d12inv2 * d12inv2 * d12inv2;
            p2  = REBO_MOS2_s12e4_SS * d12inv6 * d12inv6;
            p2 -= REBO_MOS2_s6e4_SS * d12inv6;
            f2  = REBO_MOS2_s6e24_SS * d12inv6;
            f2 -= REBO_MOS2_s12e48_SS  * d12inv6 * d12inv6;
            f2 *= d12inv2;
        }
        else if (d12 > REBO_MOS2_r1_SS)
        {
            double dr = d12 - REBO_MOS2_r1_SS;
            p2 = (REBO_MOS2_D2_SS + REBO_MOS2_D3_SS * dr) * dr * dr;
            f2 = (REBO_MOS2_D2_SS * 2.0 + REBO_MOS2_D3_SS * 3.0 * dr) * dr;
            f2 /= d12;
        }
        else { p2 = 0.0; f2 = 0.0; }
    }
}


// 2-body part (kernel)
static __global__ void find_force_step0
(
    int number_of_particles, int N1, int N2, Box box,
    int *g_NN, int *g_NL, int *g_NN_local, int *g_NL_local,
    int *g_type, int shift,
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z, 
    double *g_p,  double *g_pp,
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
        double coordination_number = 0.0;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_NL[n1 + number_of_particles * i1];
            
            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            int type2 = g_type[n2] - shift;
            int type12 = type1 + type2; // 0 = AA; 1 = AB or BA; 2 = BB

            if (d12 < REBO_MOS2_r2_MM)
            {        
                // build the 3-body neighbor list            
                g_NL_local[n1 + number_of_particles * (count++)] = n2;
                // accumulate the coordination number
                double fc12; 
                find_fc(type12, d12, fc12);
                coordination_number += fc12;
            }

            double p2 = 0.0, f2 = 0.0;
            find_p2_and_f2(type12, d12, p2, f2);

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
        // save the P(N) function and its derivative
        double p, pp;
        find_p_and_pp(type1, coordination_number, p, pp);
        g_p[n1] = p;    // will be used in find_force_step1 
        g_pp[n1] = pp;  // will be used in find_force_step2

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


// Precompute the bond-order function and its derivative 
static __global__ void find_force_step1
(
    int N, int N1, int N2, Box box,
    int* g_NN, int* g_NL, int* g_type, int shift,
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z,
    double* g_b, double* g_bp, double *g_p
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1] - shift;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        double p = g_p[n1]; // coordination number function P(N)

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_NL[n1 + N * i1];

            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;

            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

            double zeta = 0.0;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_NL[n1 + N * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3] - shift;
                double x13 = g_x[n3] - x1;
                double y13 = g_y[n3] - y1;
                double z13 = g_z[n3] - z1;

                dev_apply_mic(box, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                double fc13, g123; 
                int type13 = type1 + type3;
                find_fc(type13, d13, fc13);
                find_g(type1, cos123, g123);
                zeta += fc13 * g123;
            }

            zeta += p;
            double b12 = pow(1.0 + zeta, -0.5);
            g_b[i1 * N + n1]  = b12;
            g_bp[i1 * N + n1] = (-0.5)*b12/(1.0+zeta); 
        }
    }
}


// calculate and save the partial forces dU_i/dr_ij
static __global__ void find_force_step2
(
    int N, int N1, int N2, Box box,
    int *g_NN, int *g_NL, int *g_type, int shift,
    const double* __restrict__ g_b, 
    const double* __restrict__ g_bp,
    const double* __restrict__ g_pp,
    const double* __restrict__ g_x, 
    const double* __restrict__ g_y, 
    const double* __restrict__ g_z, 
    double *g_potential, double *g_f12x, double *g_f12y, double *g_f12z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1] - shift;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        double pp1 = g_pp[n1];
        double potential_energy = 0.0;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * N + n1;
            int n2 = g_NL[index];
            int type2 = g_type[n2] - shift;
            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = 1.0 / d12;

            double fc12, fcp12, fa12, fap12, fr12, frp12;
            int type12 = type1 + type2;
            find_fc_and_fcp(type12, d12, fc12, fcp12);
            find_fa_and_fap(type12, d12, fa12, fap12);
            find_fr_and_frp(type12, d12, fr12, frp12);

            // accumulate_force_12 
            double b12 = g_b[index];
            double bp12 = g_bp[index];
            double factor3 = (fcp12*(fr12-b12*fa12) + fc12*(frp12-b12*fap12) 
                         - fc12*fcp12*fa12*bp12*pp1)/d12;
            double f12x = x12 * factor3 * 0.5;
            double f12y = y12 * factor3 * 0.5;
            double f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            potential_energy += fc12 * (fr12 - b12 * fa12) * 0.5;

            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_NL[n1 + N * i2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3] - shift;
                double x13 = g_x[n3] - x1;
                double y13 = g_y[n3] - y1;
                double z13 = g_z[n3] - z1;
                dev_apply_mic(box, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                double fc13, fa13;
                int type13 = type1 + type3;
                find_fc(type13, d13, fc13);
                find_fa(type13, d13, fa13);
                double bp13 = g_bp[i2 * N + n1];
                double one_over_d12d13 = 1.0 / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double g123, gp123;
                find_g_and_gp(type1, cos123, g123, gp123);

                double temp123a=(-bp12*fc12*fa12*fc13-bp13*fc13*fa13*fc12)*gp123;
                double temp123b= - bp13 * fc13 * fa13 * fcp12 * (g123+pp1) / d12;
                double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * temp123b + temp123a * cos_d)*0.5;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * temp123b + temp123a * cos_d)*0.5;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * temp123b + temp123a * cos_d)*0.5;
            }
            g_f12x[index] = f12x;
            g_f12y[index] = f12y;
            g_f12z[index] = f12z;
        }
        // accumulate potential energy on top of the 2-body part
        g_potential[n1] += potential_energy;
    }
}


// Force evaluation wrapper
void REBO_MOS::compute(Atom *atom, int potential_number)
{
    int N = atom->N;
    int shift = atom->shift[potential_number];
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

    int *NN = atom->neighbor.NN_local.data(); // for 2-body
    int *NL = atom->neighbor.NL_local.data(); // for 2-body
    int *NN_local = rebo_mos_data.NN_short.data(); // for 3-body
    int *NL_local = rebo_mos_data.NL_short.data(); // for 3-body

    int *type = atom->type.data();
    double *x = atom->position_per_atom.data();
    double *y = atom->position_per_atom.data() + atom->N;
    double *z = atom->position_per_atom.data() + atom->N * 2;
    double *fx = atom->force_per_atom.data();
    double *fy = atom->force_per_atom.data() + atom->N;
    double *fz = atom->force_per_atom.data() + 2 * atom->N;
    double *virial = atom->virial_per_atom.data();
    double *pe = atom->potential_per_atom.data();

    double *b    = rebo_mos_data.b.data();
    double *bp   = rebo_mos_data.bp.data();
    double *p    = rebo_mos_data.p.data();
    double *pp   = rebo_mos_data.pp.data();
    double *f12x = rebo_mos_data.f12x.data();
    double *f12y = rebo_mos_data.f12y.data();
    double *f12z = rebo_mos_data.f12z.data();

    // 2-body part
    find_force_step0<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, NN, NL, NN_local, NL_local, type, shift,
        x, y, z, p, pp, fx, fy, fz, virial, pe
    );
    CUDA_CHECK_KERNEL

    // pre-compute the bond-order function and its derivative
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, NN_local, NL_local,
        type, shift, x, y, z, b, bp, p
    );
    CUDA_CHECK_KERNEL

    // pre-compute the partial force
    find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, NN_local, NL_local,
        type, shift, b, bp, pp, x, y, z, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL

    // 3-body part
    find_properties_many_body(atom, NN_local, NL_local, f12x, f12y, f12z);
}


