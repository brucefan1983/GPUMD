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
#include "mic.cu" // static __device__ dev_apply_mic(...)
#include "rebo_mos2.h"

// References: 
// [1] T. Liang et al. PRB 79, 245110 (2009).
// [2] T. Liang et al. PRB 85, 199903(E) (2012).

// The parameters are hard coded as the potential only applies to Mo-S systems.


#ifdef USE_DP

#define TWOPI 6.283185307179586

#define REBO_MOS2_Q_MM     3.41912939005919
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

#define REBO_MOS2_a0_M   0.138040769883614
#define REBO_MOS2_a1_M   0.599874419749679
#define REBO_MOS2_a2_M   0.292412960851064
#define REBO_MOS2_a3_M   0.502547309062610

#define REBO_MOS2_a0_S   0.062978539843324
#define REBO_MOS2_a1_S   2.38938198826146
#define REBO_MOS2_a2_S   0.036666243238154
#define REBO_MOS2_a3_S   2.32345283264339

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

#else

#define TWOPI 6.283185307179586f

#define REBO_MOS2_Q_MM     3.41912939005919f
#define REBO_MOS2_A_MM     179.008013654688f
#define REBO_MOS2_B_MM     706.247903589221f
#define REBO_MOS2_a_MM     1.0750071299934f
#define REBO_MOS2_b_MM     1.16100322369589f
#define REBO_MOS2_r1_MM    3.5f
#define REBO_MOS2_r2_MM    3.8f

#define REBO_MOS2_Q_SS     0.254959104053671f
#define REBO_MOS2_A_SS     1228.43233679426f
#define REBO_MOS2_B_SS     1498.64815404145f
#define REBO_MOS2_a_SS     1.10775022439715f
#define REBO_MOS2_b_SS     1.1267362361032f
#define REBO_MOS2_r1_SS    2.3f
#define REBO_MOS2_r2_SS    3.0f

#define REBO_MOS2_Q_MS    1.50553783915379f
#define REBO_MOS2_A_MS    575.509677721866f
#define REBO_MOS2_B_MS    1344.46820036159f
#define REBO_MOS2_a_MS    1.1926790221882f
#define REBO_MOS2_b_MS    1.2697375220429f
#define REBO_MOS2_r1_MS   2.75f
#define REBO_MOS2_r2_MS   3.05f

#define REBO_MOS2_pi_factor_MM    10.471975511965978f
#define REBO_MOS2_pi_factor_SS    4.487989505128276f
#define REBO_MOS2_pi_factor_MS    10.471975511965978f

#define REBO_MOS2_a0_M   0.138040769883614f
#define REBO_MOS2_a1_M   0.599874419749679f
#define REBO_MOS2_a2_M   0.292412960851064f
#define REBO_MOS2_a3_M   0.502547309062610f

#define REBO_MOS2_a0_S   0.062978539843324f
#define REBO_MOS2_a1_S   2.38938198826146f
#define REBO_MOS2_a2_S   0.036666243238154f
#define REBO_MOS2_a3_S   2.32345283264339f

// G
#define REBO_MOS2_b0_M   0.132684255066327f
#define REBO_MOS2_b1_M  -0.007642788338017f
#define REBO_MOS2_b2_M   0.034139577505937f
#define REBO_MOS2_b3_M   0.252305097138087f
#define REBO_MOS2_b4_M   0.122728737222567f
#define REBO_MOS2_b5_M  -0.361387798398897f
#define REBO_MOS2_b6_M  -0.282577591351457f

#define REBO_MOS2_b0_S   0.00684876159675f
#define REBO_MOS2_b1_S  -0.02389964401024f
#define REBO_MOS2_b2_S   0.13745735331117f
#define REBO_MOS2_b3_S   0.03301646749774f
#define REBO_MOS2_b4_S  -0.3106429154485f
#define REBO_MOS2_b5_S  -0.08550273135791f
#define REBO_MOS2_b6_S   0.14925279030688f

// gamma - G
#define REBO_MOS2_c0_M  -0.012489954031047f
#define REBO_MOS2_c1_M   0.052881075696207f
#define REBO_MOS2_c2_M   0.033783229738093f
#define REBO_MOS2_c3_M  -0.289030210924907f
#define REBO_MOS2_c4_M  -0.015212259708707f
#define REBO_MOS2_c5_M   0.366352510383837f
#define REBO_MOS2_c6_M   0.152601607764937f

#define REBO_MOS2_c0_S  -0.291933961596750f
#define REBO_MOS2_c1_S   1.694924444010240f
#define REBO_MOS2_c2_S  -3.705308953311170f
#define REBO_MOS2_c3_S   3.417533432502260f
#define REBO_MOS2_c4_S  -0.907985984551500f
#define REBO_MOS2_c5_S   0.085502731357910f
#define REBO_MOS2_c6_S  -0.149252790306880f


#endif



// best block size here: 64 or 128
#define BLOCK_SIZE_FORCE 64




// The repulsive function and its derivative
static __device__ void find_fr_and_frp
(int type1, int type2, real d12, real &fr, real &frp)
{     
    if (type1 == 0 && type2 == 0)
    {   
        fr  = (ONE + REBO_MOS2_Q_MM / d12) * REBO_MOS2_A_MM 
            * exp(-REBO_MOS2_a_MM * d12);  
        frp = REBO_MOS2_a_MM + REBO_MOS2_Q_MM / (d12 * (d12 + REBO_MOS2_Q_MM));
        frp *= -fr;
    }
    else if (type1 == 1 && type2 == 1)
    {
        fr  = (ONE + REBO_MOS2_Q_SS / d12) * REBO_MOS2_A_SS 
            * exp(-REBO_MOS2_a_SS * d12);  
        frp = REBO_MOS2_a_SS + REBO_MOS2_Q_SS / (d12 * (d12 + REBO_MOS2_Q_SS));
        frp *= -fr;
    }  
    else
    {
        fr  = (ONE + REBO_MOS2_Q_MS / d12) * REBO_MOS2_A_MS 
            * exp(-REBO_MOS2_a_MS * d12);  
        frp = REBO_MOS2_a_MS + REBO_MOS2_Q_MS / (d12 * (d12 + REBO_MOS2_Q_MS));
        frp *= -fr;
    }   
}



// The attractive function and its derivative
static __device__ void find_fa_and_fap
(int type1, int type2, real d12, real &fa, real &fap)
{  
    if (type1 == 0 && type2 == 0)
    {   
        fa  = REBO_MOS2_B_MM * exp(- REBO_MOS2_b_MM * d12); 
        fap = - REBO_MOS2_b_MM * fa;
    }
    else if (type1 == 1 && type2 == 1)
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
(int type1, int type2, real d12, real &fa)
{  
    if (type1 == 0 && type2 == 0)
    {   
        fa  = REBO_MOS2_B_MM * exp(- REBO_MOS2_b_MM * d12); 
    }
    else if (type1 == 1 && type2 == 1)
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
(int type1, int type2, real d12, real &fc, real &fcp)
{
    if (type1 == 0 && type2 == 0)
    { 
        if (d12 < REBO_MOS2_r1_MM) {fc = ONE; fcp = ZERO;}
        else if (d12 < REBO_MOS2_r2_MM)
        {              
            fc  = cos(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM)) 
                * HALF + HALF;
            fcp = -sin(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM))
                * REBO_MOS2_pi_factor_MM * HALF;
        }
        else {fc  = ZERO; fcp = ZERO;}
    }
    else if (type1 == 1 && type2 == 1)
    { 
        if (d12 < REBO_MOS2_r1_SS) {fc = ONE; fcp = ZERO;}
        else if (d12 < REBO_MOS2_r2_SS)
        {              
            fc  = cos(REBO_MOS2_pi_factor_SS * (d12 - REBO_MOS2_r1_SS)) 
                * HALF + HALF;
            fcp = -sin(REBO_MOS2_pi_factor_SS * (d12 - REBO_MOS2_r1_SS))
                * REBO_MOS2_pi_factor_SS * HALF;
        }
        else {fc  = ZERO; fcp = ZERO;}
    }
    else  
    { 
        if (d12 < REBO_MOS2_r1_MS) {fc = ONE; fcp = ZERO;}
        else if (d12 < REBO_MOS2_r2_MS)
        {              
            fc  = cos(REBO_MOS2_pi_factor_MS * (d12 - REBO_MOS2_r1_MS)) 
                * HALF + HALF;
            fcp = -sin(REBO_MOS2_pi_factor_MS * (d12 - REBO_MOS2_r1_MS))
                * REBO_MOS2_pi_factor_MS * HALF;
        }
        else {fc  = ZERO; fcp = ZERO;}
    }
}


// The cutoff function
static __device__ void find_fc(int type1, int type2, real d12, real &fc)
{
    if (type1 == 0 && type2 == 0)
    { 
        if (d12 < REBO_MOS2_r1_MM) {fc = ONE;}
        else if (d12 < REBO_MOS2_r2_MM)
        {              
            fc  = cos(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM)) 
                * HALF + HALF;
        }
        else {fc  = ZERO;}
    }
    else if (type1 == 1 && type2 == 1)
    { 
        if (d12 < REBO_MOS2_r1_SS) {fc = ONE;}
        else if (d12 < REBO_MOS2_r2_SS)
        {              
            fc  = cos(REBO_MOS2_pi_factor_SS * (d12 - REBO_MOS2_r1_SS)) 
                * HALF + HALF;
        }
        else {fc  = ZERO;}
    }
    else  
    { 
        if (d12 < REBO_MOS2_r1_MS) {fc = ONE;}
        else if (d12 < REBO_MOS2_r2_MS)
        {              
            fc  = cos(REBO_MOS2_pi_factor_MS * (d12 - REBO_MOS2_r1_MS)) 
                * HALF + HALF;
        }
        else {fc  = ZERO;}
    }
}


// The angular function and its derivative
static __device__ void find_g_and_gp(int type1, real x, real &g, real &gp)
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
            
        gp =          SIX   * REBO_MOS2_b6_M; 
        gp = gp * x + FIVE  * REBO_MOS2_b5_M;
        gp = gp * x + FOUR  * REBO_MOS2_b4_M;
        gp = gp * x + THREE * REBO_MOS2_b3_M;
        gp = gp * x + TWO   * REBO_MOS2_b2_M;
        gp = gp * x +         REBO_MOS2_b1_M;

        if (x > HALF)
        { 
            // tmp = (gamma - G)
            real tmp =      REBO_MOS2_c6_M;
            tmp = tmp * x + REBO_MOS2_c5_M;
            tmp = tmp * x + REBO_MOS2_c4_M;
            tmp = tmp * x + REBO_MOS2_c3_M;
            tmp = tmp * x + REBO_MOS2_c2_M;
            tmp = tmp * x + REBO_MOS2_c1_M;
            tmp = tmp * x + REBO_MOS2_c0_M;
            
            // psi
            real psi = HALF * ( ONE - cos( TWOPI * (x-HALF) ) );
            
            // g = G + psi * (gamma - G)
            g += psi * tmp;
            
            // gp = G' + psi' * (gamma - G) now
            gp += PI * sin( TWOPI * (x-HALF) ) * tmp;
            
            // tmp = (gamma - G)'
            tmp =           SIX   * REBO_MOS2_c6_M;
            tmp = tmp * x + FIVE  * REBO_MOS2_c5_M;
            tmp = tmp * x + FOUR  * REBO_MOS2_c4_M;
            tmp = tmp * x + THREE * REBO_MOS2_c3_M;
            tmp = tmp * x + TWO   * REBO_MOS2_c2_M;
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
            
        gp =          SIX   * REBO_MOS2_b6_S; 
        gp = gp * x + FIVE  * REBO_MOS2_b5_S;
        gp = gp * x + FOUR  * REBO_MOS2_b4_S;
        gp = gp * x + THREE * REBO_MOS2_b3_S;
        gp = gp * x + TWO   * REBO_MOS2_b2_S;
        gp = gp * x +         REBO_MOS2_b1_S;

        if (x > HALF)
        {    
            // tmp = (gamma - G)
            real tmp =      REBO_MOS2_c6_S;
            tmp = tmp * x + REBO_MOS2_c5_S;
            tmp = tmp * x + REBO_MOS2_c4_S;
            tmp = tmp * x + REBO_MOS2_c3_S;
            tmp = tmp * x + REBO_MOS2_c2_S;
            tmp = tmp * x + REBO_MOS2_c1_S;
            tmp = tmp * x + REBO_MOS2_c0_S;
            
            // psi
            real psi = HALF * ( ONE - cos( TWOPI * (x-HALF) ) );
            
            // g = G + psi * (gamma - G)
            g += psi * tmp;

            // gp = G' + psi' * (gamma - G) now
            gp += PI * sin( TWOPI * (x-HALF) ) * tmp;
            
            // tmp = (gamma - G)'
            tmp =           SIX   * REBO_MOS2_c6_S;
            tmp = tmp * x + FIVE  * REBO_MOS2_c5_S;
            tmp = tmp * x + FOUR  * REBO_MOS2_c4_S;
            tmp = tmp * x + THREE * REBO_MOS2_c3_S;
            tmp = tmp * x + TWO   * REBO_MOS2_c2_S;
            tmp = tmp * x +         REBO_MOS2_c1_S;
            
            // gp = G' + psi' * (gamma - G) + psi * (gamma - G)' now
            gp += psi * tmp;
        }
    }
}




// The angular function
static __device__ void find_g(int type1, real x, real &g)
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
            
        if (x > HALF)
        {
            // tmp = (gamma - G)
            real tmp =      REBO_MOS2_c6_M;
            tmp = tmp * x + REBO_MOS2_c5_M;
            tmp = tmp * x + REBO_MOS2_c4_M;
            tmp = tmp * x + REBO_MOS2_c3_M;
            tmp = tmp * x + REBO_MOS2_c2_M;
            tmp = tmp * x + REBO_MOS2_c1_M;
            tmp = tmp * x + REBO_MOS2_c0_M;
            
            tmp *= HALF * ( ONE - cos( TWOPI * (x-HALF) ) );
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
        if (x > HALF)
        {
            // tmp = (gamma - G)
            real tmp =      REBO_MOS2_c6_S;
            tmp = tmp * x + REBO_MOS2_c5_S;
            tmp = tmp * x + REBO_MOS2_c4_S;
            tmp = tmp * x + REBO_MOS2_c3_S;
            tmp = tmp * x + REBO_MOS2_c2_S;
            tmp = tmp * x + REBO_MOS2_c1_S;
            tmp = tmp * x + REBO_MOS2_c0_S;
            
            tmp *= HALF * ( ONE - cos( TWOPI * (x-HALF) ) );
            g += tmp;
        }
    }
}


// The coordination function and its derivative
static __device__ void find_p_and_pp(int type1, real x, real &p, real &pp)
{
    if (type1 == 0)
    {
        p = REBO_MOS2_a1_M * exp(- REBO_MOS2_a2_M * x);
        pp = p * REBO_MOS2_a2_M - REBO_MOS2_a0_M;
        p = REBO_MOS2_a3_M - REBO_MOS2_a0_M * (x - ONE) - p;
    }
    else
    {
        p = REBO_MOS2_a1_S * exp(- REBO_MOS2_a2_S * x);
        pp = p * REBO_MOS2_a2_S - REBO_MOS2_a0_S;
        p = REBO_MOS2_a3_S - REBO_MOS2_a0_S * (x - ONE) - p;        
    }
}


// Precompute the bond-order function and its derivative 
static __global__ void find_force_step1
(
    int N, int pbc_x, int pbc_y, int pbc_z,
    int* g_NN, int* g_NL, int* g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z,
#else
    real* g_x, real* g_y, real* g_z,
#endif
    real *g_box,
    #ifdef TRICLINIC
    real *g_box_inv,
    #endif
    real* g_b, real* g_bp, real*g_pp
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        
        #ifndef TRICLINIC
        real lx = LDG(g_box, 0); 
        real ly = LDG(g_box, 1); 
        real lz = LDG(g_box, 2);
        #endif

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {      
            int n2 = g_NL[n1 + N * i1];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;

            #ifdef TRICLINIC
            apply_mic(pbc_x, pbc_y, pbc_z, g_box, g_box_inv, x12, y12, z12);
            #else
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            #endif

            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real zeta = ZERO;
            real n12 = ZERO; // coordination number
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_NL[n1 + N * i2];  
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;         

                #ifdef TRICLINIC
                apply_mic(pbc_x, pbc_y, pbc_z, g_box, g_box_inv, x13, y13, z13);
                #else
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                #endif

                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                real cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                real fc13, g123; 
                find_fc(type1, type3, d13, fc13);
                find_g(type1, cos123, g123);
                zeta += fc13 * g123;
                n12 += fc13; 
            }
            
            real p12, pp12;
            find_p_and_pp(type1, n12, p12, pp12); 
            zeta += p12;
            
            real b12 = pow(ONE + zeta, -HALF);
            g_b[i1 * N + n1]  = b12;
            g_bp[i1 * N + n1] = (-HALF)*b12/(ONE+zeta); 
            g_pp[i1 * N + n1] = pp12;
        }
    }
}


// Force evaluation kernel
template <int cal_p, int cal_j, int cal_q>
static __global__ void find_force_step2
(
    int N, int pbc_x, int pbc_y, int pbc_z,
    int *g_NN, int *g_NL, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_b, 
    const real* __restrict__ g_bp,
    const real* __restrict__ g_pp,
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
    const real* __restrict__ g_vx, 
    const real* __restrict__ g_vy, 
    const real* __restrict__ g_vz,
#else
    real* g_b, real* g_bp, real* g_pp, real* g_x, real* g_y, real* g_z, 
    real* g_vx, real* g_vy, real* g_vz,
#endif  
    real *g_box,
    #ifdef TRICLINIC
    real *g_box_inv,
    #endif
    real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz, real *g_potential, 
    real *g_h, int *g_label, int *g_fv_index, real *g_fv 
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ real s_fx[BLOCK_SIZE_FORCE];
    __shared__ real s_fy[BLOCK_SIZE_FORCE];
    __shared__ real s_fz[BLOCK_SIZE_FORCE];

    // if cal_p, then s1~s4 = px, py, pz, U; if cal_j, then s1~s5 = j1~j5
    __shared__ real s1[BLOCK_SIZE_FORCE];
    __shared__ real s2[BLOCK_SIZE_FORCE];
    __shared__ real s3[BLOCK_SIZE_FORCE];
    __shared__ real s4[BLOCK_SIZE_FORCE];
    __shared__ real s5[BLOCK_SIZE_FORCE];

    s_fx[threadIdx.x] = ZERO; 
    s_fy[threadIdx.x] = ZERO; 
    s_fz[threadIdx.x] = ZERO;  

    s1[threadIdx.x] = ZERO; 
    s2[threadIdx.x] = ZERO; 
    s3[threadIdx.x] = ZERO;
    s4[threadIdx.x] = ZERO;
    s5[threadIdx.x] = ZERO;

    if (n1 < N)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real vx1 = LDG(g_vx, n1); 
        real vy1 = LDG(g_vy, n1); 
        real vz1 = LDG(g_vz, n1);
        
        #ifndef TRICLINIC
        real lx = LDG(g_box, 0); 
        real ly = LDG(g_box, 1); 
        real lz = LDG(g_box, 2);
        #endif

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = g_NL[n1 + N * i1];
            int neighbor_number_2 = g_NN[n2];
            int type2 = g_type[n2];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;

            #ifdef TRICLINIC
            apply_mic(pbc_x, pbc_y, pbc_z, g_box, g_box_inv, x12, y12, z12);
            #else
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            #endif

            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real fc12, fcp12, fa12, fap12, fr12, frp12;
            find_fc_and_fcp(type1, type2, d12, fc12, fcp12);
            find_fa_and_fap(type1, type2, d12, fa12, fap12);
            find_fr_and_frp(type1, type2, d12, fr12, frp12);
            real f12x = ZERO; real f12y = ZERO; real f12z = ZERO;
            real f21x = ZERO; real f21y = ZERO; real f21z = ZERO;
         
            // accumulate_force_12 
            real b12 = LDG(g_b, i1 * N + n1);    
            real factor3 = (fcp12*(fr12-b12*fa12)+fc12*(frp12-b12*fap12))/d12;   
            f12x += x12 * factor3 * HALF; 
            f12y += y12 * factor3 * HALF;
            f12z += z12 * factor3 * HALF;

            if (cal_p) // accumulate potential energy
            {
                s4[threadIdx.x] += fc12 * (fr12 - b12 * fa12) * HALF;
            }

            // accumulate_force_21
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_NL[n2 + N * k]) 
                { 
                    offset = k; break; 
                }
            }
            // b12 here actually means b21
            b12 = LDG(g_b, offset * N + n2);
            factor3 = (fcp12*(fr12-b12*fa12)+fc12*(frp12-b12*fap12))/d12;   
            f21x -= x12 * factor3 * HALF; 
            f21y -= y12 * factor3 * HALF;
            f21z -= z12 * factor3 * HALF;      

            // accumulate_force_123
            real bp12 = LDG(g_bp, i1 * N + n1);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_NL[n1 + N * i2];   
                if (n3 == n2) { continue; } 
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;

                #ifdef TRICLINIC
                apply_mic(pbc_x, pbc_y, pbc_z, g_box, g_box_inv, x13, y13, z13);
                #else
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                #endif

                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);   
                real fc13, fa13;
                find_fc(type1, type3, d13, fc13);
                find_fa(type1, type3, d13, fa13); 
                real bp13 = LDG(g_bp, i2 * N + n1);
                real pp13 = LDG(g_pp, i2 * N + n1); // extra term for REBO-MoS2
                real cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                real g123, gp123;
                find_g_and_gp(type1, cos123, g123, gp123);
                real cos_x = x13 / (d12 * d13) - x12 * cos123 / (d12 * d12);
                real cos_y = y13 / (d12 * d13) - y12 * cos123 / (d12 * d12);
                real cos_z = z13 / (d12 * d13) - z12 * cos123 / (d12 * d12);
                real temp123a=(-bp12*fc12*fa12*fc13-bp13*fc13*fa13*fc12)*gp123;
                real temp123b= - bp13 * fc13 * fa13 * fcp12 * (g123+pp13) / d12;
                f12x += (x12 * temp123b + temp123a * cos_x)*HALF; 
                f12y += (y12 * temp123b + temp123a * cos_y)*HALF;
                f12z += (z12 * temp123b + temp123a * cos_z)*HALF;
            }

            // accumulate_force_213 (bp12 here actually means bp21)
            bp12 = LDG(g_bp, offset * N + n2); 
            for (int i2 = 0; i2 < neighbor_number_2; ++i2)
            {
                int n3 = g_NL[n2 + N * i2];      
                if (n3 == n1) { continue; } 
                int type3 = g_type[n3];
                real x23 = LDG(g_x, n3) - LDG(g_x, n2);
                real y23 = LDG(g_y, n3) - LDG(g_y, n2);
                real z23 = LDG(g_z, n3) - LDG(g_z, n2);

                #ifdef TRICLINIC
                apply_mic(pbc_x, pbc_y, pbc_z, g_box, g_box_inv, x23, y23, z23);
                #else
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x23, y23, z23, lx, ly, lz);
                #endif

                real d23 = sqrt(x23 * x23 + y23 * y23 + z23 * z23);     
                real fc23, fa23;
                find_fc(type2, type3, d23, fc23);
                find_fa(type2, type3, d23, fa23);
                real bp23 = LDG(g_bp, i2 * N + n2);
                real pp23 = LDG(g_pp, i2 * N + n2); // extra term for REBO-MoS2 
                real cos213 = - (x12 * x23 + y12 * y23 + z12 * z23)/(d12 * d23);
                real g213, gp213;
                find_g_and_gp(type2, cos213, g213, gp213);
                real cos_x = x23 / (d12 * d23) + x12 * cos213 / (d12 * d12);
                real cos_y = y23 / (d12 * d23) + y12 * cos213 / (d12 * d12);
                real cos_z = z23 / (d12 * d23) + z12 * cos213 / (d12 * d12);
                real temp213a=(-bp12*fc12*fa12*fc23-bp23*fc23*fa23*fc12)*gp213;
                real temp213b= - bp23 * fc23 * fa23 * fcp12 * (g213+pp23) / d12;
                f21x += (-x12 * temp213b + temp213a * cos_x)*HALF; 
                f21y += (-y12 * temp213b + temp213a * cos_y)*HALF;
                f21z += (-z12 * temp213b + temp213a * cos_z)*HALF;
            }  
  
            // per atom force
            s_fx[threadIdx.x] += f12x - f21x; 
            s_fy[threadIdx.x] += f12y - f21y; 
            s_fz[threadIdx.x] += f12z - f21z;  

            // per-atom stress
            if (cal_p)
            {
                s1[threadIdx.x] -= x12 * (f12x - f21x) * HALF; 
                s2[threadIdx.x] -= y12 * (f12y - f21y) * HALF; 
                s3[threadIdx.x] -= z12 * (f12z - f21z) * HALF;
            }

            // per-atom heat current
            if (cal_j)
            {
                s1[threadIdx.x] += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s2[threadIdx.x] += (f21z * vz1) * x12;               // x-out
                s3[threadIdx.x] += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s4[threadIdx.x] += (f21z * vz1) * y12;               // y-out
                s5[threadIdx.x] += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            }
 
            // accumulate heat across some sections (for NEMD)
            if (cal_q)
            {
                int index_12 = g_fv_index[n1] * 12;
                if (index_12 >= 0 && g_fv_index[n1 + N] == n2)
                {
                    g_fv[index_12 + 0]  = f12x;
                    g_fv[index_12 + 1]  = f12y;
                    g_fv[index_12 + 2]  = f12z;
                    g_fv[index_12 + 3]  = f21x;
                    g_fv[index_12 + 4]  = f21y;
                    g_fv[index_12 + 5]  = f21z;
                    g_fv[index_12 + 6]  = vx1;
                    g_fv[index_12 + 7]  = vy1;
                    g_fv[index_12 + 8]  = vz1;
                    g_fv[index_12 + 9]  = LDG(g_vx, n2);
                    g_fv[index_12 + 10] = LDG(g_vy, n2);
                    g_fv[index_12 + 11] = LDG(g_vz, n2);
                }  
            }
            
        }

        // save force
        g_fx[n1] = s_fx[threadIdx.x]; 
        g_fy[n1] = s_fy[threadIdx.x]; 
        g_fz[n1] = s_fz[threadIdx.x];

        if (cal_p) // save stress and potential
        {
            g_sx[n1] = s1[threadIdx.x]; 
            g_sy[n1] = s2[threadIdx.x]; 
            g_sz[n1] = s3[threadIdx.x];
            g_potential[n1] = s4[threadIdx.x];
        }

        if (cal_j) // save heat current
        {
            g_h[n1 + 0 * N] = s1[threadIdx.x];
            g_h[n1 + 1 * N] = s2[threadIdx.x];
            g_h[n1 + 2 * N] = s3[threadIdx.x];
            g_h[n1 + 3 * N] = s4[threadIdx.x];
            g_h[n1 + 4 * N] = s5[threadIdx.x];
        }

    }
}   




// Force evaluation wrapper
void gpu_find_force_rebo_mos2(Parameters *para, GPU_Data *gpu_data)
{
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE_FORCE + 1;
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
#ifdef FIXED_NL
    int *NN = gpu_data->NN; 
    int *NL = gpu_data->NL;
#else
    int *NN = gpu_data->NN_local; 
    int *NL = gpu_data->NL_local;
#endif
    int *type = gpu_data->type;
    real *x = gpu_data->x; 
    real *y = gpu_data->y; 
    real *z = gpu_data->z;
    real *vx = gpu_data->vx; 
    real *vy = gpu_data->vy; 
    real *vz = gpu_data->vz;
    real *fx = gpu_data->fx; 
    real *fy = gpu_data->fy; 
    real *fz = gpu_data->fz;
    real *b = gpu_data->b; 
    real *bp = gpu_data->bp; 
   
    #ifdef TRICLINIC
    real *box     = gpu_data->box_matrix;
    real *box_inv = gpu_data->box_matrix_inv;
    #else
    real *box = gpu_data->box_length;
    #endif

    real *sx = gpu_data->virial_per_atom_x; 
    real *sy = gpu_data->virial_per_atom_y; 
    real *sz = gpu_data->virial_per_atom_z; 
    real *pe = gpu_data->potential_per_atom;
    real *h = gpu_data->heat_per_atom;   
    
    int *label = gpu_data->label;
    int *fv_index = gpu_data->fv_index;
    real *fv = gpu_data->fv;
    
    real *pp;
    cudaMalloc((void**)&pp, sizeof(real) * N * para->neighbor.MN);
    
    #ifdef TRICLINIC
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (N, pbc_x, pbc_y, pbc_z, NN, NL, type, x, y, z, box, box_inv, b, bp, pp);
    #else
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (N, pbc_x, pbc_y, pbc_z, NN, NL, type, x, y, z, box, b, bp, pp);
    #endif

    if (para->hac.compute)
    {
        #ifdef TRICLINIC
        find_force_step2<0, 1, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, 
            b, bp, pp, x, y, z, vx, vy, vz, 
            box, box_inv, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
        #else
        find_force_step2<0, 1, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, 
            b, bp, pp, x, y, z, vx, vy, vz, 
            box, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
        #endif
    }
    else if (para->shc.compute)
    {
        #ifdef TRICLINIC
        find_force_step2<0, 0, 1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, 
            b, bp, pp, x, y, z, vx, vy, vz, 
            box, box_inv, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
        #else
        find_force_step2<0, 0, 1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, 
            b, bp, pp, x, y, z, vx, vy, vz, 
            box, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
        #endif
    }
    else
    {
        #ifdef TRICLINIC
        find_force_step2<1, 0, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, 
            b, bp, pp, x, y, z, vx, vy, vz, 
            box, box_inv, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
        #else
        find_force_step2<1, 0, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, 
            b, bp, pp, x, y, z, vx, vy, vz, 
            box, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
        #endif
    }
    
    cudaFree(pp);
}




