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
#include "mic.inc"
#include "force.inc"
#include "rebo_mos2.cuh"
#include "ldg.cuh"
#include "measure.cuh"

#define BLOCK_SIZE_FORCE 64

// References: 
// [1] T. Liang et al. PRB 79, 245110 (2009).
// [2] T. Liang et al. PRB 85, 199903(E) (2012).
// [3] J. A. Stewart et al. MSMSE 21, 045003 (2013).
// We completely followed Ref. [3] and Stewart's LAMMPS implementation
// The parameters are hard coded as the potential only applies to Mo-S systems.


#ifdef USE_DP

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

// LJ (From Stewart)

/*
#define REBO_MOS2_EPSILON_MM  0.00058595
#define REBO_MOS2_EPSILON_SS  0.01386
#define REBO_MOS2_EPSILON_MS  0.002849783676001

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
#define REBO_MOS2_s12e4_SS    49017.96441039995
#define REBO_MOS2_s12e4_MS    66951.39547508015

// 4 * epsilon * sigma^6
#define REBO_MOS2_s6e4_MM     12.865192601587204
#define REBO_MOS2_s6e4_SS     52.130182686353336
#define REBO_MOS2_s6e4_MS     27.625857011890133

// 48 * epsilon * sigma^12
#define REBO_MOS2_s12e48_MM   847409.4069934335
#define REBO_MOS2_s12e48_SS   588215.5729247995
#define REBO_MOS2_s12e48_MS   803416.7457009618

// 24 * epsilon * sigma^6
#define REBO_MOS2_s6e24_MM    77.19115560952322
#define REBO_MOS2_s6e24_SS    312.7810961181200
#define REBO_MOS2_s6e24_MS    165.7551420713408

// pre-computed coefficient of the (r - r1)^2 term
#define REBO_MOS2_D2_MM      0.031194467724753
#define REBO_MOS2_D2_SS      0.568571579107700
#define REBO_MOS2_D2_MS      0.094150096066930

// pre-computed coefficient of the (r - r1)^3 term
#define REBO_MOS2_D3_MM     -0.053895558827613
#define REBO_MOS2_D3_SS     -0.755238613586170
#define REBO_MOS2_D3_MS     -0.114401791730260
*/

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

// From Stewart
#define REBO_MOS2_a0_M   0.138040769883614f
#define REBO_MOS2_a1_M   0.803625443023934f
#define REBO_MOS2_a2_M   0.292412960851064f
#define REBO_MOS2_a3_M   0.640588078946224f

#define REBO_MOS2_a0_S   0.062978539843324f
#define REBO_MOS2_a1_S   2.478617619878250f
#define REBO_MOS2_a2_S   0.036666243238154f
#define REBO_MOS2_a3_S   2.386431372486710f

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

// LJ (From Stewart)

/*
#define REBO_MOS2_EPSILON_MM  0.00058595f
#define REBO_MOS2_EPSILON_SS  0.01386f
#define REBO_MOS2_EPSILON_MS  0.002849783676001f

#define REBO_MOS2_SIGMA_MM    4.2f
#define REBO_MOS2_SIGMA_SS    3.13f
#define REBO_MOS2_SIGMA_MS    3.665f

// 0.95 * sigma
#define REBO_MOS2_LJCUT1_MM   3.99f
#define REBO_MOS2_LJCUT1_SS   2.9735f
#define REBO_MOS2_LJCUT1_MS   3.48175f

// 2.5 * sigma
#define REBO_MOS2_LJCUT2_MM   10.5f
#define REBO_MOS2_LJCUT2_SS   7.825f
#define REBO_MOS2_LJCUT2_MS   9.1625f

// 4 * epsilon * sigma^12
#define REBO_MOS2_s12e4_MM    70617.45058278613f
#define REBO_MOS2_s12e4_SS    49017.96441039995f
#define REBO_MOS2_s12e4_MS    66951.39547508015f

// 4 * epsilon * sigma^6
#define REBO_MOS2_s6e4_MM     12.865192601587204f
#define REBO_MOS2_s6e4_SS     52.130182686353336f
#define REBO_MOS2_s6e4_MS     27.625857011890133f

// 48 * epsilon * sigma^12
#define REBO_MOS2_s12e48_MM   847409.4069934335f
#define REBO_MOS2_s12e48_SS   588215.5729247995f
#define REBO_MOS2_s12e48_MS   789381.6316330632f

// 24 * epsilon * sigma^6
#define REBO_MOS2_s6e24_MM    77.19115560952322f
#define REBO_MOS2_s6e24_SS    312.7810961181200f
#define REBO_MOS2_s6e24_MS    165.7551420713408f

// pre-computed coefficient of the (r - r1)^2 term
#define REBO_MOS2_D2_MM      0.031194467724753f
#define REBO_MOS2_D2_SS      0.568571579107700f
#define REBO_MOS2_D2_MS      0.094150096066930f

// pre-computed coefficient of the (r - r1)^3 term
#define REBO_MOS2_D3_MM     -0.053895558827613f
#define REBO_MOS2_D3_SS     -0.755238613586170f
#define REBO_MOS2_D3_MS     -0.114401791730260f
*/


// LJ (From Liang)

#define REBO_MOS2_EPSILON_MM  0.00058595f
#define REBO_MOS2_EPSILON_SS  0.02f
#define REBO_MOS2_EPSILON_MS  0.003423302499050f

#define REBO_MOS2_SIGMA_MM    4.2f
#define REBO_MOS2_SIGMA_SS    3.13f
#define REBO_MOS2_SIGMA_MS    3.665f

// 0.95 * sigma
#define REBO_MOS2_LJCUT1_MM   3.99f
#define REBO_MOS2_LJCUT1_SS   2.9735f
#define REBO_MOS2_LJCUT1_MS   3.48175f

// 2.5 * sigma
#define REBO_MOS2_LJCUT2_MM   10.5f
#define REBO_MOS2_LJCUT2_SS   7.825f
#define REBO_MOS2_LJCUT2_MS   9.1625f

// 4 * epsilon * sigma^12
#define REBO_MOS2_s12e4_MM    70617.45058278613f
#define REBO_MOS2_s12e4_SS    70732.99337720051f
#define REBO_MOS2_s12e4_MS    80425.36048432751f

// 4 * epsilon * sigma^6
#define REBO_MOS2_s6e4_MM     12.865192601587204f
#define REBO_MOS2_s6e4_SS     75.223928840336711f
#define REBO_MOS2_s6e4_MS     33.185559361443481f

// 48 * epsilon * sigma^12
#define REBO_MOS2_s12e48_MM   847409.4069934335f
#define REBO_MOS2_s12e48_SS   848795.9205264060f
#define REBO_MOS2_s12e48_MS   965104.3258119302f

// 24 * epsilon * sigma^6
#define REBO_MOS2_s6e24_MM    77.19115560952322f
#define REBO_MOS2_s6e24_SS    451.3435730420202f
#define REBO_MOS2_s6e24_MS    199.1133561686609f

// pre-computed coefficient of the (r - r1)^2 term
#define REBO_MOS2_D2_MM      0.031194467724753f
#define REBO_MOS2_D2_SS      0.820449609102021f
#define REBO_MOS2_D2_MS      0.113097798217445f

// pre-computed coefficient of the (r - r1)^3 term
#define REBO_MOS2_D3_MM     -0.053895558827613f
#define REBO_MOS2_D3_SS     -1.089810409215252f
#define REBO_MOS2_D3_MS     -0.137425146625715f

#endif




REBO_MOS::REBO_MOS(Parameters *para)
{
    int num = ((para->neighbor.MN<20) ? para->neighbor.MN : 20);
    int memory1 = sizeof(real) * para->N;
    int memory2 = sizeof(real) * para->N * num;
    int memory3 = sizeof(int) * para->N;
    int memory4 = sizeof(int) * para->N * num;
    CHECK(cudaMalloc((void**)&rebo_mos_data.p,    memory1));
    CHECK(cudaMalloc((void**)&rebo_mos_data.pp,   memory1));
    CHECK(cudaMalloc((void**)&rebo_mos_data.b,    memory2));
    CHECK(cudaMalloc((void**)&rebo_mos_data.bp,   memory2));
    CHECK(cudaMalloc((void**)&rebo_mos_data.f12x, memory2));
    CHECK(cudaMalloc((void**)&rebo_mos_data.f12y, memory2));
    CHECK(cudaMalloc((void**)&rebo_mos_data.f12z, memory2));
    CHECK(cudaMalloc((void**)&rebo_mos_data.NN_short, memory3));
    CHECK(cudaMalloc((void**)&rebo_mos_data.NL_short, memory4));

    printf("INPUT: use the potential in [PRB 79, 245110 (2009)].\n");
    rc = 10.5;
}




REBO_MOS::~REBO_MOS(void)
{
    cudaFree(rebo_mos_data.p);
    cudaFree(rebo_mos_data.pp);
    cudaFree(rebo_mos_data.b);
    cudaFree(rebo_mos_data.bp);
    cudaFree(rebo_mos_data.f12x);
    cudaFree(rebo_mos_data.f12y);
    cudaFree(rebo_mos_data.f12z);
    cudaFree(rebo_mos_data.NN_short);
    cudaFree(rebo_mos_data.NL_short);
}




// The repulsive function and its derivative
static __device__ void find_fr_and_frp
(int type12, real d12, real &fr, real &frp)
{     
    if (type12 == 0)
    {   
        fr  = (ONE + REBO_MOS2_Q_MM / d12) * REBO_MOS2_A_MM 
            * exp(-REBO_MOS2_a_MM * d12);  
        frp = REBO_MOS2_a_MM + REBO_MOS2_Q_MM / (d12 * (d12 + REBO_MOS2_Q_MM));
        frp *= -fr;
    }
    else if (type12 == 2)
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
(int type12, real d12, real &fa, real &fap)
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
(int type12, real d12, real &fa)
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
(int type12, real d12, real &fc, real &fcp)
{
    if (type12 == 0)
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
    else if (type12 == 2)
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
static __device__ void find_fc(int type12, real d12, real &fc)
{
    if (type12 == 0)
    {
        if (d12 < REBO_MOS2_r1_MM) {fc = ONE;}
        else if (d12 < REBO_MOS2_r2_MM)
        {
            fc  = cos(REBO_MOS2_pi_factor_MM * (d12 - REBO_MOS2_r1_MM)) 
                * HALF + HALF;
        }
        else {fc  = ZERO;}
    }
    else if (type12 == 2)
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




// get U_ij and (d U_ij / d r_ij) / r_ij for the 2-body part
static __device__ void find_p2_and_f2(int type12, real d12, real &p2, real &f2)
{
    if (type12 == 0) // Mo-Mo
    {
        if      (d12 >= REBO_MOS2_LJCUT2_MM) { p2 = ZERO; f2 = ZERO; }
        else if (d12 >  REBO_MOS2_LJCUT1_MM)
        {
            real d12inv2 = ONE / (d12 * d12);
            real d12inv6 = d12inv2 * d12inv2 * d12inv2;
            p2  = REBO_MOS2_s12e4_MM * d12inv6 * d12inv6;
            p2 -= REBO_MOS2_s6e4_MM * d12inv6;
            f2  = REBO_MOS2_s6e24_MM * d12inv6;
            f2 -= REBO_MOS2_s12e48_MM * d12inv6 * d12inv6;
            f2 *= d12inv2;
        }
        else if (d12 > REBO_MOS2_r1_MM)
        {
            real dr = d12 - REBO_MOS2_r1_MM;
            p2 = (REBO_MOS2_D2_MM + REBO_MOS2_D3_MM * dr) * dr * dr;
            f2 = (REBO_MOS2_D2_MM * TWO + REBO_MOS2_D3_MM * THREE * dr) * dr;
            f2 /= d12;
        }
        else { p2 = ZERO; f2 = ZERO; }
    }
    else if (type12 == 1) // Mo-S
    {
        if      (d12 >= REBO_MOS2_LJCUT2_MS) { p2 = ZERO; f2 = ZERO; }
        else if (d12 >  REBO_MOS2_LJCUT1_MS)
        {
            real d12inv2 = ONE / (d12 * d12);
            real d12inv6 = d12inv2 * d12inv2 * d12inv2;
            p2  = REBO_MOS2_s12e4_MS * d12inv6 * d12inv6;
            p2 -= REBO_MOS2_s6e4_MS * d12inv6;
            f2  = REBO_MOS2_s6e24_MS * d12inv6;
            f2 -= REBO_MOS2_s12e48_MS * d12inv6 * d12inv6;
            f2 *= d12inv2;
        }
        else if (d12 > REBO_MOS2_r1_MS)
        {
            real dr = d12 - REBO_MOS2_r1_MS;
            p2 = (REBO_MOS2_D2_MS + REBO_MOS2_D3_MS * dr) * dr * dr;
            f2 = (REBO_MOS2_D2_MS * TWO + REBO_MOS2_D3_MS * THREE * dr) * dr;
            f2 /= d12;
        }
        else { p2 = ZERO; f2 = ZERO; }
    }
    else // S-S
    {
        if      (d12 >= REBO_MOS2_LJCUT2_SS) { p2 = ZERO; f2 = ZERO; }
        else if (d12 >  REBO_MOS2_LJCUT1_SS)
        {
            real d12inv2 = ONE / (d12 * d12);
            real d12inv6 = d12inv2 * d12inv2 * d12inv2;
            p2  = REBO_MOS2_s12e4_SS * d12inv6 * d12inv6;
            p2 -= REBO_MOS2_s6e4_SS * d12inv6;
            f2  = REBO_MOS2_s6e24_SS * d12inv6;
            f2 -= REBO_MOS2_s12e48_SS  * d12inv6 * d12inv6;
            f2 *= d12inv2;
        }
        else if (d12 > REBO_MOS2_r1_SS)
        {
            real dr = d12 - REBO_MOS2_r1_SS;
            p2 = (REBO_MOS2_D2_SS + REBO_MOS2_D3_SS * dr) * dr * dr;
            f2 = (REBO_MOS2_D2_SS * TWO + REBO_MOS2_D3_SS * THREE * dr) * dr;
            f2 /= d12;
        }
        else { p2 = ZERO; f2 = ZERO; }
    }
}




// 2-body part (kernel)
template <int cal_j, int cal_q, int cal_k>
static __global__ void find_force_step0
(
    real fe_x, real fe_y, real fe_z,
    int number_of_particles, int N1, int N2, int pbc_x, int pbc_y, int pbc_z,
    int *g_NN, int *g_NL, int *g_NN_local, int *g_NL_local, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
    const real* __restrict__ g_vx, 
    const real* __restrict__ g_vy, 
    const real* __restrict__ g_vz,
#else
    real *g_x,  real *g_y,  real *g_z, real *g_vx, real *g_vy, real *g_vz,
#endif
    real *g_box, real *g_p,  real *g_pp,
    real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz, real *g_potential, 
    real *g_h, int *g_label, int *g_fv_index, real *g_fv,
    int *g_a_map, int *g_b_map, int *g_count_b
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
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
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real vx1 = LDG(g_vx, n1); 
        real vy1 = LDG(g_vy, n1); 
        real vz1 = LDG(g_vz, n1);
        real lx = g_box[0]; 
        real ly = g_box[1]; 
        real lz = g_box[2];
        
        int count = 0; // initialize g_NN_local[n1] to 0
        real coordination_number = ZERO;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_NL[n1 + number_of_particles * i1];
            
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            int type2 = g_type[n2];
            int type12 = type1 + type2; // 0 = AA; 1 = AB or BA; 2 = BB

            if (d12 < REBO_MOS2_r2_MM)
            {        
                // build the 3-body neighbor list            
                g_NL_local[n1 + number_of_particles * (count++)] = n2;
                // accumulate the coordination number
                real fc12; 
                find_fc(type12, d12, fc12);
                coordination_number += fc12;
            }

            real p2 = ZERO, f2 = ZERO;
            find_p2_and_f2(type12, d12, p2, f2);

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
            s_sx -= x12 * (f12x - f21x) * HALF; 
            s_sy -= y12 * (f12y - f21y) * HALF; 
            s_sz -= z12 * (f12z - f21z) * HALF;

            if (cal_j || cal_k)
            {
                s_h1 += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s_h2 += (f21z * vz1) * x12;               // x-out
                s_h3 += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s_h4 += (f21z * vz1) * y12;               // y-out
                s_h5 += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            }

            // accumulate heat across some sections (for NEMD)
            //    	check if AB pair possible & exists
            if (cal_q && g_a_map[n1] != -1 && g_b_map[n2] != -1 &&
                g_fv_index[g_a_map[n1] * *(g_count_b) + g_b_map[n2]] != -1)
            {
                int index_12 = 
                    g_fv_index[g_a_map[n1] * *(g_count_b) + g_b_map[n2]] * 12;
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
        // save the P(N) function and its derivative
        real p, pp;
        find_p_and_pp(type1, coordination_number, p, pp);
        g_p[n1] = p;    // will be used in find_force_step1 
        g_pp[n1] = pp;  // will be used in find_force_step2 

        // driving force
        if (cal_k)
        {
            s_fx += fx_driving; // with driving force
            s_fy += fy_driving; // with driving force
            s_fz += fz_driving; // with driving force
        }

        g_fx[n1] += s_fx; // save force
        g_fy[n1] += s_fy;
        g_fz[n1] += s_fz;
        // save stress and potential
        g_sx[n1] += s_sx;
        g_sy[n1] += s_sy;
        g_sz[n1] += s_sz;
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




// Precompute the bond-order function and its derivative 
static __global__ void find_force_step1
(
    int N, int N1, int N2, int pbc_x, int pbc_y, int pbc_z,
    int* g_NN, int* g_NL, int* g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z,
#else
    real* g_x, real* g_y, real* g_z,
#endif
    real *g_box,
    real* g_b, real* g_bp, real *g_p
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real lx = LDG(g_box, 0); 
        real ly = LDG(g_box, 1); 
        real lz = LDG(g_box, 2);
        real p = g_p[n1]; // coordination number function P(N)

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_NL[n1 + N * i1];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;

            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

            real zeta = ZERO;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_NL[n1 + N * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;

                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                real cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                real fc13, g123; 
                int type13 = type1 + type3;
                find_fc(type13, d13, fc13);
                find_g(type1, cos123, g123);
                zeta += fc13 * g123;
            }

            zeta += p;
            real b12 = pow(ONE + zeta, -HALF);
            g_b[i1 * N + n1]  = b12;
            g_bp[i1 * N + n1] = (-HALF)*b12/(ONE+zeta); 
        }
    }
}




// calculate and save the partial forces dU_i/dr_ij
static __global__ void find_force_step2
(
    int N, int N1, int N2, int pbc_x, int pbc_y, int pbc_z,
    int *g_NN, int *g_NL, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_b, 
    const real* __restrict__ g_bp,
    const real* __restrict__ g_pp,
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z,
#else
    real* g_b, real* g_bp, real* g_pp, real* g_x, real* g_y, real* g_z,
#endif
    real *g_box, real *g_potential, real *g_f12x, real *g_f12y, real *g_f12z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real pp1 = LDG(g_pp, n1); 
        real lx = LDG(g_box, 0); 
        real ly = LDG(g_box, 1); 
        real lz = LDG(g_box, 2);
        real potential_energy = ZERO;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * N + n1;
            int n2 = g_NL[index];
            int type2 = g_type[n2];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real d12inv = ONE / d12;

            real fc12, fcp12, fa12, fap12, fr12, frp12;
            int type12 = type1 + type2;
            find_fc_and_fcp(type12, d12, fc12, fcp12);
            find_fa_and_fap(type12, d12, fa12, fap12);
            find_fr_and_frp(type12, d12, fr12, frp12);

            // accumulate_force_12 
            real b12 = LDG(g_b, index);
            real bp12 = LDG(g_bp, index);
            real factor3 = (fcp12*(fr12-b12*fa12) + fc12*(frp12-b12*fap12) 
                         - fc12*fcp12*fa12*bp12*pp1)/d12;
            real f12x = x12 * factor3 * HALF;
            real f12y = y12 * factor3 * HALF;
            real f12z = z12 * factor3 * HALF;

            // accumulate potential energy
            potential_energy += fc12 * (fr12 - b12 * fa12) * HALF;

            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_NL[n1 + N * i2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                real fc13, fa13;
                int type13 = type1 + type3;
                find_fc(type13, d13, fc13);
                find_fa(type13, d13, fa13);
                real bp13 = LDG(g_bp, i2 * N + n1);
                real one_over_d12d13 = ONE / (d12 * d13);
                real cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv;
                real g123, gp123;
                find_g_and_gp(type1, cos123, g123, gp123);

                real temp123a=(-bp12*fc12*fa12*fc13-bp13*fc13*fa13*fc12)*gp123;
                real temp123b= - bp13 * fc13 * fa13 * fcp12 * (g123+pp1) / d12;
                real cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * temp123b + temp123a * cos_d)*HALF;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * temp123b + temp123a * cos_d)*HALF;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * temp123b + temp123a * cos_d)*HALF;
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
void REBO_MOS::compute(Parameters *para, GPU_Data *gpu_data, Measure *measure)
{
    int N = para->N;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;

    int *NN = gpu_data->NN_local;           // for 2-body
    int *NL = gpu_data->NL_local;           // for 2-body
    int *NN_local = rebo_mos_data.NN_short; // for 3-body
    int *NL_local = rebo_mos_data.NL_short; // for 3-body

    int *type = gpu_data->type_local;
    real *x = gpu_data->x;
    real *y = gpu_data->y;
    real *z = gpu_data->z;
    real *vx = gpu_data->vx;
    real *vy = gpu_data->vy;
    real *vz = gpu_data->vz;
    real *fx = gpu_data->fx;
    real *fy = gpu_data->fy;
    real *fz = gpu_data->fz;
    real *box = gpu_data->box_length;
    real *sx = gpu_data->virial_per_atom_x;
    real *sy = gpu_data->virial_per_atom_y;
    real *sz = gpu_data->virial_per_atom_z;
    real *pe = gpu_data->potential_per_atom;
    real *h = gpu_data->heat_per_atom;

    int *label = gpu_data->label;
    int *fv_index = gpu_data->fv_index;
    int *a_map = gpu_data->a_map;
    int *b_map = gpu_data->b_map;
    int *count_b = gpu_data->count_b;
    real *fv = gpu_data->fv;

    real fe_x = measure->hnemd.fe_x;
    real fe_y = measure->hnemd.fe_y;
    real fe_z = measure->hnemd.fe_z;

    real *b    = rebo_mos_data.b;
    real *bp   = rebo_mos_data.bp;
    real *p    = rebo_mos_data.p;
    real *pp   = rebo_mos_data.pp;
    real *f12x = rebo_mos_data.f12x;
    real *f12y = rebo_mos_data.f12y;
    real *f12z = rebo_mos_data.f12z;

    // 2-body part
    if (measure->hac.compute)
    {
        find_force_step0<1, 0, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            fe_x, fe_y, fe_z,
            N, N1, N2, pbc_x, pbc_y, pbc_z, NN, NL, NN_local, NL_local, type,
            x, y, z, vx, vy, vz, box, p, pp, fx, fy, fz,
            sx, sy, sz, pe, h, label, fv_index, fv, a_map, b_map, count_b
        );
    }
    else if (measure->hnemd.compute && !measure->shc.compute)
    {
        find_force_step0<0, 0, 1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            fe_x, fe_y, fe_z,
            N, N1, N2, pbc_x, pbc_y, pbc_z, NN, NL, NN_local, NL_local, type,
            x, y, z, vx, vy, vz, box, p, pp, fx, fy, fz,
            sx, sy, sz, pe, h, label, fv_index, fv, a_map, b_map, count_b
        );
    }
    else if (measure->shc.compute && !measure->hnemd.compute)
    {
        find_force_step0<0, 1, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            fe_x, fe_y, fe_z,
            N, N1, N2, pbc_x, pbc_y, pbc_z, NN, NL, NN_local, NL_local, type,
            x, y, z, vx, vy, vz, box, p, pp, fx, fy, fz,
            sx, sy, sz, pe, h, label, fv_index, fv, a_map, b_map, count_b
        );
    }
    else if (measure->shc.compute && measure->hnemd.compute)
    {
        find_force_step0<0, 1, 1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            fe_x, fe_y, fe_z,
            N, N1, N2, pbc_x, pbc_y, pbc_z, NN, NL, NN_local, NL_local, type,
            x, y, z, vx, vy, vz, box, p, pp, fx, fy, fz,
            sx, sy, sz, pe, h, label, fv_index, fv, a_map, b_map, count_b
        );
    }
    else
    {
        find_force_step0<0, 0, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            fe_x, fe_y, fe_z,
            N, N1, N2, pbc_x, pbc_y, pbc_z, NN, NL, NN_local, NL_local, type,
            x, y, z, vx, vy, vz, box, p, pp, fx, fy, fz,
            sx, sy, sz, pe, h, label, fv_index, fv, a_map, b_map, count_b
        );
    }

    // pre-compute the bond-order function and its derivative
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, pbc_x, pbc_y, pbc_z, NN_local, NL_local, type, 
        x, y, z, box, b, bp, p
    );

    // pre-compute the partial force
    find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, pbc_x, pbc_y, pbc_z, NN_local, NL_local, type, 
        b, bp, pp, x, y, z, box, pe, f12x, f12y, f12z
    );

    // 3-body part
    find_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        measure->hac.compute, measure->shc.compute, measure->hnemd.compute,
        fe_x, fe_y, fe_z, N, N1, N2, pbc_x, pbc_y, pbc_z, 
        NN_local, NL_local, f12x, f12y, f12z,
        x, y, z, vx, vy, vz, box, fx, fy, fz,
        sx, sy, sz, h, label, fv_index, fv, a_map, b_map, count_b
    );
}




