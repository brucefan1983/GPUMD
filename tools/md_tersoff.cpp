/*
    This is a simple molecular dynamics (MD) code for calculating 
    thermal conductivity using the Tersoff potential. 

    Author: Zheyong Fan (brucenju@gmail.com)

    Note: 
    1) This is a serial code, which is used for teaching;
    2) The Tersoff potential parameters by Lindsay&Broido are hard coded; 
    3) The neighbor list is only built in the beginning (OK for stable solids);
    4) The box is assumed to be rectangular and is fixed (no pressure control);
    5) The temperature control is achieved by velocity re-scaling;
    6) The simulated system and various parameters are hard coded;
    7) The convective term of the heat current is dropped (OK for solids);
    8) The formulas in [PRB 92, 094301 (2015)] are used;
    9) My natural unit system: length--Angstrom; mass--amu; energy--eV;
    10) compile with "g++ -O3 md_tersoff.cpp" and run with "./a.out"
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define K_B                      8.617343e-5 // Boltzmann's constant  
#define TIME_UNIT_CONVERSION     1.018051e+1 // fs     <-> my natural unit
#define KAPPA_UNIT_CONVERSION    1.573769e+5 // W/(mK) <-> my natural unit
#define PRESSURE_UNIT_CONVERSION 1.602177e+2 // eV/A^3 <-> my natural unit

// apply the minimum image convention
void apply_mic
(
    int pbc[3], double box[3], double lxh, double lyh, double lzh, 
    double &x12, double &y12, double &z12
)
{
    if (pbc[0] == 1)
    {
        if (x12 < - lxh) {x12 += box[0];} else if (x12 > lxh) {x12 -= box[0];}
    }
    if (pbc[1] == 1)
    {
        if (y12 < - lyh) {y12 += box[1];} else if (y12 > lyh) {y12 -= box[1];}
    }
    if (pbc[2] == 1)
    {
        if (z12 < - lzh) {z12 += box[2];} else if (z12 > lzh) {z12 -= box[2];}
    }
}

// contruct the neighbor list
void find_neighbor
(
    int N, int *NN, int *NL, int pbc[3], double box[3],
    double *x, double *y, double *z, int MN, double cutoff
)              
{
    double lxh = box[0] * 0.5;
    double lyh = box[1] * 0.5;
    double lzh = box[2] * 0.5; 
    double cutoff_square = cutoff * cutoff;
    for (int n = 0; n < N; n++) {NN[n] = 0;}
    for (int n1 = 0; n1 < N - 1; n1++)
    {  
        for (int n2 = n1 + 1; n2 < N; n2++)
        {   
            double x12 = x[n2] - x[n1];
            double y12 = y[n2] - y[n1];
            double z12 = z[n2] - z[n1];
            apply_mic(pbc, box, lxh, lyh, lzh, x12, y12, z12);
            double  distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < cutoff_square)
            {        
                NL[n1 * MN + NN[n1]] = n2;
                NN[n1]++;
                NL[n2 * MN + NN[n2]] = n1;
                NN[n2]++;
            }
            if (NN[n1] > MN)
            {
                printf("Error: MN is too small.\n");
                exit(1);
            }
        }
    } 
}

// initialize the positions: I take graphene as an example here 
void initialize_position 
(
    int nx, int ny, int nz, double ax, double ay, double az, 
    double *x, double *y, double *z
)
{
    int n0 = 4; // rectangular unit cell
    double x0[4] = {0.0,   0.0, 0.5, 0.5  }; 
    double y0[4] = {1.0/6, 0.5, 0.0, 2.0/3}; 
    double z0[4] = {0.0,   0.0, 0.0, 0.0  };
    int n = 0;
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                for (int i = 0; i < n0; ++i)
                {
                    x[n] = (ix + x0[i]) * ax;
                    y[n] = (iy + y0[i]) * ay;
                    z[n] = (iz + z0[i]) * az;
                    n++;
                }
            }
        }
    }
} 

// scale the velocities to reach the target temperature
void scale_velocity
(int N, double T_0, double *m, double *vx, double *vy, double *vz)
{  
    double temperature = 0.0;
    for (int n = 0; n < N; ++n) 
    {
        double v2 = vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n];     
        temperature += m[n] * v2; 
    }
    temperature /= 3.0 * K_B * N;
    double scale_factor = sqrt(T_0 / temperature);
    for (int n = 0; n < N; ++n)
    { 
        vx[n] *= scale_factor;
        vy[n] *= scale_factor;
        vz[n] *= scale_factor;
    }
}  

// initialize the velocites (only the linear momentum is zeroed)  
void initialize_velocity
(int N, double T_0, double *m, double *vx, double *vy, double *vz)
{
    double momentum_average[3] = {0.0, 0.0, 0.0};
    for (int n = 0; n < N; ++n)
    { 
        vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX; 
        vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;    
        
        momentum_average[0] += m[n] * vx[n] / N;
        momentum_average[1] += m[n] * vy[n] / N;
        momentum_average[2] += m[n] * vz[n] / N;
    } 
    for (int n = 0; n < N; ++n) 
    { 
        vx[n] -= momentum_average[0] / m[n];
        vy[n] -= momentum_average[1] / m[n];
        vz[n] -= momentum_average[2] / m[n]; 
    }
    scale_velocity(N, T_0, m, vx, vy, vz);
}

// The repulsive function and its derivative in the Tersoff potential
inline void find_fr_and_frp(double d12, double &fr, double &frp)
{     
    const double a = 1393.6;   
    const double lambda = 3.4879;    
    fr  = a * exp(- lambda * d12);    
    frp = - lambda * fr;
}

// The attractive function and its derivative in the Tersoff potential
inline void find_fa_and_fap(double d12, double &fa, double &fap)
{     
    const double b = 430.0; // optimized
    const double mu = 2.2119;   
    fa  = b * exp(- mu * d12);    
    fap = - mu * fa;
}

// The attractive function in the Tersoff potential
inline void find_fa(double d12, double &fa)
{     
    const double b = 430.0;  
    const double mu = 2.2119;   
    fa  = b * exp(- mu * d12);    
}

// The cutoff function and its derivative in the Tersoff potential
inline void find_fc_and_fcp(double d12, double &fc, double &fcp)
{
    const double r1 = 1.8;
    const double r2 = 2.1;
    const double pi = 3.141592653589793;
    const double pi_factor = pi / (r2 - r1);
    if (d12 < r1)
    {
        fc  = 1.0;
        fcp = 0.0;
    }
    else if (d12 < r2)
    {              
        fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
        fcp = - sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
    }
    else
    {
        fc = 0.0;
        fcp = 0.0;
    }
}

// The cutoff function in the Tersoff potential
inline void find_fc(double d12, double &fc)
{
    const double r1 = 1.8;
    const double r2 = 2.1;
    const double pi = 3.141592653589793;
    const double pi_factor = pi / (r2 - r1);
    if (d12 < r1)
    {
        fc  = 1.0;
    }
    else if (d12 < r2)
    {             
        fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
    }
    else 
    {
        fc = 0.0;
    }
}

// The angular function and its derivative in the Tersoff potential
inline void find_g_and_gp(double cos, double &g, double &gp)
{
    const double c = 38049.0;
    const double d = 4.3484;
    const double h = - 0.930; // optimized
    const double c2 = c * c;
    const double d2 = d * d;
    const double c2overd2 = c2 / d2;  
    double temp = d2 + (cos - h) * (cos - h);
    g  = 1.0 + c2overd2 - c2 / temp;    
    gp = 2.0 * c2 * (cos - h) / (temp * temp);    
}

// The angular function in the Tersoff potential
inline void find_g(double cos, double &g)
{
    const double c = 38049.0;
    const double d = 4.3484;
    const double h = - 0.930;  // optimized
    const double c2 = c * c;
    const double d2 = d * d;
    const double c2overd2 = c2 / d2;  
    double temp = d2 + (cos - h) * (cos - h);
    g  = 1.0 + c2overd2 - c2 / temp;      
}

// pre-compute the bond-order functions and their derivatives
void find_b_and_bp
(
    int N, int *NN, int*NL, int MN, int pbc[3], double box[3],
    double *x, double *y, double *z, double *b, double *bp
)
{
    const double beta = 1.5724e-7;
    const double n = 0.72751;     
    const double minus_half_over_n = - 0.5 / n;

    double lxh = box[0] * 0.5;
    double lyh = box[1] * 0.5;
    double lzh = box[2] * 0.5;
    for (int n1 = 0; n1 < N; ++n1)
    {
        for (int i1 = 0; i1 < NN[n1]; ++i1)   
        {       
            int n2 = NL[n1 * MN + i1]; // we only know n2 != n1     
            double x12, y12, z12;
            x12 = x[n2] - x[n1];
            y12 = y[n2] - y[n1];
            z12 = z[n2] - z[n1]; 
            apply_mic(pbc, box, lxh, lyh, lzh, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            
            double zeta = 0.0;
            for (int i2 = 0; i2 < NN[n1]; ++i2)
            {
                int n3 = NL[n1 * MN + i2];  // we only know n3 != n1  
                if (n3 == n2) { continue; } // ensure that n3 != n2
                double x13, y13, z13;
                x13 = x[n3] - x[n1];
                y13 = y[n3] - y[n1];
                z13 = z[n3] - z[n1];
                apply_mic(pbc, box, lxh, lyh, lzh, x13, y13, z13);

                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double cos = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                double fc13, g123; 
                find_fc(d13, fc13);
                find_g(cos, g123);
                zeta += fc13 * g123;
            } 
            double bzn = pow(beta * zeta, n);
            double b12 = pow(1.0 + bzn, minus_half_over_n);
            b[n1 * MN + i1]  = b12;
            bp[n1 * MN + i1] = - b12 * bzn * 0.5 / ((1.0 + bzn) * zeta);
        }
    }
}

// The force evaluation function for the Tersoff potential
void find_force_tersoff
(
    int N, int *NN, int*NL, int MN, int pbc[3], double box[3], 
    double *b, double *bp, double *x, double *y, double *z, double *vx, 
    double *vy, double *vz, double *fx, double *fy, double *fz, double prop[7]
)
{
    for (int n = 0; n < 7; ++n) { prop[n]=0.0; }
    for (int n = 0; n < N; ++n) { fx[n]=fy[n]=fz[n]=0.0; }
    double lxh = box[0] * 0.5;
    double lyh = box[1] * 0.5;
    double lzh = box[2] * 0.5;

    for (int n1 = 0; n1 < N; ++n1)
    {
        for (int i1 = 0; i1 < NN[n1]; ++i1)   
        {       
            int n2 = NL[n1 * MN + i1];
            if (n2 < n1) { continue; } // Will use Newton's 3rd law!!!
            double x12, y12, z12;
            x12 = x[n2] - x[n1];
            y12 = y[n2] - y[n1];
            z12 = z[n2] - z[n1];
            apply_mic(pbc, box, lxh, lyh, lzh, x12, y12, z12);
          
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = 1.0 / d12;
            double d12inv_square = d12inv * d12inv;

            double fc12, fcp12;
            double fa12, fap12;
            double fr12, frp12;
            find_fc_and_fcp(d12, fc12, fcp12);
            find_fa_and_fap(d12, fa12, fap12);
            find_fr_and_frp(d12, fr12, frp12);

            double b12, bp12;
  
            double f12[3] = {0.0, 0.0, 0.0};   // d_U_i_d_r_ij
            double f21[3] = {0.0, 0.0, 0.0};   // d_U_j_d_r_ji 
            double p12 = 0.0;                  // U_ij
            double p21 = 0.0;                  // U_ji
           
            // accumulate_force_12 
            b12 = b[n1 * MN + i1]; 
            double factor1 = - b12 * fa12 + fr12;
            double factor2 = - b12 * fap12 + frp12;    
            double factor3 = (fcp12 * factor1 + fc12 * factor2) / d12;   
            f12[0] += x12 * factor3 * 0.5; 
            f12[1] += y12 * factor3 * 0.5;
            f12[2] += z12 * factor3 * 0.5;     
            p12 += factor1 * fc12;

            // accumulate_force_21
            int offset = 0;
            for (int k = 0; k < NN[n2]; ++k)
            {
                if (NL[n2 * MN + k] == n1) 
                { 
                    offset = k;
                    break; 
                }
            }
            b12 = b[n2 * MN + offset]; 
            factor1 = - b12 * fa12 + fr12;
            factor2 = - b12 * fap12 + frp12;    
            factor3 = (fcp12 * factor1 + fc12 * factor2) / d12;                   
            f21[0] += -x12 * factor3 * 0.5; 
            f21[1] += -y12 * factor3 * 0.5;
            f21[2] += -z12 * factor3 * 0.5;           
            p21 += factor1 * fc12;

            // accumulate_force_123
            bp12 = bp[n1 * MN + i1]; 
            for (int i2 = 0; i2 < NN[n1]; ++i2)
            {    
                int n3 = NL[n1 * MN + i2];     
                if (n3 == n2) { continue; } 
                double x13, y13, z13;
                x13 = x[n3] - x[n1];
                y13 = y[n3] - y[n1];
                z13 = z[n3] - z[n1];
                apply_mic(pbc, box, lxh, lyh, lzh, x13, y13, z13);

                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);         
                double fc13, fa13;
                find_fc(d13, fc13);
                find_fa(d13, fa13); 
                double bp13 = bp[n1 * MN + i2]; 

                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                double g123, gp123;
                find_g_and_gp(cos123, g123, gp123);
                double cos_x = x13 / (d12 * d13) - x12 * cos123 / (d12 * d12);
                double cos_y = y13 / (d12 * d13) - y12 * cos123 / (d12 * d12);
                double cos_z = z13 / (d12 * d13) - z12 * cos123 / (d12 * d12);                        
                double factor123a = (-bp12*fc12*fa12*fc13 - bp13*fc13*fa13*fc12)*gp123;
                double factor123b = - bp13 * fc13 * fa13 * fcp12 * g123 * d12inv;
                f12[0] += (x12 * factor123b + factor123a * cos_x) * 0.5; 
                f12[1] += (y12 * factor123b + factor123a * cos_y) * 0.5;
                f12[2] += (z12 * factor123b + factor123a * cos_z) * 0.5;
            }

            // accumulate_force_213
            bp12 = bp[n2 * MN + offset]; 
            for (int i2 = 0; i2 < NN[n2]; ++i2)
            {
                int n3 = NL[n2 * MN + i2];       
                if (n3 == n1) { continue; } 
                double x23, y23, z23;
                x23 = x[n3] - x[n2];
                y23 = y[n3] - y[n2];
                z23 = z[n3] - z[n2];
                apply_mic(pbc, box, lxh, lyh, lzh, x23, y23, z23);

                double d23 = sqrt(x23 * x23 + y23 * y23 + z23 * z23);         
                double fc23, fa23;
                find_fc(d23, fc23);
                find_fa(d23, fa23);
                double bp13 = bp[n2 * MN + i2]; 

                double cos213 = - (x12 * x23 + y12 * y23 + z12 * z23) / (d12 * d23);
                double g213, gp213;
                find_g_and_gp(cos213, g213, gp213);
                double cos_x = x23 / (d12 * d23) + x12 * cos213 / (d12 * d12);
                double cos_y = y23 / (d12 * d23) + y12 * cos213 / (d12 * d12);
                double cos_z = z23 / (d12 * d23) + z12 * cos213 / (d12 * d12);                       
                double factor213a = (-bp12*fc12*fa12*fc23 - bp13*fc23*fa23*fc12)*gp213;
                double factor213b = - bp13 * fc23 * fa23 * fcp12 * g213 * d12inv;
                f21[0] += (-x12 * factor213b + factor213a * cos_x) * 0.5; 
                f21[1] += (-y12 * factor213b + factor213a * cos_y) * 0.5;
                f21[2] += (-z12 * factor213b + factor213a * cos_z) * 0.5;
            }

            // accumulate force: see Eq. (37) in [PRB 92, 094301 (2015)]   
            double fx12 = f12[0] - f21[0];
            double fy12 = f12[1] - f21[1];
            double fz12 = f12[2] - f21[2];
            fx[n1] += fx12; 
            fy[n1] += fy12; 
            fz[n1] += fz12; 
            fx[n2] -= fx12; // Newton's 3rd law used here
            fy[n2] -= fy12; 
            fz[n2] -= fz12;

            // accumulate potential energy:           
            prop[0] += (p12 + p21) * 0.5;    

            // accumulate virial; see Eq. (39) in [PRB 92, 094301 (2015)]
            prop[1] -= fx12*x12;
            prop[2] -= fy12*y12;
            prop[3] -= fz12*z12;

            // accumulate heat current; see Eq. (43) in [PRB 92, 094301 (2015)]
            double f12_dot_v2 = f12[0]*vx[n2] + f12[1]*vy[n2] + f12[2]*vz[n2];   
            double f21_dot_v1 = f21[0]*vx[n1] + f21[1]*vy[n1] + f21[2]*vz[n1];      
            prop[4] -= (f12_dot_v2 - f21_dot_v1) * x12;  
            prop[5] -= (f12_dot_v2 - f21_dot_v1) * y12;                       
            prop[6] -= (f12_dot_v2 - f21_dot_v1) * z12;
        }
    } 
} 

// a wrapper
void find_force
(
    int N, int *NN, int*NL, int MN, int pbc[3], double box[3], 
    double *b, double *bp, double *x, double *y, double *z, double *vx, 
    double *vy, double *vz, double *fx, double *fy, double *fz, double prop[7]
)
{
    find_b_and_bp(N, NN, NL, MN, pbc, box, x, y, z, b, bp);
    find_force_tersoff
    (N, NN, NL, MN, pbc, box, b, bp, x, y, z, vx, vy, vz, fx, fy, fz, prop);
} 

// velocity-Verlet
void integrate
(
    int N, double time_step, double *m, double *fx, double *fy, double *fz, 
    double *vx, double *vy, double *vz, double *x, double *y, double *z, 
    int flag
)
{
    double time_step_half = time_step * 0.5;
    for (int n = 0; n < N; ++n)
    {
        double mass_inv = 1.0 / m[n];
        double ax = fx[n] * mass_inv;
        double ay = fy[n] * mass_inv;
        double az = fz[n] * mass_inv;
        vx[n] += ax * time_step_half;
        vy[n] += ay * time_step_half;
        vz[n] += az * time_step_half;
        if (flag == 1) 
        { 
            x[n] += vx[n] * time_step; 
            y[n] += vy[n] * time_step; 
            z[n] += vz[n] * time_step; 
        }
    }
}

// find heat current autocorrelation (hac)
void find_hac
(
    int Nc, int M, double *hx, double *hy, double *hz, double *hac_x, 
    double *hac_y, double *hac_z 
)
{
    for (int nc = 0; nc < Nc; nc++) // loop over the correlation time points
    {
        for (int m = 0; m < M; m++) // loop over the time origins
        {
            hac_x[nc] += hx[m] * hx[m + nc]; 
            hac_y[nc] += hy[m] * hy[m + nc]; 
            hac_z[nc] += hz[m] * hz[m + nc]; 
        }
        hac_x[nc] /= M; hac_y[nc] /= M; hac_z[nc] /= M;
    }
}

// find running thermal conductivity (rtc)
static void find_rtc
(
    int Nc, double factor, double *hac_x, double *hac_y, double *hac_z,
    double *rtc_x, double *rtc_y, double *rtc_z
)
{
    for (int nc = 1; nc < Nc; nc++)  
    {
        rtc_x[nc] = rtc_x[nc - 1] + (hac_x[nc - 1] + hac_x[nc]) * factor;
        rtc_y[nc] = rtc_y[nc - 1] + (hac_y[nc - 1] + hac_y[nc]) * factor;
        rtc_z[nc] = rtc_z[nc - 1] + (hac_z[nc - 1] + hac_z[nc]) * factor;
    }
}

// find hac and rtc
void find_hac_kappa
(
    int Nd, int Nc, double dt, double T_0, double V, 
    double *hx, double *hy, double *hz
)
{
    double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps
    int M = Nd - Nc; // number of time origins

    double *hac_x = (double *)malloc(sizeof(double) * Nc);
    double *hac_y = (double *)malloc(sizeof(double) * Nc);
    double *hac_z = (double *)malloc(sizeof(double) * Nc);
    double *rtc_x = (double *)malloc(sizeof(double) * Nc);
    double *rtc_y = (double *)malloc(sizeof(double) * Nc);
    double *rtc_z = (double *)malloc(sizeof(double) * Nc);
    for (int nc = 0; nc < Nc; nc++) {hac_x[nc] = hac_y[nc] = hac_z[nc] = 0.0;}
    for (int nc = 0; nc < Nc; nc++) {rtc_x[nc] = rtc_y[nc] = rtc_z[nc] = 0.0;}

    find_hac(Nc, M, hx, hy, hz, hac_x, hac_y, hac_z);
    double factor = dt * 0.5 *  KAPPA_UNIT_CONVERSION / (K_B * T_0 * T_0 * V);
    find_rtc(Nc, factor, hac_x, hac_y, hac_z, rtc_x, rtc_y, rtc_z);

    FILE *fid = fopen("hac.txt", "a"); // "append" mode 
    for (int nc = 0; nc < Nc; nc++) 
    {
        fprintf
        (
            fid, "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e\n", 
            nc * dt_in_ps, // in units of ps
            hac_x[nc], hac_y[nc], hac_z[nc], // in my natural units 
            rtc_x[nc], rtc_y[nc], rtc_z[nc]  // in units of W/mK
        );
    }

    // do not forget to clear up
    fclose(fid);
    free(hac_x); free(hac_y); free(hac_z);
    free(rtc_x); free(rtc_y); free(rtc_z);
}

// Finally, we reach the main function
int main(int argc, char *argv[])
{
    srand(time(NULL)); // each run is independent
    int nx = 20; // number of unit cells in the x-direction
    int ny = 12; // number of unit cells in the y-direction
    int nz = 1;  // number of unit cells in the z-direction
    int n0 = 4;  // number of particles in the unit cell
    int N = n0 * nx * ny * nz; // total number of particles
    int Ne = 10000;   // number of steps in the equilibration stage
    int Np = 10000;   // number of steps in the production stage
    int Ns = 10;      // sampling interval
    int Nd = Np / Ns; // number of heat current data
    int Nc = Nd / 10; // number of correlation data (a good choice)
    int MN = 3;       // maximum number of neighbors for one particle
    int pbc[3] = {1, 1, 0}; // 1 for periodic boundary; 0 for free boundary

    double T_0 = 300.0;           // temperature prescribed
    double ax = 1.438 * sqrt(3.0); // lattice constant in the x direction
    double ay = 1.438 * 3.0;       // lattice constant in the y direction
    double az = 3.35;             // Just a convention
    double box[3];
    box[0] = ax * nx;             // box length in the x direction
    box[1] = ay * ny;             // box length in the y direction
    box[2] = az * nz;             // box length in the z direction
    double volume = box[0] * box[1] * box[2]; // volume of the system
    double cutoff = 2.1;          // cutoff distance for neighbor list
    double time_step = 1.0 / TIME_UNIT_CONVERSION; // time step (1 fs here)
    
    // neighbor list
    int *NN = (int*) malloc(N * sizeof(int));
    int *NL = (int*) malloc(N * MN * sizeof(int));

    // major data for the particles
    double *m  = (double*) malloc(N * sizeof(double)); // mass
    double *x  = (double*) malloc(N * sizeof(double)); // position
    double *y  = (double*) malloc(N * sizeof(double));
    double *z  = (double*) malloc(N * sizeof(double));
    double *vx = (double*) malloc(N * sizeof(double)); // velocity
    double *vy = (double*) malloc(N * sizeof(double));
    double *vz = (double*) malloc(N * sizeof(double));
    double *fx = (double*) malloc(N * sizeof(double)); // force
    double *fy = (double*) malloc(N * sizeof(double));
    double *fz = (double*) malloc(N * sizeof(double));
    double *hx = (double*) malloc(Nd * sizeof(double)); // heat current
    double *hy = (double*) malloc(Nd * sizeof(double));
    double *hz = (double*) malloc(Nd * sizeof(double));
    double *b  = (double*) malloc(N * MN * sizeof(double)); // bond order
    double *bp = (double*) malloc(N * MN * sizeof(double)); 

    // initialize mass, position, and velocity
    for (int n = 0; n < N; ++n) { m[n] = 12.0; } // mass for carbon atom
    initialize_position(nx, ny, nz, ax, ay, az, x, y, z);
    initialize_velocity(N, T_0, m, vx, vy, vz);

    // initialize neighbor list and force
    find_neighbor(N, NN, NL, pbc, box, x, y, z, MN, cutoff);
    double prop[7]; // potential, virial, and heat current
    find_force
    (N, NN, NL, MN, pbc, box, b, bp, x, y, z, vx, vy, vz, fx, fy, fz, prop);

    // open a file for outputting some thermodynamic properties
    FILE *fid = fopen("thermo.txt", "w");
    clock_t time_begin;
    clock_t time_finish;
    double time_used;

    // equilibration
    printf("\nEquilibration started:\n");
    time_begin = clock();
    for (int step = 0; step < Ne; ++step)
    { 
        integrate(N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 1);
        find_force
        (N, NN, NL, MN, pbc, box, b, bp, x, y, z, vx, vy, vz, fx, fy, fz, prop);
        integrate(N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 2);
        scale_velocity(N, T_0, m, vx, vy, vz); // control temperature
        if ((step+1) % (Ne/10) == 0)
        {
            printf("\t%d steps completed.\n", step + 1);
        }
    } 
    time_finish = clock();
    time_used = (time_finish - time_begin) / (double) CLOCKS_PER_SEC;
    fprintf(stderr, "time used for equilibration = %g s\n", time_used); 

    // production
    printf("\nProduction started:\n");
    time_begin = clock();
    int count = 0;
    for (int step = 0; step < Np; ++step)
    {  
        integrate(N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 1);
        find_force
        (N, NN, NL, MN, pbc, box, b, bp, x, y, z, vx, vy, vz, fx, fy, fz, prop);
        integrate(N, time_step, m, fx, fy, fz, vx, vy, vz, x, y, z, 2);
        if ((step+1) % (Np/10) == 0)
        {
            printf("\t%d steps completed.\n", step + 1);
        }
        if (0 == step % Ns) 
        {
            double pe = prop[0]; // total potential energy
            double px = prop[1]; // pressure in the x direction
            double py = prop[2]; // pressure in the y direction
            double pz = prop[3]; // pressure in the z direction
            double ke = 0.0;     // total kinetic energy
            for (int n = 0; n < N; ++n)
            {
                ke += m[n] * (vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n]);
            }
            ke *= 0.5;
            double temp = 2.0 * ke / (3.0 * N * K_B); // instant temperature
            // Do you remember the state equation for ideal gas: p V = N k_B T?
            px = (px + N * K_B * temp) / volume * PRESSURE_UNIT_CONVERSION; 
            py = (py + N * K_B * temp) / volume * PRESSURE_UNIT_CONVERSION;
            pz = (pz + N * K_B * temp) / volume * PRESSURE_UNIT_CONVERSION;

            fprintf
            (
                fid, "%25.15e%25.15e%25.15e%25.15e%25.15e%25.15e\n", 
                temp,       // in units of K
                ke, pe,     // in units of eV
                px, py, pz  // in units of GPa
            );
            hx[count] = prop[4]; // record the heat current data
            hy[count] = prop[5]; 
            hz[count] = prop[6]; 
            count++; 
        }
    } 

    fclose(fid);
    time_finish = clock();
    time_used = (time_finish - time_begin) / (double) CLOCKS_PER_SEC;
    fprintf(stderr, "time used for production = %g s\n", time_used); 

    // calculate hac and rtc
    find_hac_kappa(Nd, Nc, time_step * Ns, T_0, volume, hx, hy, hz);

    free(NN); free(NL); free(m);  free(x);  free(y);  free(z);
    free(vx); free(vy); free(vz); free(fx); free(fy); free(fz);
    free(hx); free(hy); free(hz); free(b);  free(bp);

    //system("PAUSE"); // for Dev-C++ in Windows
    return 0;
}
