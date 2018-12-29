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
#include "ensemble_bdp.cuh"
#include "ensemble.inc"
#include "force.cuh"
#include "memory.cuh"

#define BLOCK_SIZE 128




// These functions are from  Bussi's website
// https://sites.google.com/site/giovannibussi/Research/algorithms
// See the end of this file for the function definitions
static double resamplekin(double kk, double sigma, int ndeg, double taut);
static double resamplekin_sumnoises(int nn);
static double ran1();
static double gasdev();
static double gamdev(const int ia);




Ensemble_BDP::Ensemble_BDP(int t, real T, real Tc)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
}




Ensemble_BDP::Ensemble_BDP
(int t, int source_input, int sink_input, real T, real Tc, real dT)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
    delta_temperature = dT;
    source = source_input;
    sink = sink_input;
    // initialize the energies transferred from the system to the baths
    energy_transferred[0] = 0.0;
    energy_transferred[1] = 0.0;
}




Ensemble_BDP::~Ensemble_BDP(void)
{
    // nothing now
}




// Scale the velocity of every particle in the system by a factor
static void __global__ gpu_scale_velocity
(int N, real *g_vx, real *g_vy, real *g_vz, real factor)
{
    //<<<(number_of_particles - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        g_vx[i] *= factor;
        g_vy[i] *= factor;
        g_vz[i] *= factor;
    }
}




void Ensemble_BDP::integrate_nvt_bdp
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    int  N           = para->N;
    int  grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    real time_step   = para->time_step;
    real *mass = gpu_data->mass;
    real *x    = gpu_data->x;
    real *y    = gpu_data->y;
    real *z    = gpu_data->z;
    real *vx   = gpu_data->vx;
    real *vy   = gpu_data->vy;
    real *vz   = gpu_data->vz;
    real *fx   = gpu_data->fx;
    real *fy   = gpu_data->fy;
    real *fz   = gpu_data->fz;
    real *potential_per_atom = gpu_data->potential_per_atom;
    real *virial_per_atom_x  = gpu_data->virial_per_atom_x; 
    real *virial_per_atom_y  = gpu_data->virial_per_atom_y;
    real *virial_per_atom_z  = gpu_data->virial_per_atom_z;
    real *thermo             = gpu_data->thermo;
    real *box_length         = gpu_data->box_length;

    // standard velocity-Verlet
    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);
    force->compute(para, gpu_data);
    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // get thermo
    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];
    gpu_find_thermo<<<5, 1024>>>
    (
        N, N_fixed, fixed_group, label, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    );

    // re-scale the velocities
    real *ek;
    MY_MALLOC(ek, real, sizeof(real) * 1);
    cudaMemcpy(ek, thermo + 1, sizeof(real) * 1, cudaMemcpyDeviceToHost);
    int ndeg = 3 * (N - N_fixed);
    real sigma = ndeg * K_B * temperature * 0.5;
    real factor = resamplekin(ek[0], sigma, ndeg, temperature_coupling);
    factor = sqrt(factor / ek[0]);
    MY_FREE(ek);
    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>(N, vx, vy, vz, factor);
}




static __global__ void find_vc_and_ke
(
    int  *g_group_size,
    int  *g_group_size_sum,
    int  *g_group_contents,
    real *g_mass, 
    real *g_vx, 
    real *g_vy, 
    real *g_vz, 
    real *g_vcx,
    real *g_vcy,
    real *g_vcz,
    real *g_ke
)
{
    //<<<number_of_groups, 512>>>

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int group_size = g_group_size[bid];
    int offset = g_group_size_sum[bid];
    int number_of_patches = (group_size - 1) / 512 + 1; 

    __shared__ real s_mc[512]; // center of mass
    __shared__ real s_vx[512]; // center of mass velocity
    __shared__ real s_vy[512];
    __shared__ real s_vz[512];
    __shared__ real s_ke[512]; // relative kinetic energy

    s_mc[tid] = ZERO;
    s_vx[tid] = ZERO;
    s_vy[tid] = ZERO;
    s_vz[tid] = ZERO;
    s_ke[tid] = ZERO;
    
    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int n = tid + patch * 512;
        if (n < group_size)
        {  
            int index = g_group_contents[offset + n];     
            real mass = g_mass[index];
            real vx = g_vx[index];
            real vy = g_vy[index];
            real vz = g_vz[index];

            s_mc[tid] += mass;
            s_vx[tid] += mass * vx;
            s_vy[tid] += mass * vy;
            s_vz[tid] += mass * vz;
            s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
    }
    __syncthreads();

    if (tid < 256) 
    { 
        s_mc[tid] += s_mc[tid + 256]; 
        s_vx[tid] += s_vx[tid + 256];
        s_vy[tid] += s_vy[tid + 256];
        s_vz[tid] += s_vz[tid + 256];
        s_ke[tid] += s_ke[tid + 256];
    } 
    __syncthreads();

    if (tid < 128) 
    { 
        s_mc[tid] += s_mc[tid + 128]; 
        s_vx[tid] += s_vx[tid + 128];
        s_vy[tid] += s_vy[tid + 128];
        s_vz[tid] += s_vz[tid + 128];
        s_ke[tid] += s_ke[tid + 128];
    } 
    __syncthreads();

    if (tid <  64) 
    { 
        s_mc[tid] += s_mc[tid + 64]; 
        s_vx[tid] += s_vx[tid + 64];
        s_vy[tid] += s_vy[tid + 64];
        s_vz[tid] += s_vz[tid + 64];
        s_ke[tid] += s_ke[tid + 64];
    } 
    __syncthreads();

    if (tid <  32) 
    { 
        warp_reduce(s_mc, tid);  
        warp_reduce(s_vx, tid); 
        warp_reduce(s_vy, tid); 
        warp_reduce(s_vz, tid);    
        warp_reduce(s_ke, tid);       
    }  

    if (tid == 0) 
    { 
        real mc = s_mc[0];
        real vx = s_vx[0] / mc;
        real vy = s_vy[0] / mc;
        real vz = s_vz[0] / mc;
        g_vcx[bid] = vx; // center of mass velocity
        g_vcy[bid] = vy;
        g_vcz[bid] = vz;

        // relative kinetic energy times 2
        g_ke[bid] = (s_ke[0] - mc * (vx * vx + vy * vy + vz * vz)) * HALF; 
        
    }
}




static __global__ void gpu_scale_velocity
(
    int number_of_particles, 
    int label_1,
    int label_2,
    int *g_atom_label, 
    real factor_1,
    real factor_2,
    real *g_vcx, 
    real *g_vcy,
    real *g_vcz,
    real *g_ke,
    real *g_vx, 
    real *g_vy, 
    real *g_vz
)
{
    // <<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number_of_particles)
    {
        int atom_label = g_atom_label[n];     
                 
        if (atom_label == label_1) 
        {
            // center of mass velocity for the source
            real vcx = g_vcx[atom_label]; 
            real vcy = g_vcy[atom_label];
            real vcz = g_vcz[atom_label];  

            // momentum is conserved
            g_vx[n] = vcx + factor_1 * (g_vx[n] - vcx);
            g_vy[n] = vcy + factor_1 * (g_vy[n] - vcy);
            g_vz[n] = vcz + factor_1 * (g_vz[n] - vcz);
        }
        if (atom_label == label_2)
        {
            // center of mass velocity for the sink
            real vcx = g_vcx[atom_label]; 
            real vcy = g_vcy[atom_label];
            real vcz = g_vcz[atom_label];  

            // momentum is conserved
            g_vx[n] = vcx + factor_2 * (g_vx[n] - vcx);
            g_vy[n] = vcy + factor_2 * (g_vy[n] - vcy);
            g_vz[n] = vcz + factor_2 * (g_vz[n] - vcz);
        }
    }
}




// integrate by one step, with heating and cooling, using the BDP method
void Ensemble_BDP::integrate_heat_bdp
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    int N         = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    real time_step   = para->time_step;
    real *mass = gpu_data->mass;
    real *x    = gpu_data->x;
    real *y    = gpu_data->y;
    real *z    = gpu_data->z;
    real *vx   = gpu_data->vx;
    real *vy   = gpu_data->vy;
    real *vz   = gpu_data->vz;
    real *fx   = gpu_data->fx;
    real *fy   = gpu_data->fy;
    real *fz   = gpu_data->fz;
    int *group_size = gpu_data->group_size;
    int *group_size_sum = gpu_data->group_size_sum;
    int *group_contents = gpu_data->group_contents;

    int label_1 = source;
    int label_2 = sink;
    int Ng = para->number_of_groups;

    real kT1 = K_B * (temperature + delta_temperature); 
    real kT2 = K_B * (temperature - delta_temperature); 
    real dN1 = (real) DIM * (cpu_data->group_size[source] - 1);
    real dN2 = (real) DIM * (cpu_data->group_size[sink] - 1);
    real sigma_1 = dN1 * kT1 * 0.5;
    real sigma_2 = dN2 * kT2 * 0.5;

    // allocate some memory (to be improved)
    real *ek;
    MY_MALLOC(ek, real, sizeof(real) * Ng);
    real *vcx, *vcy, *vcz, *ke;
    cudaMalloc((void**)&vcx, sizeof(real) * Ng);
    cudaMalloc((void**)&vcy, sizeof(real) * Ng);
    cudaMalloc((void**)&vcz, sizeof(real) * Ng);
    cudaMalloc((void**)&ke, sizeof(real) * Ng);

    // veloicty-Verlet
    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);
    force->compute(para, gpu_data);
    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // get center of mass velocity and relative kinetic energy
    find_vc_and_ke<<<Ng, 512>>>
    (
        group_size, group_size_sum, group_contents, 
        mass, vx, vy, vz, vcx, vcy, vcz, ke
    );
    cudaMemcpy(ek, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);

    // get the re-scaling factors
    real factor_1 
        = resamplekin(ek[label_1], sigma_1, dN1, temperature_coupling);
    real factor_2 
        = resamplekin(ek[label_2], sigma_2, dN2, temperature_coupling);
    factor_1 = sqrt(factor_1 / ek[label_1]);
    factor_2 = sqrt(factor_2 / ek[label_2]);
    
    // accumulate the energies transferred from the system to the baths
    energy_transferred[0] += ek[label_1] * (1.0 - factor_1 * factor_1);
    energy_transferred[1] += ek[label_2] * (1.0 - factor_2 * factor_2);

    // re-scale the velocities
    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>
    (
        N, label_1, label_2, gpu_data->label, factor_1, factor_2, 
        vcx, vcy, vcz, ke, vx, vy, vz
    );

    // clean up
    MY_FREE(ek); cudaFree(vcx); cudaFree(vcy); cudaFree(vcz); cudaFree(ke);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

}



 
void Ensemble_BDP::compute
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    if (type == 4)
    {
        integrate_nvt_bdp(para, cpu_data, gpu_data, force);
    }
    else
    {
        integrate_heat_bdp(para, cpu_data, gpu_data, force);
    }
}




// The following functions are from Bussi's website
// https://sites.google.com/site/giovannibussi/Research/algorithms
// I have only added "static" in front of the functions, 
// without any other changes




static double resamplekin(double kk,double sigma, int ndeg, double taut){
/*
  kk:    present value of the kinetic energy of the atoms to be thermalized (in arbitrary units)
  sigma: target average value of the kinetic energy (ndeg k_b T/2)  (in the same units as kk)
  ndeg:  number of degrees of freedom of the atoms to be thermalized
  taut:  relaxation time of the thermostat, in units of 'how often this routine is called'
*/
  double factor,rr;
  if(taut>0.1){
    factor=exp(-1.0/taut);
  } else{
    factor=0.0;
  }
  rr = gasdev();
  return kk + (1.0-factor)* (sigma*(resamplekin_sumnoises(ndeg-1)+rr*rr)/ndeg-kk)
            + 2.0*rr*sqrt(kk*sigma/ndeg*(1.0-factor)*factor);
}




static double resamplekin_sumnoises(int nn){
/*
  returns the sum of n independent gaussian noises squared
   (i.e. equivalent to summing the square of the return values of nn calls to gasdev)
*/
  double rr;
  if(nn==0) {
    return 0.0;
  } else if(nn==1) {
    rr=gasdev();
    return rr*rr;
  } else if(nn%2==0) {
    return 2.0*gamdev(nn/2);
  } else {
    rr=gasdev();
    return 2.0*gamdev((nn-1)/2) + rr*rr;
  }
}




static double gamdev(const int ia)
{
	int j;
	double am,e,s,v1,v2,x,y;

	if (ia < 1) {}; // FATAL ERROR
	if (ia < 6) {
		x=1.0;
		for (j=1;j<=ia;j++) x *= ran1();
		x = -log(x);
	} else {
		do {
			do {
				do {
					v1=ran1();
					v2=2.0*ran1()-1.0;
				} while (v1*v1+v2*v2 > 1.0);
				y=v2/v1;
				am=ia-1;
				s=sqrt(2.0*am+1.0);
				x=s*y+am;
			} while (x <= 0.0);
			e=(1.0+y*y)*exp(am*log(x/am)-s*y);
		} while (ran1() > e);
	}
	return x;
}




static double gasdev()
{
	static int iset=0;
	static double gset;
	double fac,rsq,v1,v2;

	if (iset == 0) {
		do {
			v1=2.0*ran1()-1.0;
			v2=2.0*ran1()-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}




static double ran1()
{
	const int IA=16807,IM=2147483647,IQ=127773,IR=2836,NTAB=32;
	const int NDIV=(1+(IM-1)/NTAB);
	const double EPS=3.0e-16,AM=1.0/IM,RNMX=(1.0-EPS);
	static int iy=0;
	static int iv[NTAB];
	int j,k;
	double temp;
        static int idum=0; /* ATTENTION: THE SEED IS HARDCODED */

	if (idum <= 0 || !iy) {
		if (-idum < 1) idum=1;
		else idum = -idum;
		for (j=NTAB+7;j>=0;j--) {
			k=idum/IQ;
			idum=IA*(idum-k*IQ)-IR*k;
			if (idum < 0) idum += IM;
			if (j < NTAB) iv[j] = idum;
		}
		iy=iv[0];
	}
	k=idum/IQ;
	idum=IA*(idum-k*IQ)-IR*k;
	if (idum < 0) idum += IM;
	j=iy/NDIV;
	iy=iv[j];
	iv[j] = idum;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}




