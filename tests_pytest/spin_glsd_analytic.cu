// SPDX-License-Identifier: LGPL-3.0-or-later
// Tests the production integrator with an independent harmonic Hamiltonian.
// Build: nvcc -std=c++14 -O3 -arch=sm_70 -Isrc tests_pytest/spin_glsd_analytic.cu \
//        src/integrate/spin_glsd.cu -o spin_glsd_analytic
#include "integrate/spin_glsd.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void harmonic(int n, double k, double b, const double* s, double* h)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < 3*n) h[i] = -k*s[i] + (i >= 2*n ? b : 0.0);
}

void run(const char* name, int n, double dt, int steps, double temp,
         double k, double b, double gamma, double mobility, bool zero, int seed)
{
  Atom atom;
  atom.number_of_atoms=n; atom.has_spin=true;
  atom.spin_per_atom.resize(3*n); atom.mforce_per_atom.resize(3*n);
  atom.spin_velocity_per_atom.resize(3*n);
  std::vector<double> s(3*n,0.0);
  if (!zero) { for (int i=0;i<n;++i) { s[i]=1; s[n+i]=.2; s[2*n+i]=-.3; } }
  atom.spin_per_atom.copy_from_host(s.data());
  Spin_GLSD integrator(gamma, mobility, temp > 0 ? temp : -1, seed, atom);
  const double natural_dt=dt*1000/TIME_UNIT_CONVERSION;
  for (int j=0;j<steps;++j) {
    harmonic<<<(3*n+127)/128,128>>>(n,k,b,atom.spin_per_atom.data(),atom.mforce_per_atom.data());
    integrator.compute1(natural_dt,atom);
    harmonic<<<(3*n+127)/128,128>>>(n,k,b,atom.spin_per_atom.data(),atom.mforce_per_atom.data());
    integrator.compute2(natural_dt,atom);
  }
  atom.spin_per_atom.copy_to_host(s.data());
  double mean[3]={}, var[3]={}, fourth[3]={}, err=0, energy=0;
  double t=dt*steps, decay=exp(-mobility*k*t), angle=gamma*b*t;
  double exact[3] = {zero ? 0 : decay*(cos(angle)-.2*sin(angle)),
                     zero ? 0 : decay*(sin(angle)+.2*cos(angle)),
                     (zero ? 0 : -.3)*decay + (k ? b/k*(1-decay) : mobility*b*t)};
  if (!k) exact[2]=(zero ? 0 : -.3)+mobility*b*t;
  for (int d=0;d<3;++d) {
    for (int i=0;i<n;++i) { mean[d]+=s[d*n+i]/n; err=std::max(err,std::abs(s[d*n+i]-exact[d])); }
    for (int i=0;i<n;++i) { double x=s[d*n+i]-mean[d]; var[d]+=x*x/n; fourth[d]+=x*x*x*x/n; }
  }
  int within=0;
  const double thermal_var=temp>0 && k>0 ? K_B*temp/k : 0;
  for (int i=0;i<n;++i) {
    double r2=s[i]*s[i]+s[n+i]*s[n+i]+s[2*n+i]*s[2*n+i];
    energy+=(.5*k*r2-b*s[2*n+i])/n;
    if (r2 < 3*thermal_var) ++within;
  }
  printf("{\"name\":\"%s\",\"n\":%d,\"dt_ps\":%.17g,\"time_ps\":%.17g,\"temperature\":%g,\"error\":%.17g,\"energy\":%.17g,\"mean\":[%.17g,%.17g,%.17g],\"variance\":[%.17g,%.17g,%.17g],\"fourth\":[%.17g,%.17g,%.17g],\"radial_cdf\":%.17g}\n",
    name,n,dt,t,temp,err,energy,mean[0],mean[1],mean[2],var[0],var[1],var[2],fourth[0],fourth[1],fourth[2],double(within)/n);
}
int main()
{
  for (double dt : {.1,.05,.025}) {
    run("damped_field",4,dt,int(5/dt),0,.4,.2,3,2,false,731);
    run("precession",4,dt,int(5/dt),0,0,.2,3,0,false,731);
  }
  run("radial",4,.01,2000,0,.4,0,3,2,false,731);
  run("zero_field",4,.01,500,0,0,.2,3,2,true,731);
  run("zero_rest",4,.01,500,0,.4,0,3,2,true,731);
  for (double temp : {100.,400.}) {
    for (double dt : {.8,.4,.05})
      run("thermal",131072,dt,int(40/dt),temp,.4,0,3,2,true,731);
  }
  run("thermal_field",131072,.05,800,300,.4,.2,3,2,true,851);
  run("thermal_zero_transient",131072,.001,100,300,.4,0,3,2,true,941);
}
