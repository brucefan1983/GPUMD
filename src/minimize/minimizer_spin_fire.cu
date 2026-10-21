/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/

#include "minimizer_spin_fire.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr int BLOCK = 256;
// E, |g|^2, |v|^2, g.v, max_i |g_i|^2, invalid state, relative spin update.
constexpr int NUM_STATS = 7;

__global__ void initialize_radius(int n, const double* spin, double* radius)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) radius[i] = sqrt(spin[i]*spin[i] + spin[n+i]*spin[n+i] + spin[2*n+i]*spin[2*n+i]);
}

__global__ void project_force(
  int n, bool fixed, const double* spin, const double* radius,
  const double* force, double* gradient, double* velocity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double sg = 0, sv = 0, s2 = 0;
  for (int c = 0; c < 3; ++c) {
    const int j = c*n+i;
    sg += spin[j]*force[j]; sv += spin[j]*velocity[j]; s2 += spin[j]*spin[j];
  }
  for (int c = 0; c < 3; ++c) {
    const int j = c*n+i;
    gradient[j] = fixed ? (radius[i] > 0 ? force[j] - spin[j]*sg/s2 : 0.0) : force[j];
    if (fixed) velocity[j] = radius[i] > 0 ? velocity[j] - spin[j]*sv/s2 : 0.0;
  }
}

__global__ void collect_stats(
  int n, const double* spin, const double* force, const double* energy,
  const double* gradient, const double* velocity, const double* previous_spin, double* result)
{
  __shared__ double sums[NUM_STATS][BLOCK];
  const int t = threadIdx.x;
  double e=0, g2=0, v2=0, power=0, gmax2=0, invalid=0, change=0;
  for (int i=t; i<n; i+=BLOCK) {
    e += energy[i];
    if (!isfinite(energy[i])) invalid=1;
    double gi2=0;
    for (int c=0; c<3; ++c) {
      const int j=c*n+i;
      const double g=gradient[j], v=velocity[j];
      if (!isfinite(spin[j]) || !isfinite(force[j]) || !isfinite(g) || !isfinite(v)) invalid=1;
      gi2 += g*g; v2 += v*v; power += g*v;
      const double scale=fmax(1.0,fmax(fabs(spin[j]),fabs(previous_spin[j])));
      change=fmax(change,fabs(spin[j]-previous_spin[j])/scale);
    }
    g2 += gi2; gmax2=fmax(gmax2,gi2);
  }
  sums[0][t]=e; sums[1][t]=g2; sums[2][t]=v2; sums[3][t]=power;
  sums[4][t]=gmax2; sums[5][t]=invalid; sums[6][t]=change;
  __syncthreads();
  for (int offset=BLOCK/2; offset>0; offset/=2) {
    if (t<offset) {
      for (int k=0; k<4; ++k) sums[k][t]+=sums[k][t+offset];
      for (int k=4; k<NUM_STATS; ++k) sums[k][t]=fmax(sums[k][t],sums[k][t+offset]);
    }
    __syncthreads();
  }
  if (t==0) for (int k=0; k<NUM_STATS; ++k) result[k]=sums[k][0];
}

__global__ void kick(int n, double dt, const double* gradient, double* velocity)
{
  const int j=blockIdx.x*blockDim.x+threadIdx.x;
  if (j<3*n) velocity[j]+=dt*gradient[j];
}

__global__ void move_spins(
  int n, bool fixed, double dt, double alpha, double ratio,
  const double* radius, const double* gradient, double* velocity, double* spin)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n || (fixed && radius[i]==0)) return;
  double v[3], norm2=0;
  for (int c=0; c<3; ++c) {
    v[c]=(1-alpha)*velocity[c*n+i]+alpha*ratio*gradient[c*n+i];
    norm2+=v[c]*v[c];
  }
  const double norm=sqrt(norm2);
  if (norm==0) return;
  // Limit each update to 0.1 rad (fixed) or 0.1 mu_B (variable).
  const double limit=fixed ? 0.1*radius[i] : 0.1;
  const double scale=fmin(1.0,limit/(dt*norm));
  if (!fixed) {
    for (int c=0; c<3; ++c) { velocity[c*n+i]=v[c]*scale; spin[c*n+i]+=dt*v[c]*scale; }
    return;
  }
  // Sphere exponential map, with parallel transport of the fictitious velocity.
  const double theta=dt*norm*scale/radius[i], co=cos(theta), si=sin(theta);
  double next[3], next_norm2=0;
  for (int c=0; c<3; ++c) {
    const double unit=spin[c*n+i]/radius[i], direction=v[c]/norm;
    next[c]=co*unit+si*direction;
    next_norm2+=next[c]*next[c];
    velocity[c*n+i]=norm*scale*(co*direction-si*unit);
  }
  const double rescale=radius[i]/sqrt(next_norm2);
  for (int c=0; c<3; ++c) spin[c*n+i]=next[c]*rescale;
}
} // namespace

void Minimizer_Spin_FIRE::compute(
  Force& force, Box& box, Atom& atom, GPU_Vector<double>& position, std::vector<Group>& group)
{
  CHECK(gpuSetDevice(0));
  const int n=number_of_atoms_, blocks=(n+BLOCK-1)/BLOCK;
  if (n<=0 || atom.spin_per_atom.size()!=3*n || atom.mforce_per_atom.size()!=3*n)
    PRINT_INPUT_ERROR("Spin FIRE requires complete spin and magnetic-force arrays.");
  GPU_Vector<double> radius(n), gradient(3*n), velocity(3*n,0.0), saved(3*n), stats_gpu(NUM_STATS);
  double stats[NUM_STATS];
  // Force dispatch wraps its position buffer. Use a private copy to preserve the
  // caller's positions, periodic images and lattice velocities exactly.
  position.copy_to_device(position_per_atom_temp_.data());
  initialize_radius<<<blocks,BLOCK>>>(n,atom.spin_per_atom.data(),radius.data());
  GPU_CHECK_KERNEL
  auto evaluate = [&]() {
    force.compute(box,position_per_atom_temp_,atom.type,group,atom.potential_per_atom,
      atom.force_per_atom,atom.virial_per_atom,atom.velocity_per_atom,atom.mass,
      atom.spin_per_atom,atom.mforce_per_atom);
    CHECK(gpuSetDevice(0));
    project_force<<<blocks,BLOCK>>>(n,fixed_magnitude_,atom.spin_per_atom.data(),radius.data(),
      atom.mforce_per_atom.data(),gradient.data(),velocity.data());
    GPU_CHECK_KERNEL
  };
  auto measure = [&]() {
    collect_stats<<<1,BLOCK>>>(n,atom.spin_per_atom.data(),atom.mforce_per_atom.data(),
      atom.potential_per_atom.data(),gradient.data(),velocity.data(),saved.data(),stats_gpu.data());
    GPU_CHECK_KERNEL
    stats_gpu.copy_to_host(stats);
    for (int k=0; k<NUM_STATS; ++k) if (!std::isfinite(stats[k])) stats[5]=1;
  };
  auto report = [&](int step, double dt) {
    printf("spin_fire step %d energy %.16e max_mforce %.9e dt %.6e\n",step,stats[0],sqrt(stats[4]),dt);
  };
  double dt=0.1, alpha=0.1;
  int positive=0, nonfinite_retries=0, power_resets=0, steps=0, tiny_updates=0;
  const int interval=std::max(1,number_of_steps_/20);
  atom.spin_per_atom.copy_to_device(saved.data());
  evaluate(); measure();
  if (stats[5]) PRINT_INPUT_ERROR("Spin FIRE encountered a non-finite initial state.");
  printf("\nSpin FIRE: %s magnitude; fixed lattice; tolerance %.9e eV/mu_B.\n",
    fixed_magnitude_ ? "fixed" : "variable",force_tolerance_);
  report(0,dt);
  double progress_residual=sqrt(stats[4]);
  int last_progress=0;
  const char* reason="maximum_steps";
  const char* stagnation="none";
  for (; steps<number_of_steps_; ) {
    if (sqrt(stats[4])<=force_tolerance_) { reason="converged"; break; }
    if (dt<1.0e-12) { reason="stalled"; stagnation="time_step_floor"; break; }
    atom.spin_per_atom.copy_to_device(saved.data());
    ++steps;
    // Semi-implicit Euler FIRE: kick, test power, mix and move. Energy is
    // diagnostic only: adding a constant to the potential must not affect FIRE.
    kick<<<(3*n+BLOCK-1)/BLOCK,BLOCK>>>(n,dt,gradient.data(),velocity.data());
    GPU_CHECK_KERNEL
    measure();
    if (stats[5]) PRINT_INPUT_ERROR("Spin FIRE encountered non-finite fictitious velocities.");
    if (stats[3]<=0) {
      velocity.fill(0.0); dt*=0.5; alpha=0.1; positive=0; ++power_resets;
    } else {
      move_spins<<<blocks,BLOCK>>>(n,fixed_magnitude_,dt,alpha,sqrt(stats[2]/stats[1]),
        radius.data(),gradient.data(),velocity.data(),atom.spin_per_atom.data());
      GPU_CHECK_KERNEL
      evaluate(); measure();
      if (stats[5]) {
        saved.copy_to_device(atom.spin_per_atom.data());
        velocity.fill(0.0); dt*=0.5; alpha=0.1; positive=0; ++nonfinite_retries;
        tiny_updates=0;
        // Restore all endpoint caches as well as spins, including magnetic forces.
        evaluate(); measure();
        if (stats[5]) PRINT_INPUT_ERROR("Spin FIRE could not restore a finite state.");
      } else {
        tiny_updates = stats[6]<=8*std::numeric_limits<double>::epsilon() ? tiny_updates+1 : 0;
        if (++positive>5) { dt=std::min(dt*1.1,1.0); alpha*=0.99; }
      }
    }
    const double residual=sqrt(stats[4]);
    // Accumulated improvement counts, rather than requiring a 1% improvement
    // on each step. This conservative limit also catches cycling at a model's
    // force precision floor; it never relaxes the requested force tolerance.
    if (residual<0.99*progress_residual) { progress_residual=residual; last_progress=steps; }
    if (steps%interval==0) report(steps,dt);
    if (residual<=force_tolerance_) { reason="converged"; break; }
    if (tiny_updates>=20 || steps-last_progress>=1000) {
      reason="stalled";
      stagnation=tiny_updates>=20 ? "unresolved_spin_updates" : "no_residual_progress";
      break;
    }
  }
  if (sqrt(stats[4])<=force_tolerance_) reason="converged";
  report(steps,dt);
  printf("Spin FIRE finished: %s; steps %d; power_resets %d; nonfinite_retries %d; stagnation %s.\n",
    reason,steps,power_resets,nonfinite_retries,stagnation);
  // Keep host-side spin state consistent for later commands. Physical velocities
  // are not FIRE's fictitious velocities and remain unchanged.
  atom.cpu_spin_per_atom.resize(3*n);
  atom.spin_per_atom.copy_to_host(atom.cpu_spin_per_atom.data());
}
