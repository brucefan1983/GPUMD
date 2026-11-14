/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "model/box.cuh"
#include "force/nep_spin/spin_storage.cuh"
#include "force/nep_spin/spin_warp.cuh"
#include "utilities/nep_utilities.cuh"
#include <type_traits>

namespace nep_spin {
// The generic and compiled layouts use the same arithmetic and output ordering.
template<int Capacity, int Spins = 35, int Harmonics = 25>
struct RuntimeLayout : View {
  explicit RuntimeLayout(View v) : View(v) {}
  enum : int { component_capacity = Capacity, spin_capacity = Spins, harmonic_capacity = Harmonics, tile = 1, edge_cached = 0 };
  const float* edge_radial = nullptr;
  __device__ inline Primitive primitive(int i) const { return primitives[i]; }
  __device__ inline Sector sector(int i) const { return sectors[i]; }
  __device__ inline Term term(int i) const { return terms[i]; }
  __device__ inline Center center(int i) const { return centers[i]; }
  __device__ inline Harmonic harmonic(int i) const { return harmonics[i]; }
};
using DynamicLayout = RuntimeLayout<100>;
using WideLayout = RuntimeLayout<625>; // Maximum (L+1)^2 * (min(L,S)+1)^2.
// Fixed spin width lets the compiler unroll the neighbor contraction.
template<int Spins, int Harmonics>
struct FixedSpinLayout : RuntimeLayout<100, Spins, Harmonics> {
  explicit FixedSpinLayout(View v) : RuntimeLayout<100, Spins, Harmonics>(v) {}
  enum : int { spin_components = Spins };
};
template<class Launch>
inline void launch_runtime_layout(View v, Launch launch) {
  if (v.components > 100) launch(WideLayout(v));
  else if (v.lmax == 0 && v.spin_components == 1) launch(FixedSpinLayout<1, 1>(v));
  else if (v.spin_components == 10) launch(FixedSpinLayout<10, 25>(v));
  else launch(DynamicLayout(v));
}
// Primitive zero is unity and holds the geometry-only sum for center-spin terms.
// Values and Cartesian derivatives are evaluated together, including at s=0.
template<class Real> struct SpinValue { Real value, derivative[3]; };
template<class Real> __device__ inline Real power(Real x, int n) {
  Real result = 1;
  for (int i = 0; i < n; ++i) result *= x;
  return result;
}
template<class Layout, class Real>
__device__ __forceinline__ void spherical(Layout v, const Real* x, SpinValue<Real>* y) {
  #pragma unroll
  for (int i = 0; i < Layout::harmonic_capacity; ++i) y[i] = {0, {0, 0, 0}};
  #pragma unroll
  for (int i = 0; i < v.harmonic_count; ++i) {
    const Harmonic h = v.harmonic(i);
    if (h.component >= (v.lmax + 1) * (v.lmax + 1)) continue;
    const int p[3] = {h.x, h.y, h.z};
    Real val = h.value;
    for (int d = 0; d < 3; ++d) val *= power(x[d], p[d]);
    y[h.component].value += val;
    for (int d = 0; d < 3; ++d) {
      if (!p[d]) continue;
      Real grad = h.value * p[d];
      for (int k = 0; k < 3; ++k) grad *= power(x[k], p[k] - (d == k));
      y[h.component].derivative[d] += grad;
    }
  }
}
template<class Layout, class Real>
__device__ __forceinline__ void primitives_at(Layout v, const Real* s, SpinValue<Real>* phi) {
  SpinValue<Real> y[Layout::harmonic_capacity];
  spherical(v, s, y);
  const Real s2 = s[0]*s[0] + s[1]*s[1] + s[2]*s[2];
  #pragma unroll
  for (int p = 0; p < v.primitive_count; ++p) {
    const Primitive label = v.primitive(p);
    const int exponent = (label.degree - label.l) / 2;
    const Real factor = power(s2, exponent);
    const Real grad_factor = exponent ? 2 * exponent * power(s2, exponent-1) : 0;
    for (int m = 0; m < 2 * label.l + 1; ++m) {
      const SpinValue<Real> harmonic = y[label.l*label.l+m];
      SpinValue<Real>& value = phi[label.offset+m];
      value.value = factor * harmonic.value;
      for (int d = 0; d < 3; ++d)
        value.derivative[d] = factor * harmonic.derivative[d] + grad_factor*s[d]*harmonic.value;
    }
  }
}
template<class Real> __device__ inline void read_spin(int n, int i, const int* type, const int* active,
                                 const double* spins, Real* s) {
  for (int d = 0; d < 3; ++d) s[d] = active[type[i]] ? spins[d*n+i] : 0;
}
template<class Real> __device__ inline void displacement(int n, int i, int j, int slot, Box box,
                                    const double* positions, const double* edges, int plane,
                                    Real* r) {
  if (edges) {
    for (int d = 0; d < 3; ++d) r[d] = edges[size_t(d)*plane+slot];
  } else {
    // Reduce the image in double before converting a large-cell displacement to Real.
    double delta[3];
    for (int d = 0; d < 3; ++d) delta[d] = positions[d*n+j] - positions[d*n+i];
    apply_mic(box, delta[0], delta[1], delta[2]);
    for (int d = 0; d < 3; ++d) r[d] = delta[d];
  }
}
// Contract the shared angular coefficients with the channel mix once per model evaluation.
// B[p,c,k] = sum_a angular[p,a,k] mix[a,c]; reuse B in forward and adjoint kernels.
template<class Real>
static __global__ void contract_radial(int count, int channels, int na, int basis,
  const float* angular, const float* mix, Real* result) {
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= count) return;
  const int k = i % (basis+1), c = (i/(basis+1)) % channels;
  const int pair = i / ((basis+1)*channels);
  Real value = 0;
  for (int a = 0; a <= na; ++a)
    value += Real(angular[(pair*(na+1)+a)*(basis+1)+k]) * mix[a*channels+c];
  result[i] = value;
}
// Materialize only the contracted value and derivative, not the much larger
// basis/harmonic tables. All channels and both passes share each edge evaluation.
template<class Layout, int Basis>
static __global__ void cache_radial_edges(
  int n, int plane, int types, const int* type, const int* nn, const int* nl,
  const double* edges, const float* cutoffs, const float* coefficients, float* cache,
  Box box = Box{}, const double* positions = nullptr) {
  const int slot = blockIdx.x*blockDim.x+threadIdx.x;
  if (slot >= plane) return;
  const int i = slot % n;
  if (slot/n >= nn[i]) return;
  const int pair = type[i]*types+type[nl[slot]];
  float r[3];
  displacement(n, i, nl[slot], slot, box, positions, edges, plane, r);
  const float distance = sqrtf(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]), cutoff = cutoffs[pair];
  float fc, fcp, fn[Basis+1], fnp[Basis+1];
  find_fc_and_fcp(cutoff, 1/cutoff, distance, fc, fcp);
  find_fn_and_fnp(Basis, 1/cutoff, distance, fc, fcp, fn, fnp);
  #pragma unroll
  for (int c = 0; c < Layout::channels; ++c) {
    float value = 0, derivative = 0;
    #pragma unroll
    for (int k = 0; k <= Basis; ++k) {
      const float coefficient = coefficients[(pair*Layout::channels+c)*(Basis+1)+k];
      value += fn[k]*coefficient;
      derivative += fnp[k]*coefficient;
    }
    cache[size_t(2*c)*plane+slot] = value;
    cache[size_t(2*c+1)*plane+slot] = derivative;
  }
}
template<class Layout, class Real>
__device__ inline void radial(Layout v, int slot, int plane, int c, int channels, int basis, int pair, float cutoff,
                              Real distance, const Real* coefficients,
                              Real& weight, Real& derivative) {
  if (Layout::edge_cached) {
    weight = v.edge_radial[size_t(2*c)*plane+slot];
    derivative = v.edge_radial[size_t(2*c+1)*plane+slot];
    return;
  }
  Real fc, fcp, fn[17], fnp[17];
  const Real inv = Real(1)/cutoff;
  find_fc_and_fcp(Real(cutoff), inv, distance, fc, fcp);
  find_fn_and_fnp(basis, inv, distance, fc, fcp, fn, fnp);
  weight = derivative = 0;
  for (int k = 0; k <= basis; ++k) {
    const Real coefficient = coefficients[(pair*channels+c)*(basis+1)+k];
    weight += fn[k]*coefficient;
    derivative += fnp[k]*coefficient;
  }
}

template<class Layout, class Real = float, class Output = float>
static __global__ void descriptors(
  Layout v, int n, int structural_dim, int types, int na, int basis, Box box,
  const int* type, const int* active, const double* positions, const double* spins,
  const int* nn, const int* nl, const double* edges, int plane, const float* cutoffs,
  const Real* coefficients, const float* parameters, float* moments, Output* q) {
  const int work = (blockIdx.x*blockDim.x+threadIdx.x) / Layout::tile;
  const int lane = threadIdx.x % Layout::tile;
  const int i = work / v.channels, c = work % v.channels;
  // Capture participation before divergent neighbor loops. An activemask taken
  // at a reduction can omit lanes still processing their last neighbor.
  const WarpMask mask = Layout::tile > 1 ? warp_ballot(i < n) : 0;
  if (i >= n) return;
  Real s[3], h[Layout::component_capacity] = {};
  SpinValue<Real> phi_i[Layout::spin_capacity];
  read_spin(n, i, type, active, spins, s);
  primitives_at(v, s, phi_i);
  const float* weights = parameters + (na+1)*v.channels;
  const float* center = weights + v.path_count*v.channels;
  Real density[Layout::harmonic_capacity][Layout::spin_capacity] = {};
  const int geometry_count = (v.lmax+1)*(v.lmax+1);
  for (int neighbor = lane; neighbor < nn[i]; neighbor += Layout::tile) {
    int slot = neighbor*n+i, j = nl[slot], pair = type[i]*types+type[j];
    Real r[3], u[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    Real distance = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (distance >= cutoffs[pair]) continue;
    for (int d = 0; d < 3; ++d) u[d] = r[d]/distance;
    SpinValue<Real> y[Layout::harmonic_capacity], phi_j[Layout::spin_capacity];
    spherical(v, u, y);
    read_spin(n, j, type, active, spins, s);
    primitives_at(v, s, phi_j);
    Real weight, derivative;
    radial(v, slot, plane, c, v.channels, basis, pair, cutoffs[pair], distance, coefficients, weight, derivative);
    for (int g = 0; g < geometry_count; ++g) {
      const Real weighted_y = weight*y[g].value;
      density[g][0] += weighted_y;
      #pragma unroll
      for (int a = 1; a < v.spin_components; ++a)
        density[g][a] += weighted_y*phi_j[a].value;
    }
  }
  for (int t = 0; t < v.term_count; ++t) {
    const Term term = v.term(t);
    const Real* sums = density[term.geometry];
    const Real product = term.neighbor ? sums[term.spin] : sums[0]*phi_i[term.spin].value;
    h[term.output] += weights[term.path*v.channels+c]*term.coefficient*product;
  }
  #pragma unroll
  for (int m = 0; m < v.components; ++m) {
    #pragma unroll
    for (int offset = Layout::tile/2; offset > 0; offset /= 2)
      h[m] += __shfl_down_sync(mask, h[m], offset, Layout::tile);
  }
  if (lane != 0) return;
  #pragma unroll
  for (int t = 0; t < v.center_count; ++t) {
    const Center term = v.center(t);
    h[term.output] += center[term.weight*v.channels+c]*phi_i[term.primitive].value;
  }
  #pragma unroll
  for (int m = 0; m < v.components; ++m) moments[size_t(c*v.components+m)*n+i] = h[m];
  if (c == 0) {
    int column = structural_dim;
    #pragma unroll
    for (int p = 1; p < v.primitive_count; ++p)
      if (v.primitive(p).l == 0) q[(column++)*n+i] = phi_i[v.primitive(p).offset].value;
  }
  #pragma unroll
  for (int k = 0; k < v.sector_count; ++k) {
    Sector sector = v.sector(k);
    Real norm = 0;
    for (int m = 0; m < sector.size; ++m) norm += h[sector.offset+m]*h[sector.offset+m];
    q[(structural_dim+sector.norm_column+c)*n+i] = norm/(sector.size);
    if (sector.scalar_column >= 0)
      q[(structural_dim+sector.scalar_column+c)*n+i] = h[sector.offset];
  }
}

template<class Layout>
__device__ __forceinline__ Term force_term(Layout v, int i, std::false_type) { return v.term(i); }
template<class Layout>
__device__ __forceinline__ Term force_term(Layout v, int i, std::true_type) { return v.quadratic_term(i); }

template<class Layout, bool CenterVirial = false, bool Quadratic = false, bool FuseChannels = false, class Real = float>
static __global__ void forces(
  Layout v, int n, int structural_dim, int types, int na, int basis, Box box,
  const int* type, const int* active, const double* positions, const double* spins,
  const int* nn, const int* nl, const double* edges, int plane, const float* cutoffs,
  const Real* coefficients, const float* parameters, const float* moments, const float* fp,
  double* force, double* mforce, double* virial, bool neighbor_virial) {
  static_assert(!Quadratic || (Layout::spin_capacity == 10 &&
    Layout::harmonic_capacity == 9 && Layout::tile > 1), "quadratic tiled layout required");
  static_assert(!FuseChannels || Quadratic, "channel fusion requires the quadratic adjoint");
  constexpr int channels = FuseChannels ? 3 : 1;
  const int work = (blockIdx.x*blockDim.x+threadIdx.x) / Layout::tile;
  const int lane = threadIdx.x % Layout::tile;
  const int i = FuseChannels ? work : work / v.channels;
  const int first_channel = FuseChannels ? 0 : work % v.channels;
  // Capture participation before divergent neighbor loops. An activemask taken
  // at a reduction can omit lanes still processing their last neighbor.
  const WarpMask mask = Layout::tile > 1 ? warp_ballot(i < n) : 0;
  if (i >= n) return;
  Real s[3], center_pull[3] = {};
  double center_force[3] = {};
  double center_virial[CenterVirial ? 9 : 1] = {};
  SpinValue<Real> phi_i[Layout::spin_capacity];
  read_spin(n, i, type, active, spins, s);
  primitives_at(v, s, phi_i);
  const int geometry_count = (v.lmax+1)*(v.lmax+1);
  Real density_pull[Quadratic ? 1 : Layout::harmonic_capacity][Quadratic ? 1 : Layout::spin_capacity] = {};
  Real center_gradient[Quadratic ? 1 : Layout::harmonic_capacity][3] = {}, geometry_sum[channels][Layout::harmonic_capacity] = {};
  // The native/JIT force launch uses 64 threads; each tile shares one adjoint.
  constexpr int groups = 64 / Layout::tile;
  constexpr int neighbor_offset = 4*9, adjoint_size = neighbor_offset+9*9;
  __shared__ volatile Real edge_pull[Quadratic ? adjoint_size * groups * channels : 1];
  const int group = threadIdx.x / Layout::tile;
  const float* weights = parameters + (na+1)*v.channels;
  const float* center = weights + v.path_count*v.channels;
  // Training shares geometry and scatters one edge result across all three channels.
  for (int channel = 0; channel < channels; ++channel) {
    const int c = first_channel + channel;
    Real pull[Layout::component_capacity] = {};
    volatile Real* channel_pull = edge_pull + channel*adjoint_size*groups;
    if (!Quadratic || lane == 0) {
      #pragma unroll
      for (int k = 0; k < v.sector_count; ++k) {
        Sector sector = v.sector(k);
        Real derivative = 2*fp[(structural_dim+sector.norm_column+c)*n+i]/(sector.size);
        for (int m = 0; m < sector.size; ++m)
          pull[sector.offset+m] = derivative*moments[size_t(c*v.components+sector.offset+m)*n+i];
        if (sector.scalar_column >= 0) pull[sector.offset] += fp[(structural_dim+sector.scalar_column+c)*n+i];
      }
    }
    if (lane == 0) {
      #pragma unroll
      for (int t = 0; t < v.center_count; ++t) {
        Center term = v.center(t);
        for (int d = 0; d < 3; ++d)
          center_pull[d] += pull[term.output]*center[term.weight*v.channels+c]*phi_i[term.primitive].derivative[d];
      }
      if (c == 0) {
        int column = structural_dim;
        #pragma unroll
        for (int p = 1; p < v.primitive_count; ++p)
          if (v.primitive(p).l == 0) {
            for (int d = 0; d < 3; ++d)
              center_pull[d] += fp[column*n+i]*phi_i[v.primitive(p).offset].derivative[d];
            ++column;
          }
      }
    }
    if (Quadratic) {
      for (int k = lane; k < adjoint_size; k += Layout::tile) channel_pull[k*groups+group] = 0;
      __syncwarp(mask);
    }
    if (!Quadratic || lane == 0) {
      constexpr int term_unroll = Quadratic ? 512 : 1;
      #pragma unroll term_unroll
      for (int t = 0; t < v.term_count; ++t) {
        const Term term = force_term(v, t, std::integral_constant<bool, Quadratic>{});
        const Real adjoint = pull[term.output]*weights[term.path*v.channels+c]*term.coefficient;
        if (term.neighbor) {
          if (Quadratic) channel_pull[(neighbor_offset+9*term.geometry+term.spin-1)*groups+group] += adjoint;
          else density_pull[term.geometry][term.spin] += adjoint;
        } else {
          const SpinValue<Real> spin = phi_i[term.spin];
          if (Quadratic) {
            channel_pull[term.geometry*groups+group] += adjoint*spin.value;
            for (int d = 0; d < 3; ++d)
              channel_pull[((d+1)*9+term.geometry)*groups+group] += adjoint*spin.derivative[d];
          } else {
            density_pull[term.geometry][0] += adjoint*spin.value;
            for (int d = 0; d < 3; ++d)
              center_gradient[term.geometry][d] += adjoint*spin.derivative[d];
          }
        }
      }
    }
    if (Quadratic) {
      __syncwarp(mask);
      // phi[1..9] spans x,y,z,s2,xz,xy,Qyy,yz,Qzz-xx. Fold it to Cartesian form once.
      for (int g = lane; g < 9; g += Layout::tile) {
        const int base = neighbor_offset+9*g;
        Real a[9];
        #pragma unroll
        for (int k = 0; k < 9; ++k) a[k] = channel_pull[(base+k)*groups+group];
        const Real linear = v.harmonic(3).value, diagonal = v.harmonic(4).value;
        const Real difference = v.harmonic(5).value, cross = v.harmonic(6).value;
        const Real cartesian[9] = {
          linear*a[0], linear*a[1], linear*a[2],
          a[3]+diagonal*a[6]-difference*a[8], a[3]-2*diagonal*a[6],
          a[3]+diagonal*a[6]+difference*a[8], cross*a[5], cross*a[4], cross*a[7]};
        #pragma unroll
        for (int k = 0; k < 9; ++k) channel_pull[(base+k)*groups+group] = cartesian[k];
      }
      __syncwarp(mask);
    }
  }
  for (int neighbor = lane; neighbor < nn[i]; neighbor += Layout::tile) {
    int slot = neighbor*n+i, j = nl[slot], pair = type[i]*types+type[j];
    Real r[3], u[3];
    displacement(n, i, j, slot, box, positions, edges, plane, r);
    Real distance = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (distance >= cutoffs[pair]) continue;
    for (int d = 0; d < 3; ++d) u[d] = r[d]/distance;
    SpinValue<Real> y[Layout::harmonic_capacity], phi_j[Layout::spin_capacity];
    spherical(v, u, y);
    read_spin(n, j, type, active, spins, s);
    if (!Quadratic) primitives_at(v, s, phi_j);
    double edge_force[3] = {}, edge_mforce[3] = {}, edge_virial[9] = {};
    const int rows[9] = {0,1,2,0,0,1,1,2,2};
    const int cols[9] = {0,1,2,1,2,2,0,0,1};
    for (int channel = 0; channel < channels; ++channel) {
      const int c = first_channel + channel;
      const volatile Real* channel_pull = edge_pull + channel*adjoint_size*groups;
      Real weight, derivative;
      radial(v, slot, plane, c, v.channels, basis, pair, cutoffs[pair], distance, coefficients, weight, derivative);
      Real radial_pull = 0, angular_pull[3] = {}, neighbor_pull[3] = {};
      if (Quadratic) {
        #pragma unroll
        for (int g = 0; g < 9; ++g) {
          Real a[9];
          #pragma unroll
          for (int k = 0; k < 9; ++k) a[k] = channel_pull[(neighbor_offset+9*g+k)*groups+group];
          const Real value = channel_pull[g*groups+group] +
            s[0]*(a[0]+a[3]*s[0]+a[6]*s[1]+a[7]*s[2]) +
            s[1]*(a[1]+a[4]*s[1]+a[8]*s[2]) + s[2]*(a[2]+a[5]*s[2]);
          const Real gradient[3] = {
            a[0]+2*a[3]*s[0]+a[6]*s[1]+a[7]*s[2],
            a[1]+2*a[4]*s[1]+a[6]*s[0]+a[8]*s[2],
            a[2]+2*a[5]*s[2]+a[7]*s[0]+a[8]*s[1]};
          const Real weighted_y = weight*y[g].value;
          geometry_sum[channel][g] += weighted_y;
          radial_pull += value*y[g].value;
          for (int d = 0; d < 3; ++d) {
            angular_pull[d] += (weight*value)*y[g].derivative[d];
            neighbor_pull[d] += weighted_y*gradient[d];
          }
        }
      } else {
        Real spin_pull[Layout::spin_capacity] = {};
        for (int g = 0; g < geometry_count; ++g) {
          Real y_pull = density_pull[g][0];
          const Real weighted_y = weight*y[g].value;
          geometry_sum[channel][g] += weighted_y;
          #pragma unroll
          for (int a = 1; a < v.spin_components; ++a) {
            y_pull += density_pull[g][a]*phi_j[a].value;
            spin_pull[a] += weighted_y*density_pull[g][a];
          }
          radial_pull += y_pull*y[g].value;
          for (int d = 0; d < 3; ++d)
            angular_pull[d] += (weight*y_pull)*y[g].derivative[d];
        }
        for (int a = 1; a < v.spin_components; ++a)
          for (int d = 0; d < 3; ++d)
            neighbor_pull[d] += spin_pull[a]*phi_j[a].derivative[d];
      }
      Real longitudinal = 0;
      for (int d = 0; d < 3; ++d) longitudinal += angular_pull[d]*u[d];
      Real grad[3];
      for (int d = 0; d < 3; ++d) {
        grad[d] = derivative*radial_pull*u[d] + (angular_pull[d]-longitudinal*u[d])/distance;
        edge_force[d] += static_cast<double>(grad[d]);
        edge_mforce[d] += static_cast<double>(neighbor_pull[d]);
      }
      #pragma unroll
      for (int k = 0; k < 9; ++k)
        edge_virial[k] -= static_cast<double>(r[rows[k]]*grad[cols[k]]);
    }
    for (int d = 0; d < 3; ++d) {
      if (Layout::tile > 1) center_force[d] += edge_force[d];
      else atomicAdd(force+d*n+i, edge_force[d]);
      atomicAdd(force+d*n+j, -edge_force[d]);
      if (active[type[j]]) atomicAdd(mforce+d*n+j, -edge_mforce[d]);
    }
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
      if (CenterVirial) center_virial[k] += edge_virial[k];
      else atomicAdd(virial+k*n+(neighbor_virial ? j : i), edge_virial[k]);
    }
  }
  for (int channel = 0; channel < channels; ++channel) {
    const volatile Real* channel_pull = edge_pull + channel*adjoint_size*groups;
    for (int g = 0; g < geometry_count; ++g)
      for (int d = 0; d < 3; ++d)
        center_pull[d] += geometry_sum[channel][g]*(Quadratic
          ? channel_pull[((d+1)*9+g)*groups+group] : center_gradient[g][d]);
  }
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    #pragma unroll
    for (int offset = Layout::tile/2; offset > 0; offset /= 2) {
      center_pull[d] += __shfl_down_sync(mask, center_pull[d], offset, Layout::tile);
      center_force[d] += __shfl_down_sync(mask, center_force[d], offset, Layout::tile);
    }
  }
  // MD assigns each edge virial to its center. Reduce before the atomic write;
  // the training path retains its neighbor-owned per-atom convention.
  if (CenterVirial) {
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
      #pragma unroll
      for (int offset = Layout::tile/2; offset > 0; offset /= 2)
        center_virial[k] += __shfl_down_sync(mask, center_virial[k], offset, Layout::tile);
      if (lane == 0) atomicAdd(virial+k*n+i, center_virial[k]);
    }
  }
  if (lane != 0) return;
  if (Layout::tile > 1)
    for (int d = 0; d < 3; ++d) atomicAdd(force+d*n+i, center_force[d]);
  if (active[type[i]])
    for (int d = 0; d < 3; ++d) atomicAdd(mforce+d*n+i, -static_cast<double>(center_pull[d]));
}
} // namespace nep_spin
