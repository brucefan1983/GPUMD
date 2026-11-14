/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
// Compare the shared polynomial path with the generic adjoint, including tail tiles.
#include "force/nep_spin/spin_kernels.cuh"
#include "force/nep_spin/spin_md_layouts.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#define QUAD_CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } } while (0)

template<class T> T* allocate(int n) {
  T* p;
  QUAD_CHECK(cudaMallocManaged(&p, n*sizeof(T)));
  return p;
}

template<class Layout, bool CenterVirial = true, bool FuseChannels = false>
void check(int n) {
  Layout v;
  constexpr int dim = 96, types = 3, na = 4, basis = 8, neighbors = 6;
  const int plane = neighbors*n, parameters = (na+1+v.path_count+v.center_count)*v.channels;
  int* type = allocate<int>(n), *active = allocate<int>(types);
  int* nn = allocate<int>(n), *nl = allocate<int>(plane);
  double* spin = allocate<double>(3*n), *edges = allocate<double>(3*plane);
  float* cutoff = allocate<float>(types*types), *radial = allocate<float>(types*types*v.channels*(basis+1));
  float* p = allocate<float>(parameters), *moments = allocate<float>(v.channels*v.components*n);
  float* fp = allocate<float>(dim*n);
  double* reference = allocate<double>(15*n), *candidate = allocate<double>(15*n);
  for (int i = 0; i < types; ++i) active[i] = i == 0;
  for (int i = 0; i < n; ++i) {
    type[i] = i % types;
    nn[i] = i % (neighbors+1);
    for (int d = 0; d < 3; ++d) spin[d*n+i] = i % 5 ? 3*std::sin(.7*i+.3*d) : 0;
    for (int k = 0; k < neighbors; ++k) {
      nl[k*n+i] = (i+k+1) % n;
      for (int d = 0; d < 3; ++d) edges[d*plane+k*n+i] = .8+.07*k+.013*i+.2*d;
    }
  }
  for (int i = 0; i < types*types; ++i) cutoff[i] = 6;
  for (int i = 0; i < types*types*v.channels*(basis+1); ++i) radial[i] = .03*std::sin(.17*i);
  for (int i = 0; i < parameters; ++i) p[i] = .05*std::sin(.23*i);
  for (int i = 0; i < v.channels*v.components*n; ++i) moments[i] = .04*std::cos(.31*i);
  for (int i = 0; i < dim*n; ++i) fp[i] = .01*std::sin(.41*i);
  QUAD_CHECK(cudaMemset(reference, 0, 15*n*sizeof(double)));
  QUAD_CHECK(cudaMemset(candidate, 0, 15*n*sizeof(double)));
  const int blocks = (n*v.channels*Layout::tile+63)/64;
  nep_spin::forces<Layout, CenterVirial, false><<<blocks, 64>>>(v, n, 30, types, na, basis,
    Box{}, type, active, nullptr, spin, nn, nl, edges, plane, cutoff, radial,
    p, moments, fp, reference, reference+3*n, reference+6*n, !CenterVirial);
  QUAD_CHECK(cudaGetLastError());
  nep_spin::forces<Layout, CenterVirial, true, FuseChannels>
    <<<(n*(FuseChannels ? 1 : v.channels)*Layout::tile+63)/64, 64>>>(v, n, 30, types, na, basis,
    Box{}, type, active, nullptr, spin, nn, nl, edges, plane, cutoff, radial,
    p, moments, fp, candidate, candidate+3*n, candidate+6*n, !CenterVirial);
  QUAD_CHECK(cudaGetLastError());
  QUAD_CHECK(cudaDeviceSynchronize());
  double maximum = 0;
  for (int i = 0; i < 15*n; ++i) {
    const double error = std::fabs(candidate[i]-reference[i]);
    maximum = std::fmax(maximum, error);
    const double absolute = i >= 3*n && i < 6*n ? 5e-6 : 5e-5;
    if (!std::isfinite(candidate[i]) || error > absolute+1e-5*std::fabs(reference[i])) {
      std::fprintf(stderr, "FAIL tile=%d n=%d index=%d ref=%.17g got=%.17g\n",
        Layout::tile, n, i, reference[i], candidate[i]);
      std::exit(2);
    }
  }
  for (int i = 0; i < n; ++i)
    if (!active[type[i]])
      for (int d = 0; d < 3; ++d)
        if (candidate[(3+d)*n+i] != 0) std::exit(3);
  QUAD_CHECK(cudaFree(type)); QUAD_CHECK(cudaFree(active)); QUAD_CHECK(cudaFree(nn)); QUAD_CHECK(cudaFree(nl));
  QUAD_CHECK(cudaFree(spin)); QUAD_CHECK(cudaFree(edges)); QUAD_CHECK(cudaFree(cutoff)); QUAD_CHECK(cudaFree(radial));
  QUAD_CHECK(cudaFree(p)); QUAD_CHECK(cudaFree(moments)); QUAD_CHECK(cudaFree(fp));
  QUAD_CHECK(cudaFree(reference)); QUAD_CHECK(cudaFree(candidate));
  std::printf("PASS tile=%d components=%d center_virial=%d fused=%d n=%d max_abs=%.3e\n",
    Layout::tile, v.components, int(CenterVirial), int(FuseChannels), n, maximum);
}

int main() {
  for (int n : {1, 13, 37}) {
    check<nep_spin::MDLayoutSOC<2>>(n);
    check<nep_spin::MDLayoutNoSOC<2>>(n);
    check<nep_spin::MDLayoutSOC<4>>(n);
    check<nep_spin::MDLayoutNoSOC<4>>(n);
    check<nep_spin::MDLayoutSOC<2>, false>(n);
    check<nep_spin::MDLayoutNoSOC<2>, false>(n);
    check<nep_spin::MDLayoutSOC<4>, false>(n);
    check<nep_spin::MDLayoutNoSOC<4>, false>(n);
    check<nep_spin::MDLayoutSOC<2>, false, true>(n);
    check<nep_spin::MDLayoutNoSOC<2>, false, true>(n);
    check<nep_spin::MDLayoutSOC<4>, false, true>(n);
    check<nep_spin::MDLayoutNoSOC<4>, false, true>(n);
  }
}
