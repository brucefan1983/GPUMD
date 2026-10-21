/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
DFT-DN dispersion Hessian for the analytic NEP Hessian command.

The hessian-only branch is purely additive; this module does not touch the
DFT-DN implementation in src/force/dftd3.cu, which is internal to that
translation unit.  Instead, when run.in enables `dftd3`, this header exposes
a helper that computes the DFT-DN contribution to the Cartesian Hessian by
central-differencing the analytic dispersion force that the existing
force.compute path produces.

The central-difference path is mathematically equivalent to the FD mode of
`compute_hessian fd`, but it is invoked from the analytic entry point so the
`compute_hessian analytic` keyword produces a hessian that includes the DFT-DN
contribution.  It is "more complex than the conservative degradation" (which
used to return false when dft3 was enabled) because we now run 3N * 2 force
evaluations on the GPU rather than bailing out.
------------------------------------------------------------------------------*/

#pragma once
#include <vector>

class Atom;
class Box;
class Force;
class Group;

namespace sparse_hessian { struct BlockPattern; }

namespace nep_hessian {

// Compute the DFT-DN dispersion Hessian contribution via central-differencing
// the analytic dispersion force.  Returns true on success and populates:
//   - hessian_out:    N3*N3 row-major dense Hessian (eV/A^2)
//   - force_out:      N3 baseline dispersion force (eV/A)
//   - pe_out:         baseline dispersion potential energy (eV)
//   - sparse_pattern: optional sparse block pattern derived from the NEP
//                     radial/angular neighbor lists so the sparse output
//                     layout matches the analytic path.
//
// The atom buffers are restored to the caller-supplied geometry on exit.
bool compute_nep_hessian_with_dftd3(
  Force& force,
  Box& box,
  Atom& atom,
  const std::vector<double>& position_soa,
  int N,
  std::vector<double>& hessian_out,
  std::vector<double>& force_out,
  double& pe_out,
  sparse_hessian::BlockPattern* sparse_pattern);

} // namespace nep_hessian
