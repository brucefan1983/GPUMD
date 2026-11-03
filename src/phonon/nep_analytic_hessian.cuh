/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
Analytic Hessian for NEP potentials.

Computes d^2E / dx_i dx_j analytically from the NEP descriptor chain:
  E = ANN(q),  q = f(positions)

The Hessian has two terms (chain rule):
  H_ij = sum_d,d' (d^2E/dq_d dq_d') * (dq_d/dx_i) * (dq_d'/dx_j)
       + sum_d     (dE/dq_d)         * (d^2q_d / dx_i dx_j)

All position, force, and Cartesian Hessian indices use GPUMD's SoA ordering:
axis * N + atom.  Atom-major ordering is an export-layer concern only.

The current implementation covers radial, angular, radial-angular ANN cross,
and NEP ZBL terms for model classes accepted by the entry point. Unsupported
models and periodic small-box systems return false so that the caller can
report an explicit error. Angular derivatives are prepared once per
center-neighbor-n tuple and reused by staged pair and radial-angular kernels.
Unsupported model classes return false.

This module is analytic only: there is no finite-difference fallback here.
Configurations that enable DFT-D3 dispersion are rejected explicitly.
------------------------------------------------------------------------------*/

#pragma once
#include "utilities/gpu_vector.cuh"
#include "sparse_hessian.cuh"
#include <vector>

class Force;
class Box;
class Atom;
class Group;
class NEP_Hessian_Model;

bool compute_nep_analytic_hessian(
  Force& force,
  NEP_Hessian_Model& model,
  Box& box,
  Atom& atom,
  const std::vector<double>& position_soa, // 3 * N, component * N + atom
  int N,
  std::vector<double>& hessian_soa,        // 3N x 3N, row-major SoA
  std::vector<double>& force_soa,          // 3N, component * N + atom
  double& pe_out,
  std::vector<Group>* groups = nullptr,
  sparse_hessian::BlockPattern* sparse_pattern = nullptr);
