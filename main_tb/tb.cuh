/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "utilities/gpu_vector.cuh"

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

class Parameters;
class Dataset;
struct Neighbor;

class TB
{
public:
  struct Result {
    int num_orbitals = 0;
    int num_kpoints = 0;
    std::vector<float> descriptors;
    std::vector<float> onsite;
    std::vector<float> hopping;
    std::vector<double> hamiltonian_real;
    std::vector<double> hamiltonian_imag;
    std::vector<float> eigenvalues;
    std::vector<float> eigenvectors_real;
    std::vector<float> eigenvectors_imag;
  };

  TB(const Parameters& para, const Dataset& dataset);
  ~TB();

  static void initialize_parameters(const Parameters& para, std::vector<double>& parameters);
  void forward(const double* parameters, Result& result);
  void evaluate_bands(
    const double* parameters,
    const std::vector<int>& k_indices,
    std::vector<float>& eigenvalues,
    bool with_vectors = false);
  void evaluate_all_bands(const double* parameters, std::vector<float>& eigenvalues);
  void backward(
    const std::vector<int>& k_indices,
    const std::vector<double>& eigenvalue_gradients,
    std::vector<float>& grad_parameters);

  int num_orbitals() const { return num_orbitals_; }
  void get_q_scaler(std::vector<float>& scaler);
  void get_reference_descriptor_max_abs(std::vector<float>& max_abs) const;
  void set_q_scaler(const std::vector<float>& scaler);
  void write_model_file(const std::string& filename, const double* parameters);
  void read_model_file(
    const std::string& filename, std::vector<double>& parameters, std::vector<float>& scaler) const;

private:
  const Parameters& para_;
  const Dataset& dataset_;
  int num_atoms_ = 0;
  int num_bonds_ = 0;
  int num_kpoints_ = 0;
  int k_batch_size_ = 0;
  int orbitals_per_atom_ = 0;
  int num_orbitals_ = 0;
  struct Eigensolver;
  std::unique_ptr<Eigensolver> eigensolver_;

  GPU_Vector<int> atom_types_;
  struct NeighborList {
    GPU_Vector<int> nn_sum_, nn_, nl_;
    GPU_Vector<float> dx_, dy_, dz_;
    void initialize(int num_atoms, const std::vector<Neighbor>& neighbors);
  };
  NeighborList radial_, angular_;
  GPU_Vector<int> bond_i_;
  GPU_Vector<int> bond_j_;
  GPU_Vector<float> bond_r_;
  GPU_Vector<float> descriptors_;
  GPU_Vector<float> q_scaler_;
  GPU_Vector<float> parameters_;
  std::vector<float> parameters_host_;
  GPU_Vector<float> onsite_;
  GPU_Vector<float> hopping_;
  GPU_Vector<int> hopping_pair_offsets_;
  GPU_Vector<float> grad_descriptors_;
  GPU_Vector<float> grad_parameters_;
  GPU_Vector<float> grad_onsite_;
  GPU_Vector<float> grad_hopping_;
  std::vector<float> reference_descriptor_max_abs_;
  std::vector<float> onsite_host_;
  std::vector<float> hopping_host_;
  std::vector<float> eigenvectors_real_;
  std::vector<float> eigenvectors_imag_;
  std::vector<double> sk_derivative_blocks_;
  std::vector<int> active_hopping_outputs_;

  void write_model_header(std::ostream& output) const;
  void copy_neighbor_lists_to_gpu();
  void initialize_q_scaler();
  void compute_descriptors(const float* descriptor_parameters);
  void compute_model(const double* parameters);
  void precompute_sk_derivative_blocks();
  void build_hamiltonians(
    const std::vector<int>& k_indices, std::vector<double>& hr, std::vector<double>& hi) const;
  void solve_bands(
    const std::vector<int>& k_indices, std::vector<float>& eigenvalues, bool with_vectors);
  void solve_all_bands(std::vector<float>& eigenvalues, Result* result = nullptr);
};
