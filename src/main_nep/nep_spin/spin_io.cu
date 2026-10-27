/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "main_nep/common/nep_io.cuh"
#include "nep_spin.cuh"
#include "spin_io.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace
{
void validate_model(Spin_Parameters& para, const NEP_Spin_Model_File& model)
{
  const int l_max[] = {
    para.L_max,
    para.has_q_222 ? 2 : 0,
    para.has_q_1111,
    para.has_q_112,
    para.has_q_123,
    para.has_q_233,
    para.has_q_134};
  if (
    model.elements != para.elements || model.n_max[0] != para.n_max_radial ||
    model.n_max[1] != para.n_max_angular || model.basis_size[0] != para.basis_size_radial ||
    model.basis_size[1] != para.basis_size_angular ||
    !std::equal(model.l_max, model.l_max + 7, l_max) || model.num_neurons != para.num_neurons1 ||
    model.spin_basis_size != para.spin_basis_size[0] || model.spin_l_max != para.spin_l_max[0] ||
    model.spin_compress != para.spin_compress || model.spin_order != para.spin_order ||
    model.spin_soc != para.spin_soc ||
    model.spin_projection_size != para.number_of_variables_spin_projection ||
    model.dim != para.dim || model.parameters.size() != size_t(para.number_of_variables)) {
    PRINT_INPUT_ERROR("nep.in is inconsistent with the Spin model in nep.txt.");
  }
  const auto same = [](float a, float b) {
    return std::abs(a - b) <= 1.0e-5f * (std::abs(a) + std::abs(b) + 1.0e-10f);
  };
  if (!same(model.rc_radial, para.rc_radial[0]) || !same(model.rc_angular, para.rc_angular[0])) {
    PRINT_INPUT_ERROR("Structural cutoffs in nep.txt do not match nep.in.");
  }
  for (int t = 0; t < para.num_types; ++t) {
    if (
      std::abs(model.spin_cutoff[t] - para.spin_cutoff_by_type[t]) >
      1.0e-6f * std::max(1.0f, model.spin_cutoff[t])) {
      PRINT_INPUT_ERROR("spin_cutoff in nep.txt does not match nep.in.");
    }
  }
  if (
    model.enable_zbl != para.enable_zbl ||
    (model.enable_zbl &&
     (model.use_typewise_cutoff_zbl != para.use_typewise_cutoff_zbl ||
      std::abs(model.zbl_rc_inner - para.zbl_rc_inner) > 1.0e-6f ||
      std::abs(model.zbl_rc_outer - para.zbl_rc_outer) > 1.0e-6f ||
      (model.use_typewise_cutoff_zbl &&
       std::abs(model.typewise_cutoff_zbl_factor - para.typewise_cutoff_zbl_factor) > 1.0e-6f)))) {
    PRINT_INPUT_ERROR("Spin ZBL checkpoint does not match nep.in.");
  }
  if (para.is_spin_type_set && para.spin_type_active != model.spin_type) {
    PRINT_INPUT_ERROR("spin_type in nep.in conflicts with the Spin checkpoint.");
  }
}

void write_checkpoint_metadata(FILE* fid_nep, const Spin_Parameters& para)
{
  // Serialize effective model state, including masks inherited from a checkpoint.
  const bool write_dof =
    std::any_of(para.spin_type_active.begin(), para.spin_type_active.end(), [](int active) {
      return active != 1;
    });
  fprintf(fid_nep, "spin_version 1\n");
  fprintf(fid_nep, "spin_basis_size %d\n", para.spin_basis_size[0]);
  fprintf(fid_nep, "spin_l_max %d\n", para.spin_l_max[0]);
  fprintf(fid_nep, "spin_compress %d\n", para.spin_compress);
  fprintf(fid_nep, "spin_cutoff");
  const bool uniform_spin_cutoff = std::all_of(
    para.spin_cutoff_by_type.begin() + 1, para.spin_cutoff_by_type.end(), [&](float value) {
      return std::abs(value - para.spin_cutoff_by_type.front()) <= 1.0e-7f;
    });
  const int cutoff_count = uniform_spin_cutoff ? 1 : para.num_types;
  for (int type = 0; type < cutoff_count; ++type) {
    fprintf(fid_nep, " %.9g", para.spin_cutoff_by_type[type]);
  }
  fprintf(fid_nep, "\n");
  fprintf(fid_nep, "spin_order %d\n", para.spin_order);
  fprintf(fid_nep, "spin_soc %d\n", para.spin_soc);
  fprintf(fid_nep, "spin_projection_size %d\n", para.number_of_variables_spin_projection);
  fprintf(fid_nep, "spin_scaler 1\n");
  if (write_dof) {
    fprintf(fid_nep, "spin_type");
    for (int type = 0; type < para.num_types; ++type) {
      if (para.spin_type_active[type]) {
        fprintf(fid_nep, " %s", para.elements[type].c_str());
      }
    }
    fprintf(fid_nep, "\n");
  }
}
} // namespace

void read_spin_model(Spin_Parameters& para, float* parameters)
{
  std::ifstream input("nep.txt");
  NEP_Spin_Model_File model;
  std::string error;
  if (!model.read(input, error))
    PRINT_INPUT_ERROR(error.c_str());
  validate_model(para, model);
  para.energy_reference = model.energy_reference;
  para.spin_type_active = model.spin_type;
  std::copy(model.parameters.begin(), model.parameters.end(), parameters);
  para.q_scaler_cpu = model.q_scaler;
  int devices;
  CHECK(gpuGetDeviceCount(&devices));
  for (int d = 0; d < devices; ++d) {
    CHECK(gpuSetDevice(d));
    para.q_scaler_gpu[d].copy_from_host(para.q_scaler_cpu.data());
  }
  CHECK(gpuSetDevice(0));
}

void Spin_Parameters::initialize_spin_checkpoint()
{
  auto& para = *this;
  para.spin_restart = !para.prediction && std::ifstream("nep.restart").good();
  para.q_scaler_cpu.assign(para.dim, 1.0e10f);
  int devices;
  CHECK(gpuGetDeviceCount(&devices));
  if (devices > 16)
    PRINT_INPUT_ERROR("At most 16 training devices are supported.");
  for (int d = 0; d < devices; ++d) {
    CHECK(gpuSetDevice(d));
    para.q_scaler_gpu[d].resize(para.dim);
    para.q_scaler_gpu[d].copy_from_host(para.q_scaler_cpu.data());
    para.q_scaler_max[d].resize(para.dim);
    para.q_scaler_max[d].fill(-1.0e10f);
    para.q_scaler_min[d].resize(para.dim);
    para.q_scaler_min[d].fill(1.0e10f);
    para.q_scaler_square_sum[d].resize(para.dim);
    para.q_scaler_square_sum[d].fill(0.0f);
    para.q_scaler_count[d].resize(para.dim);
    para.q_scaler_count[d].fill(0);
  }
  CHECK(gpuSetDevice(0));
  if (para.prediction || para.spin_restart || para.import_q_scaler) {
    std::vector<float> parameters(para.number_of_variables);
    read_spin_model(para, parameters.data());
  }
  if (para.spin_restart) {
    std::vector<float> mu, sigma;
    read_nep_restart("nep.restart", para, mu, sigma, &para.energy_reference);
  } else if (!para.prediction) {
    // import_q_scaler imports only the scaler; new training fits its own reference.
    para.energy_reference.clear();
  }
}

void NEP_Spin_Model::write_spin_model(
  FILE* fid_nep, const float* elite, int max_NN_radial, int max_NN_angular)
{
  auto& para = *model_parameters;
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  write_nep_header(fid_nep, para, max_NN_radial, max_NN_angular);
  write_checkpoint_metadata(fid_nep, para);
  write_nep_model_parameters(fid_nep, para, elite);
}
