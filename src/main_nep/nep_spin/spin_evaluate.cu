/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "main_nep/dataset.cuh"
#include "model/box.cuh"
#include "nep_spin.cuh"
#include "spin_parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>

#include "spin_training_kernels.cuh"
#include "force/nep_spin/spin_kernels.cuh"
#include "spin_compile.cuh"

void NEP_Spin_Model::initialize_spin(Spin_Parameters& para, int N, int device_count)
{
  spin_compress_ = para.spin_compress;
  struct_dim_ = para.dim_struct;
  spin_projection_offset_ = para.number_of_variables_descriptor - para.number_of_variables_spin_projection;
  num_types_ = para.num_types;

  for (int device_id = 0; device_id < device_count; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    auto& data = spin_data_[device_id];
    data.spin_type_active.resize(num_types_);
    data.spin_type_active.copy_from_host(para.spin_type_active.data());
    std::vector<float> spin_cutoff_pair(static_cast<std::size_t>(num_types_) * num_types_);
    for (int type_i = 0; type_i < num_types_; ++type_i) {
      for (int type_j = 0; type_j < num_types_; ++type_j) {
        spin_cutoff_pair[type_i * num_types_ + type_j] =
          0.5f * (para.spin_cutoff_by_type[type_i] + para.spin_cutoff_by_type[type_j]);
      }
    }
    data.spin_cutoff_pair.resize(spin_cutoff_pair.size());
    data.spin_cutoff_pair.copy_from_host(spin_cutoff_pair.data(), spin_cutoff_pair.size());
    data.spin_layout.initialize(para.spin_compress, para.spin_l_max, para.spin_order, para.spin_soc);
    if (nep_spin::needs_wide_forward(para.spin_soc, para.spin_l_max, para.spin_order)) {
      data.wide_spin_descriptors.resize(size_t(N)*para.dim_spin);
      data.wide_radial.resize(size_t(num_types_)*num_types_*spin_compress_*(para.basis_size_angular+1));
    }
    data.spin_moments.resize(size_t(N) * data.spin_layout.view.channels * data.spin_layout.view.components);

    data.force.resize(3 * N);
    data.mforce.resize(3 * N);
    data.virial.resize(9 * N);
  }
}

void NEP_Spin_Model::find_additional_descriptors(Spin_Parameters& para, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  const int plane_size = dataset.N * magnetic_data.max_NN_spin;
  const float* projection = annmb[device_id].c + spin_projection_offset_;
  const int count = num_types_ * num_types_ * spin_compress_ * (para.basis_size_angular + 1);
  const size_t edge_count = spin_descriptor_jit_ && para.spin_l_max == 2 && para.spin_order == 2 ?
    size_t(2) * spin_compress_ * plane_size : 0;
  auto contract = [&](auto* radial) {
    nep_spin::contract_radial<<<(count + 127) / 128, 128>>>(
      count, spin_compress_, para.n_max_angular, para.basis_size_angular,
      annmb[device_id].c + paramb.num_c_radial, projection, radial);
    GPU_CHECK_KERNEL
  };
  if (data.wide_radial.size()) {
    contract(data.wide_radial.data());
  } else {
    if (data.radial_coefficients.size() < size_t(count) + edge_count)
      data.radial_coefficients.resize(size_t(count) + edge_count);
    contract(data.radial_coefficients.data());
  }
  if (!spin_descriptor_jit_) {
    auto launch = [&](auto layout, auto* radial, auto* output, int offset) {
      nep_spin::descriptors<<<(dataset.N * spin_compress_ + 63) / 64, 64>>>(
        layout, dataset.N, offset, num_types_, para.n_max_angular,
        para.basis_size_angular, Box{}, dataset.type.data(), data.spin_type_active.data(),
        magnetic_data.r_spin.data(), magnetic_data.spin.data(), magnetic_data.NN_spin.data(),
        magnetic_data.NL_spin.data(), magnetic_data.r12_spin.data(), plane_size,
        data.spin_cutoff_pair.data(), radial,
        projection, data.spin_moments.data(), output);
    };
    float* q = nep_data[device_id].descriptors.data();
    if (data.wide_spin_descriptors.size())
      launch(nep_spin::WideLayout(data.spin_layout.view), data.wide_radial.data(), data.wide_spin_descriptors.data(), 0);
    else
      nep_spin::launch_runtime_layout(data.spin_layout.view, [&](auto layout) {
        launch(layout, data.radial_coefficients.data(), q, struct_dim_);
      });
    GPU_CHECK_KERNEL
    return;
  }

  spin_descriptor_jit_(
    dataset.N,
    struct_dim_,
    data.spin_cutoff_pair.data(),
    dataset.type.data(),
    data.spin_type_active.data(),
    magnetic_data.r_spin.data(),
    magnetic_data.spin.data(),
    magnetic_data.NN_spin.data(),
    magnetic_data.NL_spin.data(),
    magnetic_data.r12_spin.data(),
    plane_size,
    data.radial_coefficients.size() ? data.radial_coefficients.data() : nullptr,
    projection,
    data.spin_moments.data(),
    nep_data[device_id].descriptors.data(),
    data.wide_spin_descriptors.size() ? data.wide_spin_descriptors.data() : nullptr,
    data.wide_radial.size() ? data.wide_radial.data() : nullptr);

}

void NEP_Spin_Model::initialize_additional_outputs(Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  data.force.fill(0.0);
  data.mforce.fill(0.0);
  data.virial.fill(0.0);
  magnetic_data.mforce.fill(0.0f);
}

void NEP_Spin_Model::find_additional_force(Spin_Parameters& para, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  const int block_size = 128;
  const int grid_size = (dataset.N + block_size - 1) / block_size;
  const int plane_size = dataset.N * magnetic_data.max_NN_spin;
  const float* projection = annmb[device_id].c + spin_projection_offset_;
  if (!spin_force_jit_) {
    auto launch = [&](auto layout, auto* radial) {
      nep_spin::forces<<<(dataset.N * spin_compress_ + 63) / 64, 64>>>(
        layout, dataset.N, struct_dim_, num_types_, para.n_max_angular,
        para.basis_size_angular, Box{}, dataset.type.data(), data.spin_type_active.data(),
        magnetic_data.r_spin.data(), magnetic_data.spin.data(), magnetic_data.NN_spin.data(),
        magnetic_data.NL_spin.data(), magnetic_data.r12_spin.data(), plane_size,
        data.spin_cutoff_pair.data(), radial,
        projection, data.spin_moments.data(), nep_data[device_id].Fp.data(), data.force.data(),
        data.mforce.data(), data.virial.data(), true);
    };
    if (data.wide_radial.size())
      launch(nep_spin::WideLayout(data.spin_layout.view), data.wide_radial.data());
    else
      nep_spin::launch_runtime_layout(data.spin_layout.view, [&](auto layout) {
        launch(layout, data.radial_coefficients.data());
      });
    add_spin_training_outputs<<<grid_size, block_size>>>(dataset.N, data.force.data(),
      data.mforce.data(), data.virial.data(), dataset.force.data(),
      magnetic_data.mforce.data(), dataset.virial.data());
    GPU_CHECK_KERNEL
    return;
  }

  spin_force_jit_(
    dataset.N,
    struct_dim_,
    data.spin_cutoff_pair.data(),
    dataset.type.data(),
    data.spin_type_active.data(),
    magnetic_data.r_spin.data(),
    magnetic_data.spin.data(),
    magnetic_data.NN_spin.data(),
    magnetic_data.NL_spin.data(),
    magnetic_data.r12_spin.data(),
    plane_size,
    nep_data[device_id].Fp.data(),
    data.radial_coefficients.size() ? data.radial_coefficients.data() : nullptr,
    projection,
    data.spin_moments.data(),
    data.force.data(),
    data.mforce.data(),
    data.virial.data(),
    data.wide_radial.size() ? data.wide_radial.data() : nullptr);

  add_spin_training_outputs<<<grid_size, block_size>>>(
    dataset.N,
    data.force.data(),
    data.mforce.data(),
    data.virial.data(),
    dataset.force.data(),
    magnetic_data.mforce.data(),
    dataset.virial.data());
  GPU_CHECK_KERNEL
}

void NEP_Spin_Model::initialize_spin_compiled(const Spin_Parameters& para)
{
  spin_descriptor_jit_ = nullptr;
  spin_force_jit_ = nullptr;
  compiled_spin_.reset();
  if (!para.nep_compile) return;
  const nep_spin::Layout layout(para.spin_compress, para.spin_l_max, para.spin_order, para.spin_soc);
  auto config = make_nep_compile_config(para);
  config.model_name = "Spin";
  config.source_file = "main_nep/nep_spin/spin_specialized.cu";
  config.extra_config = nep_spin::compile_layout(layout, para.spin_soc);
  if (nep_spin::needs_wide_forward(para.spin_soc, layout.lmax, layout.order))
    config.extra_config += "#define NEP_SPIN_WIDE_FORWARD\n";
  compiled_spin_.reset(new NEP_Compile(config));
  if (
    !compiled_spin_->is_valid() ||
    !compiled_spin_->load_symbol("nep_spin_launch_descriptor_v2", spin_descriptor_jit_) ||
    !compiled_spin_->load_symbol("nep_spin_launch_force_v2", spin_force_jit_)) {
    spin_descriptor_jit_ = nullptr;
    spin_force_jit_ = nullptr;
    compiled_spin_.reset();
  } else {
    printf(
      "Spin aggregate JIT ready: C=%d L=%d order=%d SOC=%d.\n",
      para.spin_compress,
      para.spin_l_max,
      para.spin_order,
      para.spin_soc);
  }
}
