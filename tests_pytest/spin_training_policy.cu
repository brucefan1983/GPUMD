// GPU-node interface checks for the host-side Spin3 training policies.
// Run in a directory containing a valid Spin3 nep.in (C1/C2/C3/C9).
#include "main_nep/nep_spin/spin_parameters.cuh"
#include "main_nep/parameter_input.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/nep_spin/spin_policy.cuh"
#include "main_nep/nep_spin/spin_init.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

int main()
{
  ParameterInput input;
  Spin_Parameters para(input);
  // Full force retains longitudinal errors and zero spins; torque retains neither.
  {
    Dataset loss_data;
    Spin_Data magnetic;
    magnetic.labels.resize(1);
    loss_data.N = 2;
    loss_data.Nc = 1;
    loss_data.structures.resize(1);
    auto& frame = loss_data.structures[0];
    frame.num_atom = 2;
    frame.type = {0, 0};
    magnetic.labels[0].sx = magnetic.labels[0].sy = {0.0, 0.0};
    magnetic.labels[0].sz = {2.0, 0.0};
    magnetic.has_mforce_cpu = {1};
    loss_data.weight_cpu = {1.0f};
    loss_data.has_type.assign(para.num_types + 1, 1);
    loss_data.error_cpu.resize(1);
    loss_data.error_gpu.resize(1);
    auto upload = [](auto& gpu, const auto& values) {
      gpu.resize(values.size());
      gpu.copy_from_host(values.data());
    };
    upload(loss_data.Na, std::vector<int>{2});
    upload(loss_data.Na_sum, std::vector<int>{0});
    upload(loss_data.type, std::vector<int>{0, 0});
    upload(magnetic.has_mforce_gpu, std::vector<int>{1});
    upload(magnetic.spin_dof_type_active_gpu, para.spin_dof_type_active);
    upload(magnetic.spin, std::vector<double>{0, 0, 0, 0, 2, 0});
    upload(magnetic.mforce_ref_gpu, std::vector<float>(6, 0));
    upload(magnetic.mforce, std::vector<float>{3, 0, 4, 0, 0, 6});
    assert(std::abs(magnetic.get_rmse_mforce(para, loss_data, false, 0).back() - std::sqrt(61.0 / 6)) < 1e-6);
    assert(std::abs(magnetic.get_rmse_tau(para, loss_data, false, 0).back() - std::sqrt(50.0)) < 1e-6);
    upload(magnetic.mforce, std::vector<float>{3, 0, 4, 0, 5, 6});
    assert(std::abs(magnetic.get_rmse_mforce(para, loss_data, false, 0).back() - std::sqrt(86.0 / 6)) < 1e-6);
    assert(std::abs(magnetic.get_rmse_tau(para, loss_data, false, 0).back() - std::sqrt(50.0)) < 1e-6);
  }
  std::mt19937 rng(12345678), same_rng(12345678);
  std::vector<float> mu(para.number_of_variables), sigma(mu.size());
  auto same_mu = mu;
  auto same_sigma = sigma;
  snes_spin::initialize_search(para, rng, mu, sigma);
  snes_spin::initialize_search(para, same_rng, same_mu, same_sigma);
  assert(mu == same_mu && sigma == same_sigma && rng == same_rng);
  for (std::size_t i = 0; i < mu.size(); ++i) {
    assert(std::isfinite(mu[i]) && std::isfinite(sigma[i]) && sigma[i] > 0);
  }
  const int spin_offset = para.number_of_variables_ann +
    para.number_of_variables_descriptor - para.number_of_variables_descriptor_spin -
    para.number_of_variables_spin_projection;
  for (int i = 0; i < para.number_of_variables_descriptor_spin; ++i) {
    assert(sigma[spin_offset + i] == para.sigma0 * 0.1f);
  }
  std::vector<int> types(mu.size(), para.num_types);
  snes_spin::assign_variable_types(para, spin_offset, types);
  for (int i = 0; i < para.number_of_variables_descriptor_spin; ++i) {
    assert(types[spin_offset + i] == (i % (para.num_types * para.num_types)) / para.num_types);
  }
  assert(std::all_of(types.begin(), types.begin() + spin_offset,
    [&](int type) { return type == para.num_types; }));

  std::vector<Structure> structures(2);
  for (int t = 0; t < 2; ++t) {
    structures[t].num_atom = 1;
    structures[t].type = {t};
    structures[t].energy = -2.0f - t;
  }
  fitness_spin::fit_spin_energy_baseline(structures, para);
  assert(para.spin_baseline[0] == -2.0f && para.spin_baseline[1] == -3.0f);

  Dataset dataset;
  Spin_Data magnetic;
  magnetic.labels.resize(3);
  dataset.Nc = dataset.N = 3;
  dataset.Na_sum_cpu = {0, 1, 2};
  dataset.structures.resize(3);
  magnetic.mforce_cpu.resize(9);
  magnetic.mforce.resize(9);
  std::vector<float> mf(9, 0.0f);
  for (int i = 0; i < 3; ++i) {
    dataset.structures[i].num_atom = 1;
    auto& frame = magnetic.labels[i];
    frame.has_spin_response = 1;
    frame.spin_response_group = "rotation";
    frame.spin_response_coordinate = i - 1;
    frame.spin_tangent_x = {1};
    frame.spin_tangent_y = frame.spin_tangent_z = {0};
    frame.mfx = {float(i - 1)};
    frame.mfy = frame.mfz = {0};
    mf[i] = i - 1;
  }
  magnetic.mforce.copy_from_host(mf.data());
  fitness_spin::ResponseLoss exact;
  assert(exact.value() == 0);
  exact.append(0, dataset, magnetic);
  assert(exact.value() == 0);
  for (int i = 0; i < 3; ++i) mf[i] += 1;
  magnetic.mforce.copy_from_host(mf.data());
  fitness_spin::ResponseLoss shifted;
  shifted.append(0, dataset, magnetic);
  const double expected = 0.25 * (1.0 / std::sqrt(2.0 / 3.0) - 0.5);
  assert(std::abs(shifted.value() - expected) < 1.0e-7);

  // Adding a much stronger, perfectly reproduced group must not hide the
  // existing weak group's error; equal groups imply exactly half its loss.
  for (int i = 0; i < 3; ++i) {
    auto& frame = magnetic.labels[i];
    frame.spin_response_group = "strong";
    frame.mfx[0] = 100.0f * (i - 1);
    mf[i] = frame.mfx[0];
  }
  magnetic.mforce.copy_from_host(mf.data());
  shifted.append(0, dataset, magnetic);
  assert(std::abs(shifted.value() - 0.5 * expected) < 1.0e-7);

  // A genuinely zero-response reference uses the documented finite floor.
  fitness_spin::ResponseLoss zero;
  for (int i = 0; i < 3; ++i) {
    magnetic.labels[i].mfx[0] = 0.0f;
    mf[i] = 0.0f;
  }
  magnetic.mforce.copy_from_host(mf.data());
  zero.append(0, dataset, magnetic);
  assert(zero.value() == 0.0f);
  puts("spin training policy interface checks passed");
}
