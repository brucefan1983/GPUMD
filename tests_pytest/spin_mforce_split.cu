// Independent numerical oracle for the native Spin3 magnetic-loss reductions.
// Link against the NEP objects except main_nep/main.o; run with a valid nep.in.
#include "main_nep/nep_spin/spin_data.cuh"
#include "main_nep/parameter_input.cuh"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

template <class T> void upload(GPU_Vector<T>& target, const std::vector<T>& values)
{
  target.resize(values.size());
  target.copy_from_host(values.data());
}

int main()
{
  ParameterInput input;
  Spin_Parameters para(input);
  assert(para.num_types == 2 && para.spin_dof_type_active[0] && !para.spin_dof_type_active[1]);
  Dataset data;
  Spin_Data mag;
  // Unequal active counts, an inactive atom, an unlabeled frame, and an
  // inactive-only frame; all errors are nonzero so masking failures are visible.
  const std::vector<int> sizes{2, 3, 1, 1}, starts{0, 2, 5, 6};
  const std::vector<int> types{0, 1, 0, 0, 0, 0, 1};
  const std::vector<int> labeled{1, 1, 0, 1};
  const std::vector<std::array<double, 3>> axes{
    {0, 0, 2}, {1, 0, 0}, {0, 0, 0}, {1e-30, 0, 0}, {1, 2, 2}, {0, 0, 1}, {1, 0, 0}};
  const std::vector<std::array<float, 3>> errors{
    {3, 4, 5}, {70, 80, 90}, {0, 0, 6}, {7, 0, 0}, {2, -1, 3}, {60, 70, 80}, {80, 90, 100}};
  data.N = 7; data.Nc = 4;
  data.structures.resize(4); mag.labels.resize(4);
  data.weight_cpu = {2, .5f, 3, 4};
  data.error_cpu.resize(4); data.error_gpu.resize(4);
  data.has_type.assign(3 * 4, 0);
  std::vector<double> spins(21);
  std::vector<float> predicted(21), reference(21, .25f);
  for (int c = 0; c < 4; ++c) {
    auto& frame = data.structures[c];
    frame.num_atom = sizes[c];
    for (int a = starts[c]; a < starts[c] + sizes[c]; ++a) {
      frame.type.push_back(types[a]);
      data.has_type[types[a] * 4 + c] = data.has_type[2 * 4 + c] = 1;
      mag.labels[c].sx.push_back(axes[a][0]);
      mag.labels[c].sy.push_back(axes[a][1]);
      mag.labels[c].sz.push_back(axes[a][2]);
      for (int d = 0; d < 3; ++d) {
        spins[d * 7 + a] = axes[a][d];
        predicted[d * 7 + a] = reference[d * 7 + a] + errors[a][d];
      }
    }
  }
  upload(data.Na, sizes); upload(data.Na_sum, starts); upload(data.type, types);
  upload(mag.spin, spins); upload(mag.mforce, predicted); upload(mag.mforce_ref_gpu, reference);
  upload(mag.has_mforce_gpu, labeled); mag.has_mforce_cpu = labeled;
  upload(mag.spin_dof_type_active_gpu, para.spin_dof_type_active);
  for (bool weighted : {false, true}) {
    const auto torque = mag.get_rmse_tau(para, data, weighted, 0);
    const auto full = mag.get_rmse_mforce(para, data, weighted, 0);
    const auto parallel = mag.get_rmse_mforce_parallel(para, data, weighted, 0);
    const auto perp = mag.get_rmse_mforce_perp(para, data, weighted, 0);
    for (int type = 0; type <= 2; ++type) {
      double p = 0, t = 0, raw = 0, cross2 = 0;
      int count = 0, torque_count = 0;
      for (int c = 0; c < 4; ++c) {
        if (!labeled[c] || !data.has_type[type * 4 + c]) continue;
        const double w = weighted ? data.weight_cpu[c] * data.weight_cpu[c] : 1;
        for (int a = starts[c]; a < starts[c] + sizes[c]; ++a) {
          if (types[a] != 0) continue;
          const auto& e = errors[a]; const auto& s = axes[a];
          const double norm2 = e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
          const double length = std::hypot(std::hypot(s[0], s[1]), s[2]);
          const double dot = length == 0 ? 0 : (e[0]*s[0] + e[1]*s[1] + e[2]*s[2]) / length;
          const double projected = length == 0 ? norm2 / 3 : dot * dot;
          // Torque uses two transverse degrees of freedom per nonzero spin.
          if (length * length > 1e-20) {
            for (int d = 0; d < 3; ++d) {
              double cross = s[(d+1)%3]*e[(d+2)%3] - s[(d+2)%3]*e[(d+1)%3];
              cross2 += w * cross * cross;
            }
            ++torque_count;
          }
          p += w * projected; t += w * (norm2 - projected); raw += w * norm2;
          ++count;
        }
      }
      assert(count > 0);
      const double expected_torque = torque_count ? std::sqrt(cross2 / (2 * torque_count)) : 0;
      assert(std::abs(torque[type] - expected_torque) < 2e-6);
      assert(std::abs(parallel[type] - std::sqrt(p / count)) < 2e-6);
      assert(std::abs(perp[type] - std::sqrt(t / (2 * count))) < 2e-6);
      assert(std::abs(full[type] - std::sqrt(raw / (3 * count))) < 2e-6);
      assert(std::abs(3 * full[type]*full[type] - parallel[type]*parallel[type] - 2 * perp[type]*perp[type]) < 3e-5);
    }
  }
  // Simultaneously rotate spins and errors: the projected losses must be invariant.
  const auto p0 = mag.get_rmse_mforce_parallel(para, data, false, 0);
  const auto t0 = mag.get_rmse_mforce_perp(para, data, false, 0);
  for (int a = 0; a < 7; ++a) {
    spins[a] = -axes[a][1]; spins[7+a] = axes[a][0];
    predicted[a] = reference[a] - errors[a][1];
    predicted[7+a] = reference[7+a] + errors[a][0];
  }
  upload(mag.spin, spins); upload(mag.mforce, predicted);
  const auto p1 = mag.get_rmse_mforce_parallel(para, data, false, 0);
  const auto t1 = mag.get_rmse_mforce_perp(para, data, false, 0);
  for (int type = 0; type <= 2; ++type) {
    assert(std::abs(p0[type]-p1[type]) < 1e-6);
    assert(std::abs(t0[type]-t1[type]) < 1e-6);
  }
  // No labels / no active atoms must yield finite zeros, not 0/0.
  mag.has_mforce_cpu.assign(4, 0); upload(mag.has_mforce_gpu, mag.has_mforce_cpu);
  for (auto value : mag.get_rmse_mforce_parallel(para, data, false, 0)) assert(value == 0);
  for (auto value : mag.get_rmse_tau(para, data, false, 0)) assert(value == 0);
  mag.has_mforce_cpu = labeled; upload(mag.has_mforce_gpu, labeled);
  para.spin_dof_type_active.assign(2, 0); upload(mag.spin_dof_type_active_gpu, para.spin_dof_type_active);
  for (auto value : mag.get_rmse_mforce_perp(para, data, true, 0)) assert(value == 0);
  for (auto value : mag.get_rmse_tau(para, data, true, 0)) assert(value == 0);
  puts("spin magnetic split numerical oracle passed");
}
