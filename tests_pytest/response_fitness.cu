// Fixed-parameter model loss probe, independent of the SNES lifecycle.
#include "main_nep/nep_spin/spin_policy.cuh"
#include "spin_test_access.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
static void check_response_loss()
{
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
  for (int i = 0; i < 3; ++i)
    mf[i] += 1;
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
}

int main()
{
  check_response_loss();
  ParameterInput input;
  NEP_Spin_Model model(input);
  auto& p = *model.model_parameters;
  model.prepare_data();
  model.prepare_training();
  std::mt19937 rng(9876);
  std::vector<float> mu(p.number_of_variables), sigma(mu.size());
  snes_spin::initialize_search(p, rng, mu, sigma);
  std::vector<float> scaler(p.dim);
  p.q_scaler_gpu[0].copy_to_host(scaler.data());
  printf("SCALER");
  for (float x : scaler)
    printf(" %.9g", x);
  printf("\n");
  printf("BASELINE");
  for (double x : p.energy_reference)
    printf(" %.17g", x);
  printf("\n");
  std::vector<float> population;
  for (int i = 0; i < p.population_size; ++i)
    population.insert(population.end(), mu.begin(), mu.end());
  for (int iteration = 0; iteration < 2; ++iteration) {
    model.evaluate(iteration % model.training->num_batches, population.data());
    for (int i = 0; i < p.population_size; ++i) {
      int j = i + p.num_types * p.population_size;
      printf(
        "FIT %d %d %.9g %.9g %.9g %.9g %.9g %.9g\n",
        iteration,
        i,
        model.energy_loss.fitness_energy[j],
        model.energy_loss.fitness_force[j],
        model.energy_loss.fitness_virial[j],
        model.fitness_mforce[j],
        model.fitness_tau[j],
        model.fitness_response[j]);
    }
  }
  if (model.response_set.empty())
    return 0;
  auto& d = model.response_set[0];
  auto& magnetic = *model.response_data[0];
  magnetic.mforce.copy_to_host(magnetic.mforce_cpu.data());
  for (int n = 0; n < d.Nc; ++n) {
    const auto& s = magnetic.labels[n];
    double prediction = 0, target = 0;
    for (int i = 0; i < d.Na_cpu[n]; ++i) {
      int j = d.Na_sum_cpu[n] + i;
      prediction += magnetic.mforce_cpu[j] * s.spin_tangent_x[i] +
                    magnetic.mforce_cpu[d.N + j] * s.spin_tangent_y[i] +
                    magnetic.mforce_cpu[2 * d.N + j] * s.spin_tangent_z[i];
      target += s.mfx[i] * s.spin_tangent_x[i] + s.mfy[i] * s.spin_tangent_y[i] +
                s.mfz[i] * s.spin_tangent_z[i];
    }
    printf("POINT %s %.17g %.17g\n", s.spin_response_group.c_str(), prediction, target);
  }
}
