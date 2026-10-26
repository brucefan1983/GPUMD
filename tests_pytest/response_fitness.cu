// Fixed-parameter model loss probe, independent of the SNES lifecycle.
#include "spin_test_access.cuh"
#include <cstdio>
int main() {
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
  printf("SCALER"); for (float x : scaler) printf(" %.9g", x); printf("\n");
  printf("BASELINE"); for (double x : p.energy_reference) printf(" %.17g", x); printf("\n");
  std::vector<float> population;
  for (int i=0;i<p.population_size;++i) population.insert(population.end(),mu.begin(),mu.end());
  for (int iteration=0;iteration<2;++iteration) {
    model.evaluate(iteration % model.training->num_batches, population.data());
    for (int i=0;i<p.population_size;++i) {
      int j=i+p.num_types*p.population_size;
      printf("FIT %d %d %.9g %.9g %.9g %.9g %.9g %.9g\n",iteration,i,
        model.energy_loss.fitness_energy[j],model.energy_loss.fitness_force[j],model.energy_loss.fitness_virial[j],
        model.fitness_mforce[j],model.fitness_tau[j],model.fitness_response[j]);
    }
  }
  if (model.response_set.empty()) return 0;
  auto& d = model.response_set[0]; auto& magnetic = *model.response_data[0];
  magnetic.mforce.copy_to_host(magnetic.mforce_cpu.data());
  for (int n=0;n<d.Nc;++n) {
    const auto& s=magnetic.labels[n]; double prediction=0,target=0;
    for (int i=0;i<d.Na_cpu[n];++i) {
      int j=d.Na_sum_cpu[n]+i;
      prediction+=magnetic.mforce_cpu[j]*s.spin_tangent_x[i]+magnetic.mforce_cpu[d.N+j]*s.spin_tangent_y[i]+magnetic.mforce_cpu[2*d.N+j]*s.spin_tangent_z[i];
      target+=s.mfx[i]*s.spin_tangent_x[i]+s.mfy[i]*s.spin_tangent_y[i]+s.mfz[i]*s.spin_tangent_z[i];
    }
    printf("POINT %s %.17g %.17g\n",s.spin_response_group.c_str(),prediction,target);
  }
}
