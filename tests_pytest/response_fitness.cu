// Fixed-parameter fitness probe for independent response.xyz integration.
#include "main_nep/fitness.cuh"
#include "main_nep/parameters.cuh"
#include "main_nep/snes_spin.cuh"
#include <cstdio>
#include <random>

class Probe : public Fitness {
public:
  using Fitness::Fitness;
  void points() {
    if (response_set.empty()) return;
    auto& d = response_set[0];
    d.mforce.copy_to_host(d.mforce_cpu.data());
    for (int n = 0; n < d.Nc; ++n) {
      const auto& s = d.structures[n];
      double prediction = 0, target = 0;
      for (int i = 0; i < s.num_atom; ++i) {
        int j = d.Na_sum_cpu[n] + i;
        prediction += d.mforce_cpu[j]*s.spin_tangent_x[i] +
          d.mforce_cpu[d.N+j]*s.spin_tangent_y[i] + d.mforce_cpu[2*d.N+j]*s.spin_tangent_z[i];
        target += s.mfx[i]*s.spin_tangent_x[i] + s.mfy[i]*s.spin_tangent_y[i] + s.mfz[i]*s.spin_tangent_z[i];
      }
      printf("POINT %s %.17g %.17g\n", s.spin_response_group.c_str(), prediction, target);
    }
  }
};

int main() {
  Parameters p;
  Probe f(p);
  std::mt19937 rng(9876);
  std::vector<float> mu(p.number_of_variables), sigma(mu.size());
  snes_spin::initialize_search(p, rng, mu, sigma);
  f.initialize_q_scaler(p);
  std::vector<float> scaler(p.dim);
  p.q_scaler_gpu[0].copy_to_host(scaler.data());
  printf("SCALER"); for (float x : scaler) printf(" %.9g", x); printf("\n");
  printf("BASELINE"); for (float x : p.spin_baseline) printf(" %.9g", x); printf("\n");
  std::vector<float> population;
  for (int i=0;i<p.population_size;++i) population.insert(population.end(),mu.begin(),mu.end());
  std::vector<float> e(p.population_size*(p.num_types+1)), force(e.size()), v(e.size()),
    q(e.size()), z(e.size()), m(e.size()), tau(e.size()), response(e.size());
  for (int generation=0;generation<2;++generation) {
    f.compute(generation,p,population.data(),e.data(),force.data(),v.data(),q.data(),z.data(),m.data(),tau.data(),response.data());
    for (int i=0;i<p.population_size;++i) {
      int j=i+p.num_types*p.population_size;
      printf("FIT %d %d %.9g %.9g %.9g %.9g %.9g %.9g\n",generation,i,e[j],force[j],v[j],m[j],tau[j],response[j]);
    }
  }
  f.points();
}
