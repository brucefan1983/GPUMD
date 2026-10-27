// Inspect actual SNES restore before sampling and dump fixed-state predictions.
#include "main_nep/fitness.cuh"
#include "main_nep/snes.cuh"
#include "spin_test_access.cuh"
#include <cassert>
#include <cmath>
static void check_initialization(Parameters&) {}

static void check_initialization(Spin_Parameters& para)
{
  std::mt19937 rng(12345678), same_rng(12345678);
  std::vector<float> mu(para.number_of_variables), sigma(mu.size());
  auto same_mu = mu;
  auto same_sigma = sigma;
  snes_spin::initialize_search(para, rng, mu, sigma);
  snes_spin::initialize_search(para, same_rng, same_mu, same_sigma);
  assert(mu == same_mu && sigma == same_sigma && rng == same_rng);
  // initial_para controls the scaler reference, never the SNES distribution.
  const float saved_initial_para = para.initial_para;
  para.initial_para = 0.5f;
  same_rng.seed(12345678);
  snes_spin::initialize_search(para, same_rng, same_mu, same_sigma);
  assert(mu == same_mu && sigma == same_sigma && rng == same_rng);
  para.initial_para = saved_initial_para;
  for (std::size_t i = 0; i < mu.size(); ++i) {
    assert(std::isfinite(mu[i]) && std::isfinite(sigma[i]) && sigma[i] > 0);
  }
  const int spin_offset = para.number_of_variables_ann + para.number_of_variables_descriptor -
                          para.number_of_variables_descriptor_spin -
                          para.number_of_variables_spin_projection;
  for (int i = 0; i < para.number_of_variables_descriptor_spin; ++i) {
    assert(sigma[spin_offset + i] == para.sigma0 * 0.1f);
  }
  std::vector<int> types(mu.size(), para.num_types);
  snes_spin::assign_variable_types(para, spin_offset, types);
  for (int i = 0; i < para.number_of_variables_descriptor_spin; ++i) {
    assert(types[spin_offset + i] == (i % (para.num_types * para.num_types)) / para.num_types);
  }
  assert(std::all_of(types.begin(), types.begin() + spin_offset, [&](int type) {
    return type == para.num_types;
  }));
}

class SearchProbe : public SNES
{
public:
  using SNES::SNES;
  const std::vector<float>& center() const { return mu; }
  void dump(Parameters& p) { output_mu_and_sigma(p, "restored.restart"); }
};
template <class Dump>
void dump_magnetic(NEP&, Dataset&, Dump&)
{
}
template <class Dump>
void dump_magnetic(NEP_Spin_Model& model, Dataset& data, Dump& dump)
{
  dump("MF", model.magnetic(data).mforce, 3 * data.N);
}
template <class Model>
void evaluate(Model& model, const std::vector<float>& parameters, const char* filename)
{
  auto& p = *model.model_parameters;
  auto& training = *model.training;
  std::vector<float> population;
  for (int d = 0; d < training.device_count; ++d)
    population.insert(population.end(), parameters.begin(), parameters.end());
  std::ofstream out(filename);
  out << std::setprecision(17);
  for (auto& batch : training.train_set) {
    model.find_force(p, population.data(), batch, false, training.device_count);
    for (int d = 0; d < training.device_count; ++d) {
      gpuSetDevice(d);
      auto& data = batch[d];
      auto dump = [&](const char* name, auto& values, int count) {
        using Value = typename std::remove_pointer<decltype(values.data())>::type;
        std::vector<Value> host(count);
        values.copy_to_host(host.data());
        out << name << ' ' << d;
        for (auto x : host)
          out << ' ' << x;
        out << '\n';
      };
      dump("E", data.energy, data.N);
      dump("F", data.force, 3 * data.N);
      dump("V", data.virial, 6 * data.N);
      dump_magnetic(model, data, dump);
    }
  }
}
template <class Model>
void probe(Model& model)
{
  auto& p = *model.model_parameters;
  check_initialization(p);
  Fitness fitness(model);
  SearchProbe search(p, model);
  search.dump(p);
  model.prepare_training();
  std::vector<float> elite = search.center();
  if (std::ifstream("nep.txt").good())
    model.read_model(elite.data());
  gpuSetDevice(0);
  p.q_scaler_gpu[0].copy_to_host(p.q_scaler_cpu.data());
  std::ofstream scalers("scaler.txt");
  scalers << std::setprecision(9);
  for (float x : p.q_scaler_cpu)
    scalers << x << '\n';
  std::ofstream baseline("baseline.txt");
  baseline << std::setprecision(17);
  for (double x : p.energy_reference)
    baseline << x << '\n';
  evaluate(model, search.center(), "mu.pred");
  evaluate(model, elite, "elite.pred");
}
int main()
{
  ParameterInput input;
  if (input.spin_version) {
    NEP_Spin_Model model(input);
    probe(model);
  } else {
    NEP model(input);
    probe(model);
  }
}
