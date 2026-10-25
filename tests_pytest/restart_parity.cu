// Inspect actual SNES restore before sampling and dump fixed-state predictions.
#include "spin_test_access.cuh"
#include "main_nep/fitness.cuh"
#include "main_nep/snes.cuh"
class SearchProbe : public SNES {
public:
  using SNES::SNES;
  const std::vector<float>& center() const { return mu; }
  void dump(Parameters& p) { output_mu_and_sigma(p, "restored.restart"); }
};
template<class Model>
void evaluate(Model& model, const std::vector<float>& parameters, const char* filename) {
  auto& p=*model.model_parameters;
  auto& training=*model.training;
  std::vector<float> population;
  for(int d=0;d<training.device_count;++d) population.insert(population.end(),parameters.begin(),parameters.end());
  std::ofstream out(filename); out<<std::setprecision(17);
  for(auto& batch:training.train_set) {
    model.find_force(p,population.data(),batch,false,training.device_count);
    for(int d=0;d<training.device_count;++d) {
      gpuSetDevice(d); auto& data=batch[d];
      auto dump=[&](const char* name,auto& values,int count) {
        using Value = typename std::remove_pointer<decltype(values.data())>::type;
        std::vector<Value> host(count);values.copy_to_host(host.data());out<<name<<' '<<d;
        for(auto x:host) out<<' '<<x;out<<'\n';
      };
      dump("E",data.energy,data.N);dump("F",data.force,3*data.N);dump("V",data.virial,6*data.N);
      if constexpr (std::is_same<Model,NEP_Spin_Model>::value) dump("MF",model.magnetic(data).mforce,3*data.N);
    }
  }
}
template<class Model> void probe(Model& model) {
  auto& p=*model.model_parameters;
  Fitness fitness(model);
  SearchProbe search(p,model);search.dump(p);
  model.prepare_training();
  std::vector<float> elite=search.center();
  if(std::ifstream("nep.txt").good()) model.read_model(elite.data());
  gpuSetDevice(0);p.q_scaler_gpu[0].copy_to_host(p.q_scaler_cpu.data());
  std::ofstream scalers("scaler.txt");scalers<<std::setprecision(9);for(float x:p.q_scaler_cpu) scalers<<x<<'\n';
  std::ofstream baseline("baseline.txt");baseline<<std::setprecision(17);
  for(double x:p.energy_reference) baseline<<x<<'\n';
  evaluate(model,search.center(),"mu.pred");evaluate(model,elite,"elite.pred");
}
int main() {
  ParameterInput input;
  if(input.spin_version) {NEP_Spin_Model model(input);probe(model);}
  else {NEP model(input);probe(model);}
}
