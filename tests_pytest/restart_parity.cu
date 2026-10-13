// Inspect actual SNES restore before sampling; link against either old or new NEP objects.
#include "main_nep/parameters.cuh"
#include "main_nep/fitness.cuh"
#include "main_nep/snes.cuh"
#include "utilities/nep_parameters.cuh"
#include <fstream>
#include <iomanip>
#include <string>

class SearchProbe : public SNES {
public:
  using SNES::SNES;
  const std::vector<float>& center() const { return mu; }
  void dump(Parameters& p) {
    output_mu_and_sigma(p, "restored.restart");
    std::ofstream state("search-state.txt");
    state << "curriculum " << curriculum_enabled << '\n';
  }
};
class PredictionProbe : public Fitness {
public:
  using Fitness::Fitness;
  void evaluate(Parameters& p, const std::vector<float>& center, const char* filename) {
    int devices; gpuGetDeviceCount(&devices);
    std::vector<float> parameters;
    for (int d=0; d<devices; ++d) parameters.insert(parameters.end(), center.begin(), center.end());
    std::ofstream out(filename); out << std::setprecision(9);
    for (auto& batch : train_set) {
      potential->find_force(p, parameters.data(), batch, false, devices);
      for (int d=0; d<devices; ++d) {
        gpuSetDevice(d);
        auto& data=batch[d];
        auto dump=[&](const char* name, GPU_Vector<float>& values, int count) {
          std::vector<float> host(count); values.copy_to_host(host.data());
          out << name << ' ' << d;
          for (float x : host) out << ' ' << x;
          out << '\n';
        };
        dump("E",data.energy,data.N); dump("F",data.force,3*data.N);
        dump("V",data.virial,6*data.N);
        if (p.spin_mode) dump("MF",data.mforce,3*data.N);
      }
    }
  }
};
int main(int argc, char** argv) {
  Parameters p;
  p.maximum_generation=0; // construct/restore real SNES state without a sampled generation
  PredictionProbe f(p);
  SearchProbe search(p,&f);
  search.dump(p);
  std::ifstream input("nep.txt");
  std::vector<float> elite(p.number_of_variables), saved_scaler(p.dim);
  if (input.good()) {
    NepTxtHeader header; std::string error;
    if (!read_nep_txt_header("nep.txt",header,error)) return 2;
    std::string line;
    for (int i=0;i<header.number_of_header_lines;++i) std::getline(input,line);
    for (float& value : elite) input >> value;
    for (float& value : saved_scaler) input >> value;
    if (!input) return 3;
    descriptor_parameters_to_channel_major(elite.data(),p.number_of_variables_ann,
      p.num_types*p.num_types,p.n_max_radial,p.n_max_angular,p.basis_size_radial,p.basis_size_angular);
    if (argc>1 && std::string(argv[1])=="reference") {
      int devices;gpuGetDeviceCount(&devices);
      for (int d=0;d<devices;++d) {gpuSetDevice(d);p.q_scaler_gpu[d].copy_from_host(saved_scaler.data());}
    }
  } else elite=search.center();
  gpuSetDevice(0);p.q_scaler_gpu[0].copy_to_host(p.q_scaler_cpu.data());
  std::ofstream scalers("scaler.txt");scalers << std::setprecision(9);
  for(float x:p.q_scaler_cpu) scalers << x << '\n';
  std::ofstream baseline("baseline.txt");baseline << std::setprecision(9);
  for(float x:p.spin_baseline) baseline << x << '\n';
  f.evaluate(p,search.center(),"mu.pred");
  f.evaluate(p,elite,"elite.pred");
}
