/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "force/nep_spin/spin_storage.cuh"
#include "main_nep/common/energy_difference.cuh"
#include "main_nep/common/nep_batch_data.cuh"
#include "main_nep/common/nep_common.cuh"
#include "main_nep/common/nep_compile.cuh"
#include "main_nep/common/nep_energy_reader.cuh"
#include "main_nep/common/nep_evaluate.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "main_nep/common/nep_training.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/potential.cuh"
#include <memory>
#include <set>

#include "spin_data.cuh"
class NEP_Spin_Model : public Potential
{
public:
  explicit NEP_Spin_Model(const ParameterInput&);
  int prepare_data() override;
  void prepare_training() override;
  void evaluate(int, const float*) override;
  void accumulate_fitness(float*) override;
  void print_training_header() override;
  void report_error(const TrainingReport&) override;
  void predict(float*) override;
  Parameters& get_parameters() override { return *model_parameters; }
  void initialize_parameters(std::mt19937&, std::vector<float>&, std::vector<float>&) override;
  void parameter_groups(std::vector<int>&) override;
  void read_model(float*) override;
  void write_restart(const char*, const std::vector<float>&, const std::vector<float>&) override;


private:
  class ResponseLoss
  {
  public:
    void append(int device_id, Dataset& dataset, Spin_Data& spin_data);
    float value() const;

  private:
    struct Point {
      std::string group;
      double coordinate;
      double prediction;
      double target;
    };
    std::vector<Point> points_;
  };

  class Spin_Reader : public NEP_Energy_Reader
  {
  public:
    Spin_Reader(Spin_Parameters& p, EnergyDifference& e, NEP_Batch_Data<Spin_Data>& d)
      : NEP_Energy_Reader(p, e), parameters(p), data(d)
    {
    }
    void read_response(std::vector<Structure>&, std::vector<Spin_Data::FrameInput>&);

  private:
    void prepare_train(std::vector<Structure>&, int, int&, std::vector<int>&) override;
    void prepare_test(const std::vector<Structure>&, bool) override;
    void prepare_dataset(Dataset&) override;
    const char* energy_difference_conflict(const Structure&, const Structure&) const override;
    XYZ_Geometry* precise_geometry() override { return &labels.back().geometry; }
    void begin_reading() override { labels.clear(); }
    void read_structure(const XYZ_Record&, Structure&) override;
    struct Frame {
      const Structure& geometry;
      Spin_Data::FrameInput& label;
    };
    void derive_spin_response_tangents(std::vector<Frame>&);
    Spin_Parameters& parameters;
    NEP_Batch_Data<Spin_Data>& data;
    std::vector<Spin_Data::FrameInput> labels;
    std::set<std::string> train_groups, test_groups, test_frames;
  };

  void write_spin_model(FILE*, const float*, int max_radial, int max_angular);
  void finalize_q_scaler();
  void initialize_search(std::mt19937&, std::vector<float>&, std::vector<float>&) const;
  std::unique_ptr<Spin_Parameters> model_parameters;
  std::unique_ptr<NEP_Training> training;
  NEP_Batch_Data<Spin_Data> magnetic_data;
  std::vector<Dataset> response_set;
  std::vector<std::unique_ptr<Spin_Data>> response_data;
  std::vector<float> fitness_mforce, fitness_tau, fitness_response;
  NEP_Energy_Loss energy_loss;
  EnergyDifference energy_difference;
  NEP_Neighbor neighbor[16];
  NEP_Descriptor_Parameters paramb;
  NEP_ANN annmb[16];
  NEP_Data nep_data[16];
  NEP_ZBL zbl;
  std::unique_ptr<NEP_Compiled_Evaluation> compiled_kernel_;
  std::unique_ptr<NEP_Compile> compiled_spin_;
  using SpinDescriptorFunction = void (*)(
    int atom_count,
    int struct_dim,
    const float* spin_cutoff_pair,
    const int* type,
    const int* spin_type_active,
    const double* position,
    const double* spin,
    const int* NN_spin,
    const int* NL_spin,
    const double* r12_spin,
    int r12_plane_size,
    const float* descriptor_coefficients,
    const float* projection,
    float* moments,
    float* descriptors,
    double* wide_spin_descriptors,
    const double* wide_radial);
  SpinDescriptorFunction spin_descriptor_jit_ = nullptr;
  using SpinForceFunction = void (*)(
    int atom_count,
    int struct_dim,
    const float* spin_cutoff_pair,
    const int* type,
    const int* spin_type_active,
    const double* position,
    const double* spin,
    const int* NN_spin,
    const int* NL_spin,
    const double* r12_spin,
    int r12_plane_size,
    const float* Fp,
    const float* descriptor_coefficients,
    const float* projection,
    const float* moments,
    double* force,
    double* mforce,
    double* virial,
    const double* wide_radial);
  SpinForceFunction spin_force_jit_ = nullptr;
  void initialize_spin_compiled(const Spin_Parameters&);
  void initialize(Spin_Parameters&, int, int);
  void initialize_model(const Spin_Parameters&);
  void initialize_device(const Spin_Parameters&, int, int);
  void initialize_compiled_kernel(const Spin_Parameters&);
  void initialize_spin(Spin_Parameters&, int, int);
  void find_force(Spin_Parameters&, const float*, std::vector<Dataset>&, bool, int);
  Spin_Data& magnetic(Dataset& dataset);
  void find_additional_descriptors(Spin_Parameters&, Dataset&, int);
  void initialize_additional_outputs(Dataset&, int);
  void find_additional_force(Spin_Parameters&, Dataset&, int);
  float response_loss(int);
  void write_magnetic_predictions(const char*, const std::vector<std::pair<Dataset*, int>>&);
  struct Spin_Workspace {
    nep_spin::Storage spin_layout;
    GPU_Vector<int> spin_type_active;
    GPU_Vector<float> spin_cutoff_pair;
    GPU_Vector<float> spin_moments;
    GPU_Vector<double> wide_spin_descriptors, wide_radial;
    GPU_Vector<float> radial_coefficients;
    GPU_Vector<double> force;
    GPU_Vector<double> mforce;
    GPU_Vector<double> virial;
  };
  int spin_compress_ = 0;
  int spin_projection_offset_ = 0;
  int struct_dim_ = 0;
  int num_types_ = 0;
  Spin_Workspace spin_data_[16];
};
