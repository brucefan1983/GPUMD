/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#pragma once
#include "spin_parameters.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/common/nep_energy_reader.cuh"
#include "main_nep/common/nep_batch_data.cuh"
#include <set>
struct Spin_Labels {
  int has_mforce = 0, has_spin_response_metadata = 0, has_spin_response = 0;
  double spin_response_coordinate = 0.0;
  std::string spin_response_probe, spin_response_group;
  std::vector<float> sx, sy, sz, mfx, mfy, mfz;
  std::vector<float> spin_tangent_x, spin_tangent_y, spin_tangent_z;
};

struct Spin_Frame {
  const Structure& geometry;
  Spin_Labels& label;
};
struct Spin_Data {
  int max_NN_spin = 0;
  std::vector<Spin_Labels> labels;
  GPU_Vector<double> r_spin, spin, r12_spin;
  GPU_Vector<int> NN_spin, NL_spin, has_mforce_gpu, spin_type_active_gpu;
  GPU_Vector<float> mforce, mforce_ref_gpu;
  std::vector<float> mforce_cpu, mforce_ref_cpu;
  std::vector<int> has_mforce_cpu;
  // Per-frame loss counts, fixed with labels and the active mask at construction.
  std::vector<int> active_count_cpu, torque_active_count_cpu;
  void construct(Spin_Parameters&, Dataset&, const std::vector<Spin_Labels>&);
  std::vector<float> get_rmse_mforce(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_rmse_mforce_parallel(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_rmse_mforce_perp(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_rmse_tau(Spin_Parameters&, Dataset&, bool, int);
};
class Spin_Reader : public NEP_Energy_Reader {
public:
  Spin_Reader(Spin_Parameters& p, EnergyDifference& e, NEP_Batch_Data<Spin_Data>& d)
    : NEP_Energy_Reader(p,e), parameters(p), data(d) {}
  void prepare_train(std::vector<Structure>&, int, int&, std::vector<int>&) override;
  void prepare_test(const std::vector<Structure>&, bool) override;
  void prepare_dataset(Dataset&) override;
  const char* energy_difference_conflict(const Structure&, const Structure&) const override;
  void read_response(std::vector<Structure>&, std::vector<Spin_Labels>&);
protected:
  void begin_reading() override { labels.clear(); }
  void read_structure(const XYZ_Record&, Structure&) override;
private:
  Spin_Parameters& parameters;
  NEP_Batch_Data<Spin_Data>& data;
  std::vector<Spin_Labels> labels;
  std::set<std::string> train_groups, test_groups, test_frames;
};
