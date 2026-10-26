/* Single-process spatial decomposition for NEP-Spin. LGPL-3.0-or-later. */
#pragma once
#include "nep_spin.cuh"
#include <memory>

class NEP_Spin_MultiGPU : public Potential
{
public:
  NEP_Spin_MultiGPU(int devices, const char* model, int atoms, int direction);
  ~NEP_Spin_MultiGPU() override;
  using Potential::compute;
  void compute(Box&, const GPU_Vector<int>&, const GPU_Vector<double>&,
    GPU_Vector<double>&, GPU_Vector<double>&, GPU_Vector<double>&) override
  { PRINT_INPUT_ERROR("NEP_Spin requires spin inputs."); }
  void compute(Box&, const GPU_Vector<int>&, const GPU_Vector<double>&,
    const GPU_Vector<double>&, GPU_Vector<double>&, GPU_Vector<double>&,
    GPU_Vector<double>&, GPU_Vector<double>&) override;
private:
  struct Domain {
    std::unique_ptr<NEP_Spin> potential;
    GPU_Vector<int> type;
    GPU_Vector<double> position, spin, energy, force, virial, mforce;
    int start = 0, size = 0, first = 0, owned = 0, global_first = 0;
    void resize(int count);
  };
  int devices_, atoms_, direction_;
  std::vector<std::unique_ptr<Domain>> domains_;
  GPU_Vector<int> cell_count_, cell_sum_, order_, packed_type_;
  GPU_Vector<double> packed_position_, packed_spin_, collected_;
  std::vector<int> prefix_;
  GPU_Vector<double> partition_position_;
  GPU_Vector<int> partition_changed_;
  double partition_box_[9] = {};
  bool partition_valid_ = false;
  gpuStream_t partition_stream_ = nullptr;
};
