/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under GPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#pragma once
#include "spin_data.cuh"
namespace fitness_spin {
void fit_spin_energy_baseline(const std::vector<Structure>&, Spin_Parameters&);
void derive_spin_response_tangents(const Spin_Parameters&, std::vector<Spin_Frame>&);
void finalize_q_scaler(Spin_Parameters&, int);
void print_loss_header();
class ResponseLoss {
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

}
