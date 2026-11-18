// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "action.cuh"
#include <array>
#include <fstream>
#include <map>

struct XrdNode {
  int h, k, l;
  double kx, ky, kz; // cycles per angstrom
  double two_theta;
  double s_squared;
  double lp;
};

struct XrdFormFactor {
  double a[4], b[4], c;
};

class XRD : public Action
{
public:
  XRD(const std::vector<std::string>& tokens, const std::vector<Group>& groups);

  void pre_run(
    int number_of_steps,
    double time_step,
    Integrate& integrate,
    std::vector<Group>& groups,
    Atom& atom,
    Box& box,
    Force& force) override;

  void end_of_step(
    int number_of_steps,
    int step,
    int fixed_group,
    int move_group,
    double global_time,
    double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& groups,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

private:
  void update_factors(const Atom& atom, const std::vector<Group>& groups);
  void update_mesh(const Box& box);

  int interval_;
  double wavelength_;
  double angle_min_ = 1.0;
  double angle_max_ = 179.0;
  bool use_lp_ = true;
  bool manual_ = false;
  std::array<double, 3> mesh_{1.0, 1.0, 1.0};
  std::array<double, 3> spacing_{};
  int grouping_method_ = -1;
  int group_id_ = -1;
  int number_selected_ = 0;
  std::map<std::string, std::string> factor_labels_;
  std::string filename_ = "xrd.txt";
  std::ofstream output_;
  std::vector<XrdNode> nodes_;
  std::vector<double> intensities_;
  std::vector<XrdFormFactor> cpu_factors_;
  std::vector<bool> factor_initialized_;
  GPU_Vector<int> selected_;
  GPU_Vector<XrdFormFactor> factors_;
  GPU_Vector<XrdNode> gpu_nodes_;
  GPU_Vector<double> gpu_intensities_;
};
