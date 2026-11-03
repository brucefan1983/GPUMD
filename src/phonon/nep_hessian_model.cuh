/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
Independent NEP model loader for the analytic Hessian.

The analytic Hessian needs the NEP model parameters (cutoffs, basis sizes,
descriptor layout) and the trained ANN weights.  Rather than reaching into
the private state of class NEP, this module reads the model file itself and
populates the public nested types NEP::ParaMB, NEP::ANN and NEP::ZBL.

This mirrors the pattern already used elsewhere in GPUMD, where a module
parses the NEP model on its own: see src/mc/nep_energy.cu (NEP_Energy),
src/measure/nep_response.cu and src/force/ilp_nep.cu.

Consequently src/force/nep.cuh and src/force/nep.cu are left untouched.
------------------------------------------------------------------------------*/

#pragma once
#include "force/nep.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <fstream>
#include <string>
#include <vector>

class NEP_Hessian_Model
{
public:
  NEP::ParaMB paramb;
  NEP::ANN annmb;
  NEP::ZBL zbl;

  NEP_Hessian_Model(void);
  ~NEP_Hessian_Model(void);

  // Read `file_potential` and upload the model to the GPU.  Terminates the
  // program with an explanatory message if the file cannot be parsed.
  void initialize(const char* file_potential);

private:
  // GPU copies of the model parameters.  `parameters_` holds the ANN weights
  // followed by the descriptor parameters and the descriptor scalers;
  // `descriptor_parameters_type_pair_` holds the descriptor parameters in the
  // [type_pair][basis] layout used by the force kernels.
  GPU_Vector<float> parameters_;
  GPU_Vector<float> descriptor_parameters_type_pair_;

  // Point the ANN weight pointers into `parameters`.
  void update_potential(float* parameters, NEP::ANN& ann);
};

// True if run.in enables the DFT-D3 dispersion correction.  The analytic
// Hessian does not support DFT-D3, so the command layer uses this to reject
// such configurations explicitly.
inline bool nep_hessian_run_input_has_dftd3()
{
  std::ifstream run_input("run.in");
  if (!run_input.is_open()) return false;
  std::string line;
  while (std::getline(run_input, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (!tokens.empty() && tokens[0] == "dftd3") return true;
  }
  return false;
}

// True if the system is a periodic small box for which NEP needs an expanded
// box.  This mirrors the criterion used inside class NEP.
inline bool model_requires_expanded_box(const NEP_Hessian_Model& model, const Box& box)
{
  const double volume = box.get_volume();
  const double cutoff = model.paramb.rc_radial_max;
  return
    (box.pbc_x && volume / box.get_area(0) <= 2.5 * (cutoff + 1.0)) ||
    (box.pbc_y && volume / box.get_area(1) <= 2.5 * (cutoff + 1.0)) ||
    (box.pbc_z && volume / box.get_area(2) <= 2.5 * (cutoff + 1.0));
}
