/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

// Independent parameter storage for the analytic Hessian. The shared model
// reader validates the file; NEP's private parameters are never accessed.
#pragma once
#include "force/nep/nep.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"

class NEP_Hessian_Model
{
public:
  NEP::ParaMB paramb;
  NEP::ANN annmb;
  NEP::ZBL zbl;

  void initialize(const char* file_potential);

private:
  GPU_Vector<float> parameters_;
};

// Match NEP's choice of expanded-box neighbors for periodic small boxes.
inline bool model_requires_expanded_box(const NEP_Hessian_Model& model, const Box& box)
{
  const double volume = box.get_volume();
  const double cutoff = model.paramb.rc_radial_max;
  return
    (box.pbc_x && volume / box.get_area(0) <= 2.5 * (cutoff + 1.0)) ||
    (box.pbc_y && volume / box.get_area(1) <= 2.5 * (cutoff + 1.0)) ||
    (box.pbc_z && volume / box.get_area(2) <= 2.5 * (cutoff + 1.0));
}
