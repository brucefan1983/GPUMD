#ifdef USE_TENSORFLOW
#pragma once
#include "model/box.cuh"
#include "DeepPot.h"
#include "potential.cuh"
#include <stdio.h>
#include <vector>

namespace deepmd_compat = deepmd;

struct DEEPMD_Data {
  GPU_Vector<int> ghost_type;
  GPU_Vector<double> ghost_position;
};

struct DEEPMD_GHOST_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> ghost_number;
  GPU_Vector<int> ghost_type;
  GPU_Vector<double> ghost_position;
};

class DEEPMD : public Potential
{
public:
  using Potential::compute;
  double ener_unit_cvt_factor, dist_unit_cvt_factor, force_unit_cvt_factor, virial_unit_cvt_factor;
  DEEPMD(const char*, int);
  virtual ~DEEPMD(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void initialize_deepmd(const char*); // called by the constructor

protected:
  Box ghost_box;
  DEEPMD_Data deepmd_data;
  DEEPMD_GHOST_Data deepmd_ghost_data;
  deepmd_compat::DeepPot deep_pot;
  double **scale;

private:
  double rc;
  int numb_types;
  int numb_types_spin;
  int dim_aparam;
  unsigned numb_models;
  int stdf_comm_buff_size;
  double eps;
  double eps_v;
  bool do_ttm;
  bool do_compute_fparam;
  bool do_compute_aparam;
  bool atom_spin_flag;
  bool single_model;
  bool multi_models_mod_devi;
  bool multi_models_no_mod_devi;  
};
#endif
