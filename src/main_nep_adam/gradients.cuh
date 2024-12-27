#pragma once
#include "utilities/gpu_vector.cuh"

struct Gradients {
  void resize(int num_c, int N, int number_of_variables_ann) {
    E_c.resize(N * num_c);
    F_c_x.resize(N * num_c);
    F_c_y.resize(N * num_c);
    F_c_z.resize(N * num_c);
    V_c_xx.resize(N * num_c);
    V_c_yy.resize(N * num_c);
    V_c_zz.resize(N * num_c);
    V_c_xy.resize(N * num_c);
    V_c_yz.resize(N * num_c);
    V_c_zx.resize(N * num_c);
    grad_c_sum.resize(num_c);
    E_wb_grad.resize(N * number_of_variables_ann);
    F_wb_grad_x.resize(N * number_of_variables_ann);
    F_wb_grad_y.resize(N * number_of_variables_ann);
    F_wb_grad_z.resize(N * number_of_variables_ann);
    V_wb_grad_xx.resize(N * number_of_variables_ann);
    V_wb_grad_yy.resize(N * number_of_variables_ann);
    V_wb_grad_zz.resize(N * number_of_variables_ann);
    V_wb_grad_xy.resize(N * number_of_variables_ann);
    V_wb_grad_yz.resize(N * number_of_variables_ann);
    V_wb_grad_zx.resize(N * number_of_variables_ann);
    grad_wb_sum.resize(number_of_variables_ann);
  }

  GPU_Vector<double> E_c;         // partial energy / partial C
  GPU_Vector<double> F_c_x;       // partial force / partial C
  GPU_Vector<double> F_c_y;       // partial force / partial C
  GPU_Vector<double> F_c_z;       // partial force / partial C
  GPU_Vector<double> V_c_xx;      // partial virial_xx / partial C
  GPU_Vector<double> V_c_yy;      // partial virial_yy / partial C
  GPU_Vector<double> V_c_zz;      // partial virial_zz / partial C
  GPU_Vector<double> V_c_xy;      // partial virial_xy / partial C
  GPU_Vector<double> V_c_yz;      // partial virial_yz / partial C
  GPU_Vector<double> V_c_zx;      // partial virial_zx / partial C
  GPU_Vector<double> grad_c_sum;  // sum of energy, force, virial w.r.t. C
  GPU_Vector<double> E_wb_grad;      // energy w.r.t. w0, b0, w1, b1
  GPU_Vector<double> F_wb_grad_x;      // force w.r.t. w0, b0, w1, b1
  GPU_Vector<double> F_wb_grad_y;      // force w.r.t. w0, b0, w1, b1
  GPU_Vector<double> F_wb_grad_z;      // force w.r.t. w0, b0, w1, b1
  GPU_Vector<double> V_wb_grad_xx;      // virial w.r.t. w0, b0, w1, b1
  GPU_Vector<double> V_wb_grad_yy;      // virial w.r.t. w0, b0, w1, b1
  GPU_Vector<double> V_wb_grad_zz;      // virial w.r.t. w0, b0, w1, b1
  GPU_Vector<double> V_wb_grad_xy;      // virial w.r.t. w0, b0, w1, b1
  GPU_Vector<double> V_wb_grad_yz;      // virial w.r.t. w0, b0, w1, b1
  GPU_Vector<double> V_wb_grad_zx;      // virial w.r.t. w0, b0, w1, b1
  GPU_Vector<double> grad_wb_sum;      // sum energy, force, virial w.r.t. w0, b0, w1, b1
};
