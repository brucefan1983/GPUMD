#pragma once
#include "utilities/gpu_vector.cuh"

static void print_memory_info(const char* stage) {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    double free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
    double total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
    double used_gb = total_gb - free_gb;
    
    printf("%s:\n", stage);
    printf("  Free memory:  %.2f GB\n", free_gb);
    printf("  Used memory:  %.2f GB\n", used_gb);
    printf("  Total memory: %.2f GB\n", total_gb);
}

struct Gradients {
  void resize(int N, int max_NN_radial, int num_variables, int number_of_variables_ann, int dim, int dim_radial, int basis_size_radial) {
    grad_sum.resize(num_variables);
    // print_memory_info("after grad_sum.resize");
    E_wb_grad.resize(N * number_of_variables_ann);
    // print_memory_info("after E_wb_grad.resize");
    Fp_wb.resize(N * number_of_variables_ann * dim);
    // print_memory_info("after Fp_wb.resize");
    q_c.resize(N * max_NN_radial * dim_radial * (basis_size_radial + 1));
    // print_memory_info("after q_c.resize");
  }
  GPU_Vector<double> E_wb_grad;      // energy w.r.t. w0, b0, w1, b1
  GPU_Vector<double> grad_sum;
  GPU_Vector<double> Fp_wb;          // gradient of descriptors w.r.t. w0, b0, w1, b1
  GPU_Vector<double> q_c;        // partial descriptors / partial C
};
