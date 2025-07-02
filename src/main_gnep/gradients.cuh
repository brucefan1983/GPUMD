#pragma once
#include "utilities/gpu_vector.cuh"

static void print_memory_info(const char* stage) {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    float free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
    float total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
    float used_gb = total_gb - free_gb;
    
    printf("%s:\n", stage);
    printf("  Free memory:  %.2f GB\n", free_gb);
    printf("  Used memory:  %.2f GB\n", used_gb);
    printf("  Total memory: %.2f GB\n", total_gb);
}

struct Gradients {
  void resize(int N, int num_variables, int number_of_variables_ann, int dim) {
    grad_sum.resize(num_variables, 0.0f);
    // print_memory_info("after grad_sum.resize");
    E_wb_grad.resize(N * number_of_variables_ann, 0.0f);
    // print_memory_info("after E_wb_grad.resize");
    Fp_wb.resize(N * number_of_variables_ann * dim, 0.0f);
    // print_memory_info("after Fp_wb.resize");
  }
  void clear() {
    E_wb_grad.fill(0.0f);
    grad_sum.fill(0.0f);
    Fp_wb.fill(0.0f);
  }
  GPU_Vector<float> E_wb_grad;      // energy w.r.t. w0, b0, w1, b1
  GPU_Vector<float> grad_sum;
  GPU_Vector<float> Fp_wb;          // gradient of descriptors w.r.t. w0, b0, w1, b1
};
