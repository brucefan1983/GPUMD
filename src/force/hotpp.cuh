#ifdef USE_TORCH
#pragma once
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "neighbor.cuh"
#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

struct Hotpp : public Potential
// hotpp should be loaded as 
// potential HotppConfig model.pt
// HotppConfig includes atomic map in line1 and any other data need in the succeed lines
{
public:

    Hotpp(std::string model_path,int n_atoms);
    void compute_large_box(Box& box,const GPU_Vector<double>& position_per_atom);
    void get_neighbor_list(Box& box,const GPU_Vector<double>& position_per_atom);

    void compute(
        Box& box,
        const GPU_Vector<int>& type,
        const GPU_Vector<double>& positions,
        GPU_Vector<double>& potentials,
        GPU_Vector<double>& forces,
        GPU_Vector<double>& virial);

    void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& positions,
    GPU_Vector<double>& potentials,
    GPU_Vector<double>& forces,
    GPU_Vector<double>& virial){
       compute(box, type, positions, potentials, forces, virial);
       // enable this if opes needed.
       };

    torch::Dict<std::string, torch::Tensor> predict(const torch::Dict<std::string, torch::Tensor>& inputs);
    torch::Tensor _FromCudaMemory(double* d_array, int size);

private:

    std::vector<double> cpu_b_vector; // Box
    // GPU_Vector<double> gpu_b_vector;  // Box (GPU)
    // 找近邻
    GPU_Vector<int> cell_count;
    GPU_Vector<int> cell_count_sum;
    GPU_Vector<int> cell_contents;
    GPU_Vector<int> NN_radial;    // radial neighbor list
    GPU_Vector<int> NL_radial;    // radial neighbor list
    // torch-model
    torch::jit::script::Module model; // TorchScript 模型
};

#endif