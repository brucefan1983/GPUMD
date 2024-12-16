/*
 * adam.cuh
 *
 *  This file contains the definitions of the Adam optimizer.
 *
 *  Created on: Nov 19, 2024
 *      Author: Hongfu Huang
 *      Email: hfhuang@buaa.edu.cn
 */

#pragma once
#include "utilities/gpu_vector.cuh"
#include <curand_kernel.h>
#include <random>
#include <vector>

class Fitness;
class Parameters;

class Adam
{
public:
  Adam(Parameters&);

  // void zero_gradients();
  void update(float lr, float* gradients); // Update moments and parameters
  float* get_parameters(); // get current parameters pointer
  void output_parameters(Parameters& para); // Output parameters

protected:
  // random number generator
  std::mt19937 rng;

  // ADAM optimizer hyperparameters
  // float lr = 1e-3f;     // learning rate α
  float beta1 = 0.9f;   // first order moment decay rate β₁
  float beta2 = 0.999f; // second order moment decay rate β₂ 
  float eps = 1e-8f;    // small constant ε to avoid division by zero 
  float weight_decay = 0.0f; // L2 regularization weight

  // training parameters
  int number_of_variables = 10; // number of variables
  int step = 0; // current step
  int input_dim; // input size
  int num_neurons1; 
  int number_of_variables_descriptor; 

  // CPU vectors
  GPU_Vector<curandState> curand_states; 
  std::vector<float> parameters;  // Parameters to be optimized θ
  // std::vector<float> gradients; // Gradients of parameters g
  std::vector<float> m;         // First moment m
  std::vector<float> v;         // Second moment v (adaptive learning rate)
  // GPU vectors
  GPU_Vector<float> gpu_parameters; // Parameters to be optimized θ
  // GPU_Vector<float> gpu_gradients; // Gradients of parameters g
  GPU_Vector<float> gpu_m; // First moment m
  GPU_Vector<float> gpu_v; // Second moment v

  void initialize_parameters(Parameters& para); // Initialize optimizer parameters
};
