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
  void update(double lr, double* gradients); // Update moments and parameters
  double* get_parameters(); // get current parameters pointer
  void output_parameters(Parameters& para); // Output parameters
  void initialize_parameters(Parameters& para); // Initialize optimizer parameters

protected:
  // random number generator
  std::mt19937 rng;

  // ADAM optimizer hyperparameters
  // double lr = 1e-3f;     // learning rate α
  double beta1 = 0.9;   // first order moment decay rate β₁
  double beta2 = 0.999; // second order moment decay rate β₂ 
  double eps = 1e-8;    // small constant ε to avoid division by zero 
  double weight_decay = 0.0; // L2 regularization weight

  // training parameters
  int step = 0; // initial step
  int number_of_variables; // number of variables

  // CPU vectors
  GPU_Vector<curandState> curand_states; 
  std::vector<double> parameters;  // Parameters to be optimized θ
  std::vector<double> m;         // First moment m
  std::vector<double> v;         // Second moment v (adaptive learning rate)
  // GPU vectors
  GPU_Vector<double> gpu_parameters; // Parameters to be optimized θ
  GPU_Vector<double> gpu_m; // First moment m
  GPU_Vector<double> gpu_v; // Second moment v

};
