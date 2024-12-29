/*
 * adam.cuh
 *
 *  This file contains the definitions of the Adam optimizer.
 *
 *  Created on: Nov 19, 2024
 *      Author: Hongfu Huang
 *      Email: hfhuang@buaa.edu.cn
 */
 /*----------------------------------------------------------------------------
Use the Adam optimizer to fit potential parameters.

Reference:

Adam: A method for stochastic optimization
doi: 10.48550/arXiv.1412.6980
------------------------------------------------------------------------------*/
 

#include "adam.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include <chrono>

static __global__ void initialize_curand_states(curandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curand_init(seed, n, 0, &state[n]);
  }
}

static __global__ void update_moments(
  const int N,
  const double beta1,
  const double beta2,
  const double* __restrict__ gradients,
  double* __restrict__ m,
  double* __restrict__ v)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const double g = gradients[i];
    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
  }
}

static __global__ void apply_updates(
  const int N,
  const double lr,
  const double beta1_t,
  const double beta2_t,
  const double eps,
  const double weight_decay,
  const double* __restrict__ m,
  const double* __restrict__ v,
  double* __restrict__ parameters)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    // calculate bias correction
    const double m_hat = m[i] / (1.0 - beta1_t);
    const double v_hat = v[i] / (1.0 - beta2_t);
    
    // update parameters
    parameters[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * parameters[i]);
  }
}

__global__ void adam_update(
    const int N,
    const double step_size,  // = lr * sqrt(1 - beta2^t)/(1 - beta1^t)
    const double beta1_power,// = beta1^t
    const double beta2_power,// = beta2^t
    const double eps,
    double* __restrict__ m,
    double* __restrict__ v,
    double* __restrict__ param)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double bc1 = 1.0 - beta1_power;
        double bc2 = 1.0 - beta2_power;

        double m_hat = m[i] / bc1;   
        double v_hat = v[i] / bc2;   

        // param[i] -= step_size * (m_hat / (sqrt(v_hat) + eps));
        param[i] = param[i] - step_size * (m_hat / (sqrt(v_hat) + eps));
    }
}


Adam::Adam(Parameters& para)
{
  // initialize the parameters
  number_of_variables = para.number_of_variables;
  weight_decay = para.lambda_2;

  // initialize the CPU vectors
  parameters.resize(number_of_variables);
  // gradients.resize(number_of_variables);
  m.resize(number_of_variables, 0.0);
  v.resize(number_of_variables, 0.0);

  // initialize the GPU vectors
  cudaSetDevice(0);
  gpu_parameters.resize(number_of_variables);
  // gpu_gradients.resize(number_of_variables);
  gpu_m.resize(number_of_variables);
  gpu_v.resize(number_of_variables);
  curand_states.resize(number_of_variables);
  initialize_curand_states<<<(number_of_variables - 1) / 128 + 1, 128>>>(curand_states.data(), number_of_variables, 1234567);
  CUDA_CHECK_KERNEL

  // initialize the optimizer parameters
  // initialize_parameters(para);
}

static __global__ void gpu_create_paramters(
  const int dim,
  const int num_types,
  const int num_neurons1,
  const int number_of_variables_ann,
  const int number_of_variables,
  const double* energy_shift,
  curandState* g_state,
  double* g_parameters)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < number_of_variables_ann) {
    curandState state = g_state[n];
    int type_idx = (n * num_types) / number_of_variables_ann;
    int param_idx = n % ((dim + 2) * num_neurons1 + 1);
    if (param_idx < dim * num_neurons1) {
      double std = 1.0 / sqrt(static_cast<double>(dim + num_neurons1));
      g_parameters[n] = curand_normal(&state) * std;
    } else if (param_idx < (dim + 1) * num_neurons1) {
      g_parameters[n] = curand_normal(&state);
    } else if (param_idx < (dim + 2) * num_neurons1) {
      double std = 1.0 / sqrt(static_cast<double>(num_neurons1 + 1));
      g_parameters[n] = curand_normal(&state) * std;
    } else {
      double mean = energy_shift[type_idx];
      g_parameters[n] = mean + curand_normal(&state);
    }
  } else if (n < number_of_variables) {
    curandState state = g_state[n];
    double r1 = curand_normal(&state);
    double r2 = curand_uniform(&state) - 0.5;
    g_parameters[n] = 0.1 * r1 + r2;
  }
}

void Adam::initialize_parameters(Parameters& para)
{
  FILE* fid_restart = fopen("nep.restart", "r");
  if (fid_restart == NULL) {
    cudaSetDevice(0); // normally use GPU-0
    gpu_create_paramters<<<(number_of_variables - 1) / 128 + 1, 128>>>(
      para.dim,
      para.num_types,
      para.num_neurons1,
      para.number_of_variables_ann,
      para.number_of_variables,
      para.energy_shift_gpu.data(),
      curand_states.data(),
      gpu_parameters.data());
    CUDA_CHECK_KERNEL
    gpu_parameters.copy_to_host(parameters.data());
    // parameters[0] = 0.442027390003;
    // parameters[1] = -0.114872045815;
    // parameters[2] = 0.594669640064;
    // parameters[3] = -0.302220880985;
    // parameters[4] = 0.180428802967;
    // parameters[5] = 0.181583940983;
    // parameters[6] = 1.395557761192;
    // parameters[7] = -0.996886909008;
    // parameters[8] = 2.890779733658;
    // parameters[9] = 1.302489280701;
    // parameters[10] = 0.246913835406;
    // parameters[11] = 0.068984314799;
    // parameters[12] = -14.030391693115; // bias1
    // parameters[13] = 0.564685225487;
    // parameters[14] = -0.836443722248;
    // parameters[15] = -0.681497931480;
    // parameters[16] = -0.345571577549;
    // parameters[17] = 0.633350133896;
    // parameters[18] = 0.206515565515;
    // parameters[19] = -0.565002620220;
    // parameters[20] = -0.658045351505;
    // parameters[21] = -1.241125106812;
    // parameters[22] = 0.027299404144;
    // parameters[23] = 0.440154761076;
    // parameters[24] = 0.592076122761;
    // parameters[25] = -51.896118164062; 
    // parameters[26] = -0.359654529203;  // c[0,0,0,0],0
    // parameters[27] = 0.076675398038;   // c[0,1,0,0],1
    // parameters[28] = 0.031677418640; // c[1,0,0,0],2
    // parameters[29] = 0.099766372147; // c[1,1,0,0],3
    // parameters[30] = 0.503399248922; // c[0,0,0,1],4
    // parameters[31] = 0.402137601819; // c[0,1,0,1],5
    // parameters[32] = -0.250388690935; // c[1,0,0,1],6
    // parameters[33] = 0.273757521254; // c[1,1,0,1],7
    // parameters[34] = 0.216046105857; // c[0,0,1,0],8
    // parameters[35] = -0.047669816146; // c[0,1,1,0],9
    // parameters[36] = -0.272421504828; // c[1,0,1,0],10
    // parameters[37] = -0.297049618754; // c[1,1,1,0],11
    // parameters[38] = 0.529698182479; // c[0,0,1,1],12
    // parameters[39] = -0.488307326645; // c[0,1,1,1],13
    // parameters[40] = 0.490723012081; // c[1,0,1,1],14
    // parameters[41] = 0.077293731498; // c[1,1,1,1],15
    // parameters[42] = 1.0;
    // parameters[43] = 1.0;
    // parameters[44] = 1.0;
    // parameters[45] = 1.0;
    // gpu_parameters.copy_from_host(parameters.data());
  } else {
    for (int n = 0; n < number_of_variables; ++n) {
        int count = fscanf(fid_restart, "%lf", &parameters[n]);
        PRINT_SCANF_ERROR(count, 2, "Reading error for nep.restart.");
    }
    fclose(fid_restart);
  }
}

void Adam::update(double lr, double* gradients) {
  const int block_size = 256;
  const int grid_size = (number_of_variables + block_size - 1) / block_size + 1;

  update_moments<<<grid_size, block_size>>>(
    number_of_variables,
    beta1,
    beta2,
    gradients,
    gpu_m.data(),
    gpu_v.data()
  );
  CUDA_CHECK_KERNEL
  
  // calculate bias correction
  const double beta1_t = pow(beta1, step + 1);
  const double beta2_t = pow(beta2, step + 1);

  // apply parameter updates
  apply_updates<<<grid_size, block_size>>>(
    number_of_variables,
    lr,
    beta1_t,
    beta2_t,
    eps,
    weight_decay,
    gpu_m.data(),
    gpu_v.data(),
    gpu_parameters.data()
  );
  CUDA_CHECK_KERNEL

  gpu_parameters.copy_to_host(parameters.data());
  // printf("Parameters: \n");
  // for (int n = 0; n < number_of_variables; ++n) {
  //   printf("%d %f\n", n, parameters[n]);
  // }
  // printf("\n");
  step++; 
}

void Adam::output_parameters(Parameters& para) {
  cudaSetDevice(0); 
  gpu_parameters.copy_to_host(parameters.data());
  FILE* fid = fopen("nep.restart", "w");
  for (int i = 0; i < number_of_variables; ++i) {
    fprintf(fid, "%15.7e\n", parameters[i]);
  }
  fclose(fid);
}

double* Adam::get_parameters()
{
  return parameters.data();
}