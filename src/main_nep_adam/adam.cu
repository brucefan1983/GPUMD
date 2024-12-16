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
#include <cmath>

static __global__ void initialize_curand_states(curandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curand_init(seed, n, 0, &state[n]);
  }
}

static __global__ void update_moments(
  const int N,
  const float beta1,
  const float beta2,
  const float* __restrict__ gradients,
  float* __restrict__ m,
  float* __restrict__ v)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float g = gradients[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
  }
}

static __global__ void apply_updates(
  const int N,
  const float lr,
  const float beta1_t,
  const float beta2_t,
  const float eps,
  const float weight_decay,
  const float* __restrict__ m,
  const float* __restrict__ v,
  float* __restrict__ parameters)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    // calculate bias correction
    const float m_hat = m[i] / (1.0f - beta1_t);
    const float v_hat = v[i] / (1.0f - beta2_t);
    
    // update parameters
    parameters[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    parameters[i] -= lr * weight_decay * parameters[i];
  }
}

Adam::Adam(Parameters& para)
{
  // initialize the parameters
  step = 0;
  number_of_variables = para.number_of_variables;
  input_dim = para.dim;
  num_neurons1 = para.num_neurons1;
  number_of_variables_descriptor = para.number_of_variables_descriptor;
  weight_decay = para.lambda_2;

  // initialize the CPU vectors
  parameters.resize(number_of_variables);
  // gradients.resize(number_of_variables);
  m.resize(number_of_variables, 0.0f);
  v.resize(number_of_variables, 0.0f);

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
  initialize_parameters(para);
}

static __global__ void gpu_create_paramters(
  const int number_of_variables,
  curandState* g_state,
  float* g_parameters)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_variables) {
    curandState state = g_state[n];
    float r1 = curand_normal(&state);
    float r2 = curand_uniform(&state) - 0.5f;
    g_parameters[n] = 0.1f * r1 + r2;
  }
}

void Adam::initialize_parameters(Parameters& para)
{
    FILE* fid_restart = fopen("nep.restart", "r");
  if (fid_restart == NULL) {
    cudaSetDevice(0); // normally use GPU-0
    gpu_create_paramters<<<(number_of_variables - 1) / 128 + 1, 128>>>(
      number_of_variables,
      curand_states.data(),
      gpu_parameters.data());
    CUDA_CHECK_KERNEL
    gpu_parameters.copy_to_host(parameters.data());
    // parameters[0] = 1.292796015739f;
    // parameters[1] = 0.127129867673f;
    // parameters[2] = -0.939973294735f;
    // parameters[3] = 1.164981842041f;
    // parameters[4] = -0.251407504082f;
    // parameters[5] = 0.220846444368f;
    // parameters[6] = 0.137968629599f;
    // parameters[7] = 0.417271614075f;
    // parameters[8] = -1.099964499474f;
    // parameters[9] = -0.935172498226f;
    // parameters[10] = -0.386360764503f;
    // parameters[11] = 0.230891421437f;
    // parameters[12] = -0.555047988892f;
    // parameters[13] = -0.133708223701f;
    // parameters[14] = -0.538717210293f;
    // parameters[15] = 0.024417329580f;
    // parameters[16] = -0.559446573257f;
    // parameters[17] = 0.393686383963f;
    // parameters[18] = 1.184152245522f;
    // parameters[19] = -0.355633169413f;
    // parameters[20] = -0.174412131310f;
    // parameters[21] = 0.086221024394f;
    // parameters[22] = 0.196279138327f;
    // parameters[23] = 1.336585521698f;
    // parameters[24] = -0.298980683088f;
    // parameters[25] = -0.357109396509f; // c
    // parameters[26] = -0.382208127700f;
    // parameters[27] = 0.005223161488f;
    // parameters[28] = -0.279997548693f;
    // parameters[29] = -0.129192845901f;
    // parameters[30] = 0.311555036837f;
    // parameters[31] = 0.444509547752f;
    // parameters[32] = -0.308477275866f;
    // parameters[33] = 0.021276300562f;
    // parameters[34] = 0.334129622248f;
    // parameters[35] = -0.168401458536f;
    // parameters[36] = -0.102445381562f; //11
    // parameters[37] = -0.280292335715f;
    // parameters[38] = 0.592506792960f;
    // parameters[39] = -0.490981270350f;
    // parameters[40] = 0.423874505255f;
    // parameters[0] = 0.135209426284f;
    // parameters[1] = -0.511071622372f;
    // parameters[2] = 0.710385501385f;
    // parameters[3] = 0.319523602724f;
    // parameters[4] = 0.191613405943f;
    // parameters[5] = 0.087947189808f;
    // parameters[6] = -0.567227065563f;
    // parameters[7] = -0.279467076063f;
    // parameters[8] = 1.965929865837f;
    // parameters[9] = -1.091465830803f;
    // parameters[10] = -0.288163721561f;
    // parameters[11] = -0.166786357760f;
    // parameters[12] = 0.311797946692f;
    // parameters[13] = -0.869553208351f;
    // parameters[14] = -0.778700590134f;
    // parameters[15] = 0.530913829803f;
    // parameters[16] = 0.882725238800f;
    // parameters[17] = -0.284732282162f;
    // parameters[18] = 1.169425129890f;
    // parameters[19] = 0.684252560139f;
    // parameters[20] = 0.298993229866f;
    // parameters[21] = 0.664860904217f;
    // parameters[22] = 0.201725527644f;
    // parameters[23] = -0.128430858254f;
    // parameters[24] = -0.298980683088f;
    // parameters[25] = -0.363153636456f; // c[0,0,0,0],0
    // parameters[26] = -0.036242499948f;// c[0,1,0,0],1
    // parameters[27] = -0.711010098457f; // c[1,0,0,0],2
    // parameters[28] = -0.256476581097f;// c[1,1,0,0],3
    // parameters[29] = -0.229556530714f;// c[0,0,0,1],4
    // parameters[30] = 0.484168291092f;// c[0,1,0,1],5
    // parameters[31] = -0.549655020237f;// c[1,0,0,1],6
    // parameters[32] = 0.148251265287f;// c[1,1,0,1],7
    // parameters[33] = -0.326564490795f;// c[0,0,1,0],8
    // parameters[34] = -0.507190704346f;// c[0,1,1,0],9
    // parameters[35] = -0.018795832992f;// c[1,0,1,0],10
    // parameters[36] = 0.116535186768f;// c[1,1,1,0],11
    // parameters[37] = 0.362295359373f;// c[0,0,1,1],12
    // parameters[38] = 0.158546864986f;// c[0,1,1,1],13
    // parameters[39] = 0.073974691331f;// c[1,0,1,1],14
    // parameters[40] = 0.488178819418f;// c[1,1,1,1],15
    // parameters[41] = 1.0f;
    // parameters[42] = 1.0f;
    // parameters[43] = 1.0f;
    // parameters[44] = 1.0f;
    // gpu_parameters.copy_from_host(parameters.data());
  } else {
    for (int n = 0; n < number_of_variables; ++n) {
        int count = fscanf(fid_restart, "%f", &parameters[n]);
        PRINT_SCANF_ERROR(count, 2, "Reading error for nep.restart.");
    }
    fclose(fid_restart);
  }
}

void Adam::update(float lr, float* gradients) {
  const int block_size = 256;
  const int grid_size = (number_of_variables - 1) / block_size + 1;

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
  const float beta1_t = powf(beta1, step + 1);
  const float beta2_t = powf(beta2, step + 1);
  
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
  step++; 
}

// void Adam::zero_gradients()
// {
//   gpu_gradients.fill(0.0f);
// }

void Adam::output_parameters(Parameters& para) {
  cudaSetDevice(0); 
  gpu_parameters.copy_to_host(parameters.data());
  FILE* fid = fopen("nep.restart", "w");
  for (int i = 0; i < number_of_variables; ++i) {
    fprintf(fid, "%15.7e\n", parameters[i]);
  }
  fclose(fid);
}

float* Adam::get_parameters()
{
  return parameters.data();
}