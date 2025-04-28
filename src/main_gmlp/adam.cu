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
    parameters[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * parameters[i]);
  }
}

__global__ void adam_update(
    const int N,
    const float step_size,  // = lr * sqrt(1 - beta2^t)/(1 - beta1^t)
    const float beta1_power,// = beta1^t
    const float beta2_power,// = beta2^t
    const float eps,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ param)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float bc1 = 1.0f - beta1_power;
        float bc2 = 1.0f - beta2_power;

        float m_hat = m[i] / bc1;   
        float v_hat = v[i] / bc2;   

        // param[i] -= step_size * (m_hat / (sqrt(v_hat) + eps));
        param[i] = param[i] - step_size * (m_hat / (sqrt(v_hat) + eps));
    }
}

__global__ void compute_gradient_norm(const float* grad, int n, float* norm_result)
{
  __shared__ float shared_sum[1024];
  const int tid = threadIdx.x;
  shared_sum[tid] = 0.0f;
  
  for (int i = tid; i < n; i += blockDim.x) {
    shared_sum[tid] += grad[i] * grad[i];
  }
  __syncthreads();
  
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    *norm_result = shared_sum[0];
  }
}

__global__ void apply_gradient_clipping(float* grad, int n, float norm, float max_norm)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (norm > max_norm) {
    float scale = max_norm / norm;
    if (tid < n) {
      grad[tid] *= scale;
    }
  }
}

void clip_gradients(int step, float* d_grad, int size) 
{
  static float avg_norm = -1.0f; 

  float* d_norm;
  CHECK(cudaMalloc(&d_norm, sizeof(float)));
  CHECK(cudaMemset(d_norm, 0, sizeof(float)));
  
  // calculate the norm of the gradient
  int block_size = 1024;
  compute_gradient_norm<<<1, block_size>>>(d_grad, size, d_norm);
  
  // apply gradient clipping
  int grid_size = (size + block_size - 1) / block_size;
  float h_norm;
  CHECK(cudaMemcpy(&h_norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost));
  h_norm = sqrt(h_norm);
  
  if (avg_norm < 0) {
    avg_norm = h_norm;
  } else {
    avg_norm = 0.9f * avg_norm + 0.1f * h_norm;
  }
  
  const float alpha = 1.0f;
  const float MAX_CLIP_NORM = 10.0f; //modified by parameters?
  float max_norm = avg_norm * alpha;
  if (max_norm > MAX_CLIP_NORM) max_norm = MAX_CLIP_NORM;
  
  apply_gradient_clipping<<<grid_size, block_size>>>(d_grad, size, h_norm, max_norm);
  
  CHECK(cudaFree(d_norm));
}

Adam::Adam(Parameters& para)
{
  // initialize the parameters
  number_of_variables = para.number_of_variables;
  weight_decay = para.weight_decay;
  // beta1 = 0.8f;  
  // beta2 = 0.99f;
  // eps = 1e-5f;    

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
  // initialize_parameters(para);
}

static __global__ void gpu_create_paramters(
  const int dim,
  const int num_types,
  const int num_neurons1,
  const int number_of_variables_ann,
  const int number_of_variables,
  const float* energy_shift,
  curandState* g_state,
  float* g_parameters)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < number_of_variables_ann) {
    curandState state = g_state[n];
    int type_idx = (n * num_types) / number_of_variables_ann;
    int param_idx = n % ((dim + 2) * num_neurons1 + 1);
    if (param_idx < dim * num_neurons1) {
      // float std = 1.0f / sqrt(static_cast<float>(dim + num_neurons1));
      float std = sqrt(2.0f / (dim + num_neurons1));
      g_parameters[n] = curand_normal(&state) * std;
    } else if (param_idx < (dim + 1) * num_neurons1) {
      g_parameters[n] = 0.01f * curand_normal(&state);
    } else if (param_idx < (dim + 2) * num_neurons1) {
      // float std = 1.0f / sqrt(static_cast<float>(num_neurons1 + 1));
      float std = sqrt(2.0f / (num_neurons1 + 1));
      g_parameters[n] = curand_normal(&state) * std;
    } else {
      float mean = energy_shift[type_idx];
      g_parameters[n] = mean + 0.01f * curand_normal(&state);
    }
  } else if (n < number_of_variables) {
    curandState state = g_state[n];
    float r1 = curand_normal(&state);
    float r2 = curand_uniform(&state) - 0.5f;
    g_parameters[n] = 0.1f * r1 + r2;
  }
}

void Adam::initialize_parameters(Parameters& para)
{
  FILE* fid_restart = fopen("gmlp.restart", "r");
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
    // parameters[0] = -0.566794097424;
    // parameters[1] = 0.177166983485;
    // parameters[2] = -0.590433180332;
    // parameters[3] = -0.092442415655;
    // parameters[4] = -0.404380172491;
    // parameters[5] = -0.007976102643;
    // parameters[6] = -0.316405743361;
    // parameters[7] = -0.047861218452;
    // parameters[8] = -0.406308472157;
    // parameters[9] = -0.053448960185;
    // parameters[10] = 0.006087716669;
    // parameters[11] = -0.294198930264;
    // parameters[12] = 0.185069829226; 
    // parameters[13] = -0.152120366693;
    // parameters[14] = -0.650068521500;
    // parameters[15] = 0.157648652792;
    // parameters[16] = -0.192317202687;
    // parameters[17] = -0.055301215500;
    // parameters[18] = 0.086412981153;
    // parameters[19] = 0.362104058266;
    // parameters[20] = 0.145172387362;
    // parameters[21] = 0.134321197867;
    // parameters[22] = -0.593982517719;
    // parameters[23] = -0.161231666803;
    // parameters[24] = 0.038778539747;
    // parameters[25] = 0.233396857977; 
    // parameters[26] = -0.042355246842; 
    // parameters[27] = 0.362095177174; 
    // parameters[28] = 0.294732719660; 
    // parameters[29] = -0.126476362348; 
    // parameters[30] = -0.675600588322;  //bias0
    // parameters[31] = -1.115116715431; 
    // parameters[32] = 1.487264275551; 
    // parameters[33] = 0.045338567346; 
    // parameters[34] = -0.355737477541; 
    // parameters[35] = 0.517520904541; 
    // parameters[36] = -12.920810699463; //bias1
    // parameters[37] = -0.372985869646; 
    // parameters[38] = 0.022064467892; 
    // parameters[39] = -0.162381157279; 
    // parameters[40] = -0.207799911499; 
    // parameters[41] = 0.039595209062; 
    // parameters[42] = -0.184155151248; 
    // parameters[43] = -0.051093399525; 
    // parameters[44] = 0.245357260108; 
    // parameters[45] = 0.012506583706; 
    // parameters[46] = 0.526330411434; 
    // parameters[47] = 0.302395701408; 
    // parameters[48] = -0.570180118084; 
    // parameters[49] = 0.293095946312; 
    // parameters[50] = -0.477496147156; 
    // parameters[51] = 0.059975810349; 
    // parameters[52] = -0.043743103743; 
    // parameters[53] = -0.352179169655; 
    // parameters[54] = -0.361316055059; 
    // parameters[55] = -0.013380755670; 
    // parameters[56] = 0.052826214582; 
    // parameters[57] = 0.064544029534; 
    // parameters[58] = 0.172009438276; 
    // parameters[59] = -0.006703964435; 
    // parameters[60] = 0.055428165942; 
    // parameters[61] = 0.004689971916; 
    // parameters[62] = 0.437171846628; 
    // parameters[63] = 0.337618321180; 
    // parameters[64] = 0.166047856212; 
    // parameters[65] = -0.241618514061; 
    // parameters[66] = -0.502974987030; 
    // parameters[67] = 0.185325250030;  //bias0
    // parameters[68] = -0.853444159031; 
    // parameters[69] = 0.180195733905; 
    // parameters[70] = -0.271427333355; 
    // parameters[71] = -0.031502839178; 
    // parameters[72] = -0.174318954349; 
    // parameters[73] = -52.652294158936; //bias1
    // parameters[74] = -0.357109396509;  // c[0,0,0,0],0
    // parameters[75] = -0.382208127700;   // c[0,1,0,0],1
    // parameters[76] = 0.005223161488; // c[1,0,0,0],2
    // parameters[77] = -0.279997548693; // c[1,1,0,0],3
    // parameters[78] = -0.129192845901; // c[0,0,0,1],4
    // parameters[79] = 0.311555036837; // c[0,1,0,1],5
    // parameters[80] = 0.444509547752; // c[1,0,0,1],6
    // parameters[81] = -0.308477275866; // c[1,1,0,1],7
    // parameters[82] = 0.021276300562; // c[0,0,1,0],8
    // parameters[83] = 0.334129622248; // c[0,1,1,0],9
    // parameters[84] = -0.168401458536; // c[1,0,1,0],10
    // parameters[85] = -0.102445381562; // c[1,1,1,0],11
    // parameters[86] = -0.280292335715; // c[0,0,1,1],12
    // parameters[87] = 0.592506792960; // c[0,1,1,1],13
    // parameters[88] = -0.490981270350; // c[1,0,1,1],14
    // parameters[89] = 0.423874505255; // c[1,1,1,1],15
    // parameters[90] = -0.035936288214;
    // parameters[91] = -0.398841094248;
    // parameters[92] = -0.370980213146;
    // parameters[93] = 0.100899012854;
    // parameters[94] = 0.491831342258;
    // parameters[95] = -0.276368774732;
    // parameters[96] = 0.193875768358;
    // parameters[97] = -0.384473588174;
    // parameters[98] = -0.231054175005;
    // parameters[99] = -0.074071099228;
    // parameters[100] = 0.254499180397;
    // parameters[101] = 0.274881199087;
    // parameters[102] = 0.096353120873;
    // parameters[103] = -0.169466334807;
    // parameters[104] = -0.383005477527;
    // parameters[105] = 0.259637643865;
    // gpu_parameters.copy_from_host(parameters.data());
  } else {
    for (int n = 0; n < number_of_variables; ++n) {
        int count = fscanf(fid_restart, "%f", &parameters[n]);
        PRINT_SCANF_ERROR(count, 1, "Reading error for gmlp.restart.");
    }
    fclose(fid_restart);
    cudaSetDevice(0); // normally use GPU-0
    gpu_parameters.copy_from_host(parameters.data());
  }
}

void Adam::update(float lr, float* gradients) {
  const int block_size = 256;
  const int grid_size = (number_of_variables + block_size - 1) / block_size + 1;

  clip_gradients(step, gradients, number_of_variables);
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
  const float beta1_t = pow(beta1, step + 1);
  const float beta2_t = pow(beta2, step + 1);

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
  FILE* fid = fopen("gmlp.restart", "w");
  for (int i = 0; i < number_of_variables; ++i) {
    fprintf(fid, "%15.7e\n", parameters[i]);
  }
  fclose(fid);
}

float* Adam::get_parameters()
{
  return parameters.data();
}