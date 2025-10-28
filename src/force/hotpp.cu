#ifdef USE_TORCH
#include "hotpp.cuh"
#include "model/read_xyz.cuh"

static __global__ void find_neighbor_list_large_box_hotpp(
  const float neighbor_rc,
  const int N,
  const int N1,
  const int N2,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  // const int* g_type,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int count_radial = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f / neighbor_rc,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
        if (cell_id_x + xx < 0)
          neighbor_cell += nx;
        if (cell_id_x + xx >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + yy < 0)
          neighbor_cell += ny * nx;
        if (cell_id_y + yy >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + zz < 0)
          neighbor_cell += nz * ny * nx;
        if (cell_id_z + zz >= nz)
          neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n2 < N1 || n2 >= N2 || n1 == n2) {
            continue;
          }

          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
          float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
          
          float rc_radial = neighbor_rc;

          if (d12_square >= rc_radial * rc_radial) {
            continue;
          }
          g_NL_radial[count_radial++ * N + n1] = n2;
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
}


void Hotpp::compute_large_box(
  Box& box,
  const GPU_Vector<double>& position_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = n_atoms_;
  //N1~N2是要计算的原子的序号
//   const int grid_size = (this->N2_ - this->N1_ - 1) / BLOCK_SIZE + 1;
  const int grid_size = (N - 1) / BLOCK_SIZE + 1;

  const double rc_cell_list = 0.5 * config.neighbor_rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list,
    num_bins,
    box,
    position_per_atom,
    this->cell_count,
    this->cell_count_sum,
    this->cell_contents);

  find_neighbor_list_large_box_hotpp<<<grid_size, BLOCK_SIZE>>>(
    config.neighbor_rc,
    N,
    0,// this->N1_,
    N,// this->N2_,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    box,
    this->cell_count.data(),
    this->cell_count_sum.data(),
    this->cell_contents.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    this->NN_radial.data(),
    this->NL_radial.data());
    torch::cuda::synchronize();
}

void Hotpp::get_neighbor_list(Box& box,const GPU_Vector<double>& position_per_atom){
  this->NN_radial.fill(-1);
  this->NL_radial.fill(-1);
  this->compute_large_box(box,position_per_atom);
}

Hotpp::Hotpp(std::string model_path,int n_atoms){
    this->n_atoms_ = n_atoms;
    // 读取文件，设定参数
    try {
        // torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // 加载 TorchScript 模型
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval(); // 设置为评估模式
        std::cout << "[hotpp-Info] hotpp loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: "<< model_path << e.what() << std::endl;
        throw e;
    }


    cell_count.resize(n_atoms_);
    cell_count_sum.resize(n_atoms_);
    cell_contents.resize(n_atoms_);
    NN_radial.resize(n_atoms_);
    NL_radial.resize(n_atoms_*config.max_neighbors);

    cpu_b_vector = std::vector<double>(9); // Box
    // gpu_v_vector.resize(6);
    // gpu_v_factor.resize(9);
    torch::cuda::synchronize();

}


torch::Dict<std::string, torch::Tensor> Hotpp::predict(
    const torch::Dict<std::string, torch::Tensor>& inputs) {
    // try {
        // 将输入传递给模型
        auto result = model.forward({inputs}).toGenericDict();
        // 要花括号吗？
        // 转换返回值为 torch::Dict
        torch::Dict<std::string, torch::Tensor> outputs;
        for (const auto& item : result) {
            auto key = item.key().toStringRef();
            auto value = item.value().toTensor();
            // #endif
            outputs.insert(key, value);
        }
        return outputs;
    // } catch (const c10::Error& e) {
    //     std::cerr << "推理失败: " << e.what() << std::endl;
    //     throw;
    // }
}

void Hotpp::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& positions,
  GPU_Vector<double>& potentials,
  GPU_Vector<double>& forces,
  GPU_Vector<double>& virial)
{
    int dynamic_vector_size = positions.size();
    int n_atoms = dynamic_vector_size/3;

    this->box_to_tri(box);
    this->get_neighbor_list(box,positions);
    torch::cuda::synchronize();
    //下面的两个量在 gpumd 中为列主序，因此调用transpose
    //保留力和位力的引用，生成坐标和晶格的副本
    torch::Tensor torch_pos = _FromCudaMemory((double*)positions.data(),dynamic_vector_size).detach().clone().reshape({3,-1}).transpose(0,1);
    torch::Tensor torch_force = _FromCudaMemory(forces.data(),dynamic_vector_size).reshape({3,-1}).transpose(0,1);
    torch::Tensor torch_potential =_FromCudaMemory(potentials.data(),n_atoms);

    // cell 相关的量
    // torch::Tensor torch_virial = _FromCudaMemory(gpu_v_vector.data(),6).reshape({-1});//(0 4 8 1 2 5) 顺序
    torch::Tensor torch_virial  = _FromCudaMemory(virial.data(),dynamic_vector_size*3).reshape({9,-1}).transpose(0,1);//(xx yy zz xy xz yz yx zx zy)
    torch::Tensor torch_cell = torch::from_blob(cpu_b_vector.data(), {9}, torch::dtype(torch::kFloat64)).to(torch::kCUDA).reshape({3,3});
    torch::Tensor torch_type = torch::from_blob(type.data(), {n_atoms}, torch::dtype(torch::kInt)).to(torch::kCUDA).reshape(-1);
    // 近邻组
    torch_NN = torch::from_blob(NN_radial.data(), {n_atoms}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)).reshape(-1);
    torch_NL = torch::from_blob(NL_radial.data(), {n_atoms*config.max_neighbors}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)).reshape(-1);
    torch::cuda::synchronize();
    // torch::Tensor side_array = NL2Indices(torch_NL,n_atoms);
    int NL_size = torch_NL.size(0);
    int max_neighbor = NL_size/n_atoms;
    torch::Tensor temp = torch::arange(0, n_atoms_NL,torch::kCUDA);
    torch::Tensor start_array = temp.repeat({max_neighbor});
    torch::Tensor mask = torch_NL >= 0;

    start_array = start_array.masked_select(mask);
    torch::Tensor end_array = torch_NL.masked_select(mask);

    std::vector<long> batch(n_atoms,0);
    auto batch_ = torch::from_blob(batch.data(), {n_atoms}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));

    torch::Dict<std::string, torch::Tensor> inputs;

    inputs.insert("coordinate", torch_pos);
    inputs.insert("atomic_number", torch_type);
    inputs.insert("idx_i",start_array);
    inputs.insert("idx_j", end_array);
    inputs.insert("batch",batch_);
    inputs.insert("ghost_neigh",torch::empty({0},torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)));
    torch::cuda::synchronize();
    //计算和取出输出
    auto output_dict = this->predict(inputs);
    torch::cuda::synchronize();
   
    torch_bias = output_dict.at("energy_p").to(torch::kDouble);
    torch_potential+=torch_bias;
    torch_force+=output_dict.at("forces_p").to(torch::kDouble);
    auto cell_virial = output_dict.at("virial_p").to(torch::kDouble);
    torch_virial+=cell_virial;
    now_step++;
}

// 将cuda指针用torch视图读取的代码
torch::Tensor Hotpp::_FromCudaMemory(double* d_array, int size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    return torch::from_blob(d_array, {size}, options);
}

void Hotpp::box_to_tri(Box& box){
        if (box.triclinic == 0) {
        cpu_b_vector[0] = box.cpu_h[0];
        cpu_b_vector[1] = 0.0;
        cpu_b_vector[2] = 0.0;
        cpu_b_vector[3] = 0.0;
        cpu_b_vector[4] = box.cpu_h[1];
        cpu_b_vector[5] = 0.0;
        cpu_b_vector[6] = 0.0;
        cpu_b_vector[7] = 0.0;
        cpu_b_vector[8] = box.cpu_h[2];
      } else {
        cpu_b_vector[0] = box.cpu_h[0];
        cpu_b_vector[1] = box.cpu_h[3];
        cpu_b_vector[2] = box.cpu_h[6];
        cpu_b_vector[3] = box.cpu_h[1];
        cpu_b_vector[4] = box.cpu_h[4];
        cpu_b_vector[5] = box.cpu_h[7];
        cpu_b_vector[6] = box.cpu_h[2];
        cpu_b_vector[7] = box.cpu_h[5];
        cpu_b_vector[8] = box.cpu_h[8];
      }
}
#endif