#ifdef USE_GAS
#include "gas-metad.cuh"
#include "model/read_xyz.cuh"

namespace{
  std::vector<std::vector<double>> readFileToVector(const std::string& filename) {
    std::ifstream file(filename);  // 打开文件
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        throw std::runtime_error("无法打开文件");
    }

    std::vector<std::vector<double>> data;  // 存储每一行的浮点数
    std::string line;

    // 逐行读取文件
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;

        // 逐个浮点数读取并存入当前行的vector
        while (ss >> value) {
            row.push_back(value);
        }

        data.push_back(row);  // 将当前行添加到data中
    }

    file.close();  // 关闭文件
    return data;
}


torch::Tensor vectorToTensor(const std::vector<std::vector<double>>& data) {
    // 获取数据的维度
    auto m = data.size();           // 行数
    auto n = data.empty() ? 0 : data[0].size();  // 列数

    // 将二维std::vector平展为一维std::vector
    std::vector<double> flat_data;
    for (const auto& row : data) {
        flat_data.insert(flat_data.end(), row.begin(), row.end());
    }

    // 创建张量 (m, n)，数据类型为 double
    torch::Tensor tensor = torch::from_blob(flat_data.data(), {(long)m, (long)n}, torch::kFloat64);

    return tensor.clone();  // 返回一个副本，确保数据不受外部修改的影响
}

}



static __global__ void find_neighbor_list_large_box_gas(
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

torch::Tensor NL2Indices(torch::Tensor& torch_NL,int n_atoms_NL){
    int NL_size = torch_NL.size(0);
    int max_neighbor = NL_size/n_atoms_NL;
    torch::Tensor temp = torch::arange(0, n_atoms_NL,torch::kCUDA);
    torch::Tensor start_array = temp.repeat({max_neighbor});
    torch::Tensor mask = torch_NL >= 0;

    start_array = start_array.masked_select(mask);
    torch::Tensor end_array = torch_NL.masked_select(mask);
    torch::Tensor side_array = torch::stack({start_array, end_array}, 1);
    return side_array;
}

// 定义一个函数来对称化应力张量
torch::Tensor symmetrizeStressTensor(const torch::Tensor& stressTensor) {
    TORCH_CHECK(stressTensor.sizes() == std::vector<int64_t>({3, 3}), "Input must be a 3x3 matrix.");
    torch::Tensor symStressTensor = 0.5 * (stressTensor + stressTensor.transpose(0, 1));
    return symStressTensor;
}

// 定义一个函数来转换应力张量
torch::Tensor convertStressTensor(const torch::Tensor& stressTensor) {
    TORCH_CHECK(stressTensor.sizes() == std::vector<int64_t>({3, 3}), "Input must be a 3x3 matrix.");
    torch::Tensor symmetricStress = torch::empty({6}, stressTensor.options());
    symmetricStress[0] = stressTensor[0][0]; // σ_xx
    symmetricStress[1] = stressTensor[1][1]; // σ_yy
    symmetricStress[2] = stressTensor[2][2]; // σ_zz
    symmetricStress[3] = stressTensor[1][2]; // σ_yz
    symmetricStress[4] = stressTensor[0][2]; // σ_xz
    symmetricStress[5] = stressTensor[0][1]; // σ_xy
    return symmetricStress;
}

void TorchMetad::compute_large_box(
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

  find_neighbor_list_large_box_gas<<<grid_size, BLOCK_SIZE>>>(
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

void TorchMetad::get_neighbor_list(Box& box,const GPU_Vector<double>& position_per_atom){
  this->NN_radial.fill(-1);
  this->NL_radial.fill(-1);
  this->compute_large_box(box,position_per_atom);
}
TorchMetad::TorchMetad(int n_atoms):TorchMetad("GASCVModel.pt","GAScfg.yaml",n_atoms){}

TorchMetad::TorchMetad(std::string model_path,std::string cfg_path,int n_atoms){
    this->n_atoms_ = n_atoms;
    // 读取文件，设定参数
    try {
        // torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // 加载 TorchScript 模型
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval(); // 设置为评估模式
        std::cout << "[GAS-Info] GASCVModel loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: "<< model_path << e.what() << std::endl;
        throw e;
    }
    // 接受 GASConfig 参数
    try {
        config = Config::fromFile(cfg_path);
        std::cout << "[GAS-Info] GASConfig loaded successfully from " << cfg_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the Config: "<< cfg_path << e.what() << std::endl;
        throw e;
    }

    cell_count.resize(n_atoms_);
    cell_count_sum.resize(n_atoms_);
    cell_contents.resize(n_atoms_);
    NN_radial.resize(n_atoms_);
    NL_radial.resize(n_atoms_*config.max_neighbors);

    torch_cv_traj = torch::empty({config.max_cv_nums,config.cv_size}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_now_cvs = torch::empty({config.cv_size},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_delta_cv_save = torch::empty({config.cv_size},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_bias = torch::empty({},  torch::dtype(torch::kFloat64).device(torch::kCUDA));

    cpu_b_vector = std::vector<double>(9); // Box
    // gpu_v_vector.resize(6);
    // gpu_v_factor.resize(9);
    torch::cuda::synchronize();

}

TorchMetad::TorchMetad(std::string model_path,std::string cfg_path,std::string gaussian_path,int n_atoms){
    this->n_atoms_ = n_atoms;
    // 读取文件，设定参数
    try {
        // torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // 加载 TorchScript 模型
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval(); // 设置为评估模式
        std::cout << "[GAS-Info] GASCVModel loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: "<< model_path << e.what() << std::endl;
        throw e;
    }
    // 接受 GASConfig 参数
    try {
        config = Config::fromFile(cfg_path);
        std::cout << "[GAS-Info] GASConfig loaded successfully from " << cfg_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the Config: "<< cfg_path << e.what() << std::endl;
        throw e;
    }

    cell_count.resize(n_atoms_);
    cell_count_sum.resize(n_atoms_);
    cell_contents.resize(n_atoms_);
    NN_radial.resize(n_atoms_);
    NL_radial.resize(n_atoms_*config.max_neighbors);
    
    //读取高斯峰历史
    auto history = readFileToVector(gaussian_path);
    auto torch_history = vectorToTensor(history).to(torch::kCUDA);
    torch_history = torch_history.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch_history.size(1))});
    int history_num = torch_history.size(0);
    saved_cv_nums = history_num;

    torch_cv_traj = torch::empty({config.max_cv_nums,config.cv_size}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_now_cvs = torch::empty({config.cv_size},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_delta_cv_save = torch::empty({config.cv_size},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_bias = torch::empty({},  torch::dtype(torch::kFloat64).device(torch::kCUDA));

    torch_cv_traj.index({torch::indexing::Slice(0,history_num), torch::indexing::Slice()})= torch_history;
    cpu_b_vector = std::vector<double>(9); // Box
    // gpu_v_vector.resize(6);
    // gpu_v_factor.resize(9);
    torch::cuda::synchronize();

}


torch::Dict<std::string, torch::Tensor> TorchMetad::predict(
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
            // #ifdef USE_GAS_DEBUG
            if(key.compare("side_array") && key.compare("cv_traj") && key.compare("")){std::cout<<key<<value<<std::endl;}
            // #endif
            outputs.insert(key, value);
        }
        return outputs;
    // } catch (const c10::Error& e) {
    //     std::cerr << "推理失败: " << e.what() << std::endl;
    //     throw;
    // }
}

void TorchMetad::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& positions,
  GPU_Vector<double>& potentials,
  GPU_Vector<double>& forces,
  GPU_Vector<double>& virial)
{
    int dynamic_vector_size = positions.size();
    int n_atoms = dynamic_vector_size/3;
    if(n_atoms_!=n_atoms){
        throw std::invalid_argument("n_atoms in GASMD is inconsistent with the one in model.xyz.");
    }
    this->box_to_tri(box);
    this->get_neighbor_list(box,positions);
    // gpu_sum<<<6, 1024>>>(n_atoms, virial.data(), gpu_v_vector.data());
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
    // 近邻组
    torch_NN = torch::from_blob(NN_radial.data(), {n_atoms}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)).reshape(-1);
    torch_NL = torch::from_blob(NL_radial.data(), {n_atoms*config.max_neighbors}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)).reshape(-1);
    torch::cuda::synchronize();
    torch::Tensor side_array = NL2Indices(torch_NL,n_atoms);

    torch::Dict<std::string, torch::Tensor> inputs;

    inputs.insert("positions", torch_pos);
    inputs.insert("cell", torch_cell);
    inputs.insert("side_array",side_array);
    inputs.insert("cv_traj",torch_cv_traj);
    inputs.insert("saved_cv_nums",torch::tensor(saved_cv_nums,torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)));
    torch::cuda::synchronize();
    //计算和取出输出
    auto output_dict = this->predict(inputs);
    torch::cuda::synchronize();
    torch_now_cvs = output_dict.at("cv_now");
    bool is_opt = output_dict.contains("delta_cv_save");
    bool is_obs = output_dict.contains("cv_obs");
    if (is_opt) {torch_delta_cv_save = output_dict.at("delta_cv_save");} 
    torch_bias = output_dict.at("bias");
    torch_potential+=(torch_bias.unsqueeze(0)/n_atoms);
    torch_force+=output_dict.at("forces");
    mean_bias_force = output_dict.at("forces").norm(2,1).mean().clone().detach();
    auto cell_virial = output_dict.at("virial");//reschedule to xx yy zz xy yz xz yx zx zy
    torch_virial+=(cell_virial.unsqueeze(0)/n_atoms);
    torch::cuda::synchronize();
    if(now_step>0 && now_step%config.cv_storage_interval==0){
      this->appendCVtoTraj(is_opt);
      this->logCV_runtime(gaussian_name);
    }
    if(now_step%config.cv_log_interval==0){
      this->logCV_runtime();
    }
    now_step++;
}

// 将cuda指针用torch视图读取的代码
torch::Tensor TorchMetad::_FromCudaMemory(double* d_array, int size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    return torch::from_blob(d_array, {size}, options);
}
//如其名
void TorchMetad::appendCVtoTraj(bool is_opt){
  if(is_opt){
    if(saved_cv_nums==0){
        torch_cv_traj[saved_cv_nums]=torch_now_cvs.clone();
        saved_cv_nums++;
    }
    else if(saved_cv_nums>0 && saved_cv_nums<config.max_cv_nums){
        auto cv_last = torch_cv_traj[saved_cv_nums-1].clone();
        cv_last+=torch_delta_cv_save;
        torch_cv_traj[saved_cv_nums]=cv_last.clone();
        saved_cv_nums++;
        auto cpu_cv_storage = cv_last.to(torch::kCPU).reshape({-1});
        auto cpu_cv_data = cpu_cv_storage.accessor<double, 1>();

        std::cout<<"[GAS-Info] MetaOpt Add ND at  ";
        for (int j = 0; j < cpu_cv_storage.size(0); ++j) {
            std::cout << cpu_cv_data[j];
                if (j < cpu_cv_storage.size(0) - 1) {
                    std::cout << "\t";  // 使用 tab 分隔列
                }
        }
        std::cout<<std::endl;
    }
    else{
        printf("cv_num index out of bounds, may check your code or enlarge your memory set.\n");
    }
  }
  else{
    // normal version
    if(saved_cv_nums<config.max_cv_nums){
        torch_cv_traj[saved_cv_nums]=torch_now_cvs.clone();
        saved_cv_nums++;
    }
    else{
        printf("cv_num index out of bounds, may check your code or enlarge your memory set.\n");
    }
  }
}

void TorchMetad::logCV_runtime(void){
    auto log_cv_tensor = torch::cat({torch_now_cvs,mean_bias_force.unsqueeze(-1)},0);
    torch::Tensor cv_cpu_tensor = log_cv_tensor.to(torch::kCPU);
    torch::Tensor bias_cpu_tensor = this->torch_bias.to(torch::kCPU);
    torch::Tensor cpu_tensor = torch::cat({bias_cpu_tensor.squeeze(0).reshape({-1}),cv_cpu_tensor},0).reshape({-1});

    // 创建一个 TensorAccessor 对象来访问张量数据
    auto tensor_data = cpu_tensor.accessor<double, 1>();
    // 打开文件
    std::string log_name = "GASCVlog.txt";

    std::ofstream file(log_name,std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " <<log_name<< std::endl;
        return;
    }
    // 写入张量数据
    for (int j = 0; j < cpu_tensor.size(0); ++j) {
                file << tensor_data[j];
                if (j < cpu_tensor.size(0) - 1) {
                    file << "\t\t";  // 使用 tab 分隔列
                }
            }
            file << "\n";  // 换行分隔行

    // 关闭文件
    file.close();
}
void TorchMetad::logCV_runtime(std::string& path){
    auto log_cv_tensor = torch_now_cvs;
    torch::Tensor cv_cpu_tensor = log_cv_tensor.to(torch::kCPU);
    torch::Tensor bias_cpu_tensor = this->torch_bias.to(torch::kCPU);
    torch::Tensor cpu_tensor = torch::cat({bias_cpu_tensor.squeeze(0).reshape({-1}),cv_cpu_tensor},0).reshape({-1});

    // 创建一个 TensorAccessor 对象来访问张量数据
    auto tensor_data = cpu_tensor.accessor<double, 1>();
    // 打开文件
    std::string log_name = path;

    std::ofstream file(log_name,std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " <<log_name<< std::endl;
        return;
    }
    // 写入张量数据
    for (int j = 0; j < cpu_tensor.size(0); ++j) {
                file << tensor_data[j];
                if (j < cpu_tensor.size(0) - 1) {
                    file << "\t\t";  // 使用 tab 分隔列
                }
            }
            file << "\n";  // 换行分隔行

    // 关闭文件
    file.close();
}



void TorchMetad::box_to_tri(Box& box){
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