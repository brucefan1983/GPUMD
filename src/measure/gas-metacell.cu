
#include "measure/gas-metacell.cuh"
#include "model/read_xyz.cuh"



TorchMetaCell::TorchMetaCell(void):TorchMetaCell("GASCVModel.pt","GAScfg.yaml"){}

TorchMetaCell::TorchMetaCell(std::string model_path,std::string cfg_path){
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
        config = MCConfig::fromFile(cfg_path);
        std::cout << "[GAS-Info] GASConfig loaded successfully from " << cfg_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the Config: "<< cfg_path << e.what() << std::endl;
        throw e;
    }

    torch_cv_traj = torch::empty({config.max_cv_nums,config.cv_size}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_now_cvs = torch::empty({config.cv_size},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_bias = torch::empty({},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    cpu_b_vector = std::vector<double>(9); // Box
    torch::cuda::synchronize();

}


torch::Dict<std::string, torch::Tensor> TorchMetaCell::predict(
    const torch::Dict<std::string, torch::Tensor>& inputs) {
        auto result = model.forward({inputs}).toGenericDict();
        torch::Dict<std::string, torch::Tensor> outputs;
        for (const auto& item : result) {
            auto key = item.key().toStringRef();
            auto value = item.value().toTensor();
            #ifdef USE_GAS_DEBUG
            std::cout<<key<<value<<std::endl;
            #endif
            outputs.insert(key, value);
        }
        return outputs;
}

void TorchMetaCell::process(Box& box,GPU_Vector<double>& virial){

    this->box_to_tri(box);
    torch::cuda::synchronize();
    torch::Tensor torch_cell = torch::from_blob(cpu_b_vector.data(), {9}, torch::dtype(torch::kFloat64)).to(torch::kCUDA).reshape({3,3});
    torch::Dict<std::string, torch::Tensor> inputs;
    inputs.insert("cell", torch_cell);
    inputs.insert("cv_traj",torch_cv_traj);
    inputs.insert("saved_cv_nums",torch::tensor(saved_cv_nums,torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)));
    torch::cuda::synchronize();
    //计算和取出输出
    auto output_dict = this->predict(inputs);
    torch::cuda::synchronize();
    torch_now_cvs = output_dict.at("cell").reshape({-1});
    torch_bias = output_dict.at("bias");
    torch::cuda::synchronize();
    if(now_step>0 && now_step%config.cv_change_interval==0){
      torch::Tensor torch_virial  = _FromCudaMemory(virial.data(),virial.size()).reshape({9,-1}).transpose(0,1);//(xx yy zz xy xz yz yx zx zy)
      torch::Tensor tot_virial = torch_virial.sum(0);//.index_select(0,torch::tensor([0,4,8,1,2,5,3,6,7],torch::kCUDA)).reshape({3,3});
      tot_virial = tot_virial.index_select(0,torch::tensor({0,3,4,6,1,5,7,8,2},torch::kCUDA));
      tot_virial = tot_virial.reshape({3,3});
      torch::Tensor volume = torch::linalg::det(torch_cell);
      auto cell_force = output_dict.at("cell_force");//-db/dL
      auto out_pressure = output_dict.at("pressure");
    //   std::cout<<torch::matmul(cell_rotinv,tot_virial)<<std::endl;
      auto tot_cell_force = (tot_virial/volume)-out_pressure+cell_force;//(xx yy zz xy xz yz yx zx zy)
      std::cout<<tot_cell_force<<std::endl;
      auto delta_L = output_dict.at("delta_L");
      auto new_cell = torch_cell+(delta_L*tot_cell_force/tot_cell_force.norm());
      this->tri_to_box(box,new_cell);
    }
    if(now_step%config.cv_log_interval==0){
      this->logCV_runtime();
    }
    if(now_step%config.cv_storage_interval==0){
      this->appendCVtoTraj();
    }
    now_step++;
}

// 将cuda指针用torch视图读取的代码
torch::Tensor TorchMetaCell::_FromCudaMemory(double* d_array, int size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    return torch::from_blob(d_array, {size}, options);
}
//如其名
void TorchMetaCell::appendCVtoTraj(){
    if(saved_cv_nums<config.max_cv_nums){
        torch_cv_traj[saved_cv_nums]=torch_now_cvs.clone();
        saved_cv_nums++;
    }
    else{
        printf("cv_num index out of bounds, may check your code or enlarge your memory set.\n");
    }
}

void TorchMetaCell::logCV_runtime(void){
    auto log_cv_tensor = torch_now_cvs;
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

void TorchMetaCell::box_to_tri(Box& box){
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

void TorchMetaCell::tri_to_box(Box& box,torch::Tensor torch_cell){
  auto cpu_cell = torch_cell.to(torch::kCPU);
  auto cpu_inverse = torch::inverse(cpu_cell);
  auto cell_data = cpu_cell.accessor<double, 2>();
  auto cellinv_data = cpu_inverse.accessor<double, 2>();
  if (box.triclinic == 0) {
    box.cpu_h[0]=cell_data[0][0];
    box.cpu_h[1]=cell_data[1][1];
    box.cpu_h[2]=cell_data[2][2];
    box.cpu_h[9] =cellinv_data[0][0];
    box.cpu_h[10]=cellinv_data[1][1];
    box.cpu_h[11]=cellinv_data[2][2];
  } else {
    box.cpu_h[0]=cell_data[0][0];
    box.cpu_h[1]=cell_data[1][0];
    box.cpu_h[2]=cell_data[2][0];
    box.cpu_h[3]=cell_data[0][1];
    box.cpu_h[4]=cell_data[1][1];
    box.cpu_h[5]=cell_data[2][1];
    box.cpu_h[6]=cell_data[0][2];
    box.cpu_h[7]=cell_data[1][2];
    box.cpu_h[8]=cell_data[2][2];

    box.cpu_h[9]= cellinv_data[0][0];
    box.cpu_h[10]=cellinv_data[1][0];
    box.cpu_h[11]=cellinv_data[2][0];
    box.cpu_h[12]=cellinv_data[0][1];
    box.cpu_h[13]=cellinv_data[1][1];
    box.cpu_h[14]=cellinv_data[2][1];
    box.cpu_h[15]=cellinv_data[0][2];
    box.cpu_h[16]=cellinv_data[1][2];
    box.cpu_h[17]=cellinv_data[2][2];
  }
}