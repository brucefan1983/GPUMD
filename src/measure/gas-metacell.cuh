
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

// 定义一个结构体用于存储配置数据
struct MCConfig {
    int max_cv_nums;
    int cv_size;
    int cv_storage_interval;
    int cv_change_interval;
    int cv_log_interval;
    double neighbor_rc;
    int max_neighbors;
    int n_atoms;
    bool is_opt_=0;
    bool is_obs_=0;
    

    // 打印结构体内容
    void print() const {
        std::cout << "Max CV Nums: " << max_cv_nums << std::endl;
        std::cout << "CV Size: " << cv_size << std::endl;
        std::cout << "CV Storage Interval: " << cv_storage_interval << std::endl;
        std::cout << "CV Change Interval: " << cv_change_interval << std::endl;
        std::cout << "CV Log Interval: " << cv_log_interval << std::endl;
        // std::cout << "Neighbor RC: " << neighbor_rc << std::endl;
        // std::cout << "Max Neighbors: " << max_neighbors << std::endl;
        // std::cout << "Number of Atoms: " << n_atoms << std::endl;
    }

    // 从文件初始化结构体的静态方法
    static MCConfig fromFile(const std::string& filePath) {
        std::unordered_map<std::string, std::string> configMap;
        std::ifstream file(filePath);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::string line;
        while (std::getline(file, line)) {
            // 跳过空行和注释行
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // 查找冒号分隔符
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) {
                continue; // 跳过不符合格式的行
            }

            // 提取键和值，并去掉多余的空格
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            // 修剪键和值的空白字符
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // 存储到哈希表
            configMap[key] = value;
        }

        file.close();

        // 创建并初始化结构体实例
        MCConfig config;
        try {
            // values
            config.max_cv_nums = std::stoi(configMap.at("max_cv_nums"));
            config.cv_size = std::stoi(configMap.at("cv_size"));
            config.cv_storage_interval = std::stoi(configMap.at("cv_storage_interval"));
            config.cv_change_interval = std::stoi(configMap.at("cv_change_interval"));
            config.cv_log_interval = std::stoi(configMap.at("cv_log_interval"));
            // config.neighbor_rc = std::stod(configMap.at("neighbor_rc"));
            // config.max_neighbors = std::stoi(configMap.at("max_neighbors"));
            // config.n_atoms = std::stoi(configMap.at("n_atoms"));

            // flag
            // config.is_opt_ = (configMap.find("MetaCell")!=configMap.end());
            // config.is_obs_ = (configMap.find("Observe")!=configMap.end());


        } catch (const std::exception& ex) {
            throw std::runtime_error("Error parsing configuration: " + std::string(ex.what()));
        }



        return config;
    }
};



struct TorchMetaCell
{
public:

    TorchMetaCell(std::string model_path,std::string cfg_path);
    TorchMetaCell(void);

    void process(Box& box,GPU_Vector<double>& virial);

    torch::Dict<std::string, torch::Tensor> predict(const torch::Dict<std::string, torch::Tensor>& inputs);
    torch::Tensor _FromCudaMemory(double* d_array, int size);
    void box_to_tri(Box& box);
    void logCV_runtime(void);
    void appendCVtoTraj(void);
    void tri_to_box(Box& box,torch::Tensor torch_cell);

    static std::unique_ptr<TorchMetaCell> parse_GASMETACELL(const char** param, int num_param, const int number_of_atoms) {
       if(num_param==1)
       {
        return std::make_unique<TorchMetaCell>();
       }
       else if (num_param==2)
       {
        throw std::runtime_error("Error parsing GASMetaCell: params shapes like \"GASMD model.pt cfg.yaml\", but found "+ std::string(param[0])+std::string(param[1]));
       }
       else if (num_param>=3)
       {
        std::string model_path = param[1];
        std::string cfg_path = param[2];
        return std::make_unique<TorchMetaCell>(model_path,cfg_path);
       }
       else{
        throw std::runtime_error("Error parsing GASMetaCell: params shapes like \"GASMD model.pt cfg.yaml\"");
       }
       
    }

    MCConfig config;

private:
    int saved_cv_nums=0;
    int now_step = 0;
    torch::Tensor torch_now_cvs;
    torch::Tensor torch_cv_traj;
    torch::Tensor torch_bias;
    // 存 cell 参数
    std::vector<double> cpu_b_vector; // Box
    // torch-model
    torch::jit::script::Module model; // TorchScript 模型
};
