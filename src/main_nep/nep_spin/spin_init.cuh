/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under GPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#pragma once
#include "spin_parameters.cuh"
#include <random>
namespace snes_spin {
void initialize_search(const Spin_Parameters&, std::mt19937&, std::vector<float>&, std::vector<float>&);
void assign_variable_types(const Spin_Parameters&, int, std::vector<int>&);
}
