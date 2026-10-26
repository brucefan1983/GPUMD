/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include <string>

// Normalize the public block format to the counted Spin3 input consumed by
// the existing semantic validators. Legacy files keep their parameter order.
bool is_nep_spin_model_file(const std::string& filename);
std::string read_nep_spin_model_input(const std::string& filename);
