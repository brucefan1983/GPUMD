#pragma once
// Test-only inspection of model-owned buffers; do not expose these through Potential.
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#define private public
#include "main_nep/nep_spin/nep_spin.cuh"
#include "main_nep/nep/nep.cuh"
#undef private
#include "main_nep/parameter_input.cuh"
