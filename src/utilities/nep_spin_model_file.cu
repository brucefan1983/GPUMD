/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#include "nep_spin_model_file.cuh"
#include "nep_file.cuh"
#include "nep_spin_legacy_parameters.cuh"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cctype>

namespace {
NEP_File read_document(const std::string& filename)
{
  std::ifstream input(filename);
  NEP_File document;
  std::string error;
  if (!document.read(input, error)) throw std::runtime_error(filename + ": " + error);
  return document;
}
std::string first_token(const std::string& filename)
{
  std::ifstream input(filename);
  std::string token;
  if (!(input >> token)) throw std::runtime_error("Cannot read model " + filename);
  for (char& c : token) c = std::tolower(static_cast<unsigned char>(c));
  return token;
}
}

bool is_nep_spin_model_file(const std::string& filename)
{
  const auto token = first_token(filename);
  if (token == "nep4_spin3" || token == "nep4_spin3_zbl") return true;
  if (token != "nep") return false;
  const auto document = read_document(filename);
  const auto& model = document.find_field("model")->values[0];
  return model == "nep_spin" || model == "nep_spin3";
}

std::string read_nep_spin_model_input(const std::string& filename)
{
  if (first_token(filename) != "nep") {
    std::ifstream input(filename);
    return std::string(std::istreambuf_iterator<char>(input), {});
  }
  const auto doc = read_document(filename);
  const auto& model = doc.find_field("model")->values[0];
  if (model != "nep_spin" && model != "nep_spin3")
    throw std::runtime_error("Expected model nep_spin in " + filename);
  std::vector<std::string> fields = {"format_version", "model"};
  auto field = [&](const char* name, size_t lo, size_t hi) -> const NEP_File::Field& {
    const auto* f = doc.find_field(name);
    if (!f || f->values.size() < lo || f->values.size() > hi)
      throw std::runtime_error(std::string("Invalid Spin3 field ") + name);
    fields.push_back(name);
    return *f;
  };
  auto integer = [&](const NEP_File::Field& f, size_t i, size_t hi) -> int {
    size_t value;
    if (!NEP_File::parse_size(f.values.at(i), value) || value > hi)
      throw std::runtime_error("Invalid integer in Spin3 field " + f.name);
    return int(value);
  };
  std::ostringstream out;
  out << std::setprecision(17);
  const bool zbl = doc.find_field("zbl") != nullptr;
  const int types = doc.elements.size();
  out << (zbl ? "nep4_spin3_zbl" : "nep4_spin3") << ' ' << types;
  for (const auto& element : doc.elements) out << ' ' << element;
  out << '\n';
  if (model == "nep_spin") {
    const auto& version = field("spin_version", 1, 1);
    if (version.values[0] != "1")
      throw std::runtime_error("Unsupported spin_version (expected 1)");
  } else {
    // The previous block format names this same model nep_spin3/spin_mode 3.
    const auto& mode = field("spin_mode", 1, 1);
    if (integer(mode, 0, 3) != 3) throw std::runtime_error("Expected spin_mode 3");
  }
  const bool dof = doc.find_field("spin_dof_type") != nullptr;
  const bool env = doc.find_field("spin_env_type") != nullptr;
  out << "spin_mode 3 " << 9 + dof + env << '\n';
  auto line = [&](const NEP_File::Field& f, const char* name) {
    out << name;
    for (const auto& value : f.values) out << ' ' << value;
    out << '\n';
  };
  line(field("energy_reference", types, types), "spin_baseline");
  for (const char* name : {"spin_basis_size", "spin_l_max", "spin_compress", "spin_order", "spin_soc", "spin_projection_size", "spin_scaler"})
    line(field(name, 1, 1), name);
  line(field("spin_cutoff", 1, types), "spin_cutoff");
  if (dof) line(field("spin_dof_type", 1, types), "spin_dof_type");
  if (env) line(field("spin_env_type", 1, types), "spin_env_type");
  if (zbl) line(field("zbl", 2, 3), "zbl");
  line(field("cutoff", 4, 4), "cutoff");
  const auto& nmax = field("n_max", 2, 2);
  const auto& basis = field("basis_size", 2, 2);
  const auto& lmax = field("l_max", 3, 7);
  const auto& ann = field("ann", 2, 2);
  line(nmax, "n_max"); line(basis, "basis_size"); line(lmax, "l_max"); line(ann, "ANN");
  const int nr = integer(nmax, 0, 19), na = integer(nmax, 1, 19);
  const int br = integer(basis, 0, 19), ba = integer(basis, 1, 19);
  const int neurons = integer(ann, 0, 200);
  if (neurons == 0 || integer(ann, 1, 0) != 0) throw std::runtime_error("Spin3 requires one hidden layer");
  const auto* parameters = doc.find_block("parameters");
  const auto* scaler = doc.find_block("q_scaler");
  if (!parameters || !scaler || scaler->values.empty() || scaler->values.size() > 199)
    throw std::runtime_error("Invalid Spin3 parameter/scaler blocks");
  const int dim = scaler->values.size();
  const int ann_count = (dim + 2) * neurons * types + 1;
  const int C = integer(*doc.find_field("spin_compress"), 0, 9);
  const int sb = integer(*doc.find_field("spin_basis_size"), 0, 100);
  const int projection = integer(*doc.find_field("spin_projection_size"), 0, 324);
  const int count = ann_count + types * types * ((nr + 1) * (br + 1) + (na + 1) * (ba + 1) + C * (sb + 1)) + projection;
  if (parameters->values.size() != size_t(count)) throw std::runtime_error("Spin3 parameter count mismatch");
  std::vector<float> values;
  for (double value : parameters->values) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
      throw std::runtime_error("Spin3 parameter exceeds float range");
    values.push_back(static_cast<float>(value));
  }
  descriptor_parameters_to_basis_major(values.data(), ann_count, types * types, nr, na, br, ba);
  for (float value : values) out << value << '\n';
  for (double value : scaler->values) {
    if (!(value > 0) || value > std::numeric_limits<float>::max()) throw std::runtime_error("Invalid Spin3 scaler");
    out << value << '\n';
  }
  std::vector<std::string> blocks = {"parameters", "q_scaler"};
  if (zbl) {
    const auto& values = doc.find_field("zbl")->values;
    double inner, outer;
    if (!NEP_File::parse_double(values[0], inner) || !NEP_File::parse_double(values[1], outer))
      throw std::runtime_error("Invalid Spin3 ZBL cutoffs");
    if (inner == 0 && outer == 0) {
      const auto* block = doc.find_block("zbl_parameters");
      if (!block || block->values.size() != size_t(5 * types * (types + 1)))
        throw std::runtime_error("Invalid Spin3 ZBL parameter count");
      blocks.push_back("zbl_parameters");
      for (double value : block->values) out << value << '\n';
    }
  }
  std::string error;
  if (!doc.check_unhandled(fields, blocks, error)) throw std::runtime_error(error);
  return out.str();
}
