/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Native enhanced sampling action.
------------------------------------------------------------------------------*/

#include "enhanced_sampling.cuh"
#include "enhanced_sampling_distance_cv.cuh"
#include "enhanced_sampling_harmonic_bias.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

static void enhanced_sampling_input_error(
  const std::string& filename,
  const int line,
  const std::string& message)
{
  const std::string error_message =
    "In " + filename + ", line " + std::to_string(line) + ": " + message + "\n";
  PRINT_INPUT_ERROR(error_message.c_str());
}

static bool enhanced_sampling_valid_name(const std::string& name)
{
  if (name.empty()) {
    return false;
  }
  const unsigned char first = static_cast<unsigned char>(name[0]);
  if (!(std::isalpha(first) || name[0] == '_')) {
    return false;
  }
  for (int index = 1; index < name.size(); ++index) {
    const unsigned char character = static_cast<unsigned char>(name[index]);
    if (!(std::isalnum(character) || name[index] == '_')) {
      return false;
    }
  }
  return true;
}

static bool enhanced_sampling_is_orthogonal(const Box& box)
{
  return
    box.cpu_h[1] == 0.0 && box.cpu_h[2] == 0.0 && box.cpu_h[3] == 0.0 &&
    box.cpu_h[5] == 0.0 && box.cpu_h[6] == 0.0 && box.cpu_h[7] == 0.0;
}

EnhancedSamplingAction::EnhancedSamplingAction(const char** param, const int num_param)
  : input_line_(0),
    output_interval_(0),
    output_is_set_(false),
    start_is_set_(false),
    output_(NULL)
{
  action_name = "enhanced_sampling";
  if (num_param != 2) {
    PRINT_INPUT_ERROR("enhanced_sampling should have one parameter: the input file.\n");
  }
  input_filename_ = param[1];
  parse_input_file();

  printf("Use native enhanced sampling.\n");
  printf("    input file: %s.\n", input_filename_.c_str());
  printf("    output file: %s.\n", output_filename_.c_str());
  printf("    output every %d steps.\n", output_interval_);
  printf("    number of collective variables: %d.\n", engine_.get_number_of_cvs());
  printf("    number of biases: %d.\n", engine_.get_number_of_biases());
}

EnhancedSamplingAction::~EnhancedSamplingAction()
{
  close_output();
}

void EnhancedSamplingAction::parse_input_file()
{
  std::ifstream input(input_filename_.c_str());
  if (!input.is_open()) {
    const std::string message = "Failed to open " + input_filename_ + ".\n";
    PRINT_INPUT_ERROR(message.c_str());
  }

  std::string line;
  while (std::getline(input, line)) {
    ++input_line_;
    const std::vector<std::string> tokens = get_tokens_without_comments(line);
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0] == "output") {
      parse_output(tokens);
    } else if (tokens[0] == "start") {
      parse_start(tokens);
    } else if (tokens[0] == "cv") {
      parse_cv(tokens);
    } else if (tokens[0] == "bias") {
      parse_bias(tokens);
    } else {
      enhanced_sampling_input_error(
        input_filename_, input_line_, "Unknown enhanced-sampling keyword '" + tokens[0] + "'.");
    }
  }
  input.close();

  if (!output_is_set_) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "An output line is required.");
  }
  if (input_filename_ == output_filename_) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The input and output file names must be different.");
  }
  if (engine_.get_number_of_cvs() == 0) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "At least one collective variable is required.");
  }
  if (engine_.get_number_of_biases() == 0) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "At least one bias is required.");
  }
}

void EnhancedSamplingAction::parse_output(const std::vector<std::string>& tokens)
{
  if (output_is_set_) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The output line can only be specified once.");
  }
  if (tokens.size() != 3) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "Output syntax is: output <filename> <interval>.");
  }
  if (!is_valid_int(tokens[2].c_str(), &output_interval_) || output_interval_ <= 0) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The output interval must be a positive integer.");
  }
  output_filename_ = tokens[1];
  if (output_filename_.empty()) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The output file name cannot be empty.");
  }
  output_is_set_ = true;
}

void EnhancedSamplingAction::parse_start(const std::vector<std::string>& tokens)
{
  if (start_is_set_) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The start line can only be specified once.");
  }
  if (tokens.size() != 2 || tokens[1] != "fresh") {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "V1A only accepts: start fresh.");
  }
  start_is_set_ = true;
}

void EnhancedSamplingAction::parse_cv(const std::vector<std::string>& tokens)
{
  if (tokens.size() != 8) {
    enhanced_sampling_input_error(
      input_filename_,
      input_line_,
      "Distance CV syntax is: cv <name> distance atoms <i> <j> pbc <on|off>.");
  }
  const std::string name = tokens[1];
  if (!enhanced_sampling_valid_name(name)) {
    enhanced_sampling_input_error(
      input_filename_,
      input_line_,
      "A CV name must contain only letters, digits, and underscores and cannot "
      "start with a digit.");
  }
  if (engine_.has_cv(name)) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "CV names must be unique.");
  }
  if (tokens[2] != "distance" || tokens[3] != "atoms" || tokens[6] != "pbc") {
    enhanced_sampling_input_error(
      input_filename_,
      input_line_,
      "Distance CV syntax is: cv <name> distance atoms <i> <j> pbc <on|off>.");
  }

  int atom_i = -1;
  int atom_j = -1;
  if (!is_valid_int(tokens[4].c_str(), &atom_i) || !is_valid_int(tokens[5].c_str(), &atom_j)) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "Distance-CV atom indices must be integers.");
  }
  if (atom_i < 0 || atom_j < 0) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "Distance-CV atom indices must be non-negative.");
  }
  if (atom_i == atom_j) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "A distance CV requires two different atoms.");
  }

  bool use_pbc = false;
  if (tokens[7] == "on") {
    use_pbc = true;
  } else if (tokens[7] == "off") {
    use_pbc = false;
  } else {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The distance-CV pbc option must be 'on' or 'off'.");
  }

  std::unique_ptr<EnhancedSamplingCV> cv;
  cv.reset(new EnhancedSamplingDistanceCV(name, atom_i, atom_j, use_pbc));
  engine_.add_cv(std::move(cv));
}

void EnhancedSamplingAction::parse_bias(const std::vector<std::string>& tokens)
{
  if (tokens.size() != 9) {
    enhanced_sampling_input_error(
      input_filename_,
      input_line_,
      "Harmonic-bias syntax is: bias <name> harmonic arg <cv> center <value> kappa <value>.");
  }
  const std::string name = tokens[1];
  if (!enhanced_sampling_valid_name(name)) {
    enhanced_sampling_input_error(
      input_filename_,
      input_line_,
      "A bias name must contain only letters, digits, and underscores and cannot "
      "start with a digit.");
  }
  if (engine_.has_bias(name)) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "Bias names must be unique.");
  }
  if (
    tokens[2] != "harmonic" || tokens[3] != "arg" || tokens[5] != "center" ||
    tokens[7] != "kappa") {
    enhanced_sampling_input_error(
      input_filename_,
      input_line_,
      "Harmonic-bias syntax is: bias <name> harmonic arg <cv> center <value> kappa <value>.");
  }

  const std::string argument_name = tokens[4];
  if (!engine_.has_cv(argument_name)) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "A bias must refer to a CV defined earlier in the file.");
  }

  double center = 0.0;
  double kappa = 0.0;
  if (!is_valid_real(tokens[6].c_str(), &center) || !std::isfinite(center)) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The harmonic center must be a finite number.");
  }
  if (!is_valid_real(tokens[8].c_str(), &kappa) || !std::isfinite(kappa) || kappa < 0.0) {
    enhanced_sampling_input_error(
      input_filename_, input_line_, "The harmonic kappa must be a finite, non-negative number.");
  }

  const int argument_index = engine_.get_cv_index(argument_name);
  std::unique_ptr<EnhancedSamplingBias> bias;
  bias.reset(new EnhancedSamplingHarmonicBias(
    name, argument_name, argument_index, center, kappa));
  engine_.add_bias(std::move(bias));
}

bool EnhancedSamplingAction::should_output(const int step) const
{
  return step % output_interval_ == 0;
}

void EnhancedSamplingAction::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  const EnsembleType type = integrate.get_type();
  if (
    type != EnsembleType::NVE &&
    type != EnsembleType::NVT_NHC &&
    type != EnsembleType::NVT_BDP) {
    PRINT_INPUT_ERROR(
      "Enhanced sampling V1A only supports NVE, NVT-NHC, and NVT-BDP.\n");
  }
  if (
    (type == EnsembleType::NVT_NHC || type == EnsembleType::NVT_BDP) &&
    integrate.get_temperature1() != integrate.get_temperature2()) {
    PRINT_INPUT_ERROR(
      "Enhanced sampling V1A requires a constant target temperature.\n");
  }
  if (integrate.get_fixed_group() >= 0 || integrate.get_move_group() >= 0) {
    PRINT_INPUT_ERROR(
      "Enhanced sampling V1A does not support fixed or moving atom groups.\n");
  }
  if (!enhanced_sampling_is_orthogonal(box)) {
    PRINT_INPUT_ERROR("Enhanced sampling V1A requires an orthogonal simulation box.\n");
  }
  engine_.prepare(atom.number_of_atoms);
  output_ = my_fopen(output_filename_.c_str(), "w");
  engine_.write_header(output_);
}

void EnhancedSamplingAction::setup_force(
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  engine_.calculate(box, atom, true);
  engine_.write_output(output_, 0);
}

void EnhancedSamplingAction::post_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  const int md_step = step + 1;
  engine_.calculate(box, atom, should_output(md_step));
}

void EnhancedSamplingAction::end_of_step(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  const int md_step = step + 1;
  if (should_output(md_step)) {
    engine_.write_output(output_, md_step);
  }
}

void EnhancedSamplingAction::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  engine_.check();
  close_output();
}

void EnhancedSamplingAction::close_output()
{
  if (output_ != NULL) {
    fclose(output_);
    output_ = NULL;
  }
}
