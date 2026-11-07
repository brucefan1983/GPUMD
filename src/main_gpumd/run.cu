/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "force/force.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "integrate/thermo.cuh"
#include "measure/measure.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "phonon/hessian.cuh"
#include "replicate.cuh"
#include "rerun_reader.hpp"
#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include "velocity.cuh"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

static __global__ void gpu_find_largest_v2(
  int N, int number_of_rounds, double* g_vx, double* g_vy, double* g_vz, double* g_v2_max)
{
  int tid = threadIdx.x;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = round * 1024 + tid;
    if (n < N) {
      double vx = g_vx[n];
      double vy = g_vy[n];
      double vz = g_vz[n];
      double v2 = vx * vx + vy * vy + vz * vz;
      if (s_data[tid] < v2) {
        s_data[tid] = v2;
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_data[tid] < s_data[tid + offset]) {
        s_data[tid] = s_data[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_v2_max[0] = s_data[0];
  }
}

__device__ double device_v2_max[1];

static void calculate_time_step(
  double max_distance_per_step,
  GPU_Vector<double>& velocity_per_atom,
  double initial_time_step,
  double& time_step)
{
  if (max_distance_per_step <= 0.0) {
    return;
  }
  const int N = velocity_per_atom.size() / 3;
  double* gpu_v2_max;
  CHECK(gpuGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N,
    (N - 1) / 1024 + 1,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    gpu_v2_max);
  GPU_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(gpuMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), gpuMemcpyDeviceToHost));
  double cpu_v_max = sqrt(cpu_v2_max[0]);
  double time_step_min = max_distance_per_step / cpu_v_max;

  if (time_step_min < initial_time_step) {
    time_step = time_step_min;
  } else {
    time_step = initial_time_step;
  }
}

static bool parse_initial_replicate(
  const RunInput& run_input, int replicate_size[3])
{
  for (const auto& line : run_input.lines()) {
    if (line.tokens.empty()) {
      continue;
    }
    if (line.tokens[0] != "replicate") {
      return false;
    }
    parse_replicate(line.tokens, replicate_size);
    return true;
  }
  return false;
}

Run::Run(const RunInput& run_input)
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  has_replicate_ = parse_initial_replicate(run_input, replicate_size_);

  if (get_first_potential_filename(run_input).empty()) {
    PRINT_INPUT_ERROR("A potential is required in run.in, including for rerun.");
  }

  initialize_position(
    run_input,
    has_velocity_in_xyz,
    number_of_types,
    box,
    group,
    atom);
  first_potential_filename_ = get_first_potential_filename(run_input);

  if (has_replicate_) {
    Replicate(replicate_size_, box, atom, group);
  }

  if (atom.number_of_atoms < 2) {
    PRINT_INPUT_ERROR("Number of atoms should >= 2.");
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz, 300, atom, false, 123);

  if (has_velocity_in_xyz) {
    printf("Initialized velocities with data in model.xyz.\n");
  } else {
    printf("Initialized velocities with default T = 300 K.\n");
  }

  allocate_memory_gpu(group, atom, thermo);

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  execute_run_in(run_input);
}

void Run::execute_run_in(const RunInput& run_input)
{
  print_line_1();
  printf("Started executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  bool first_effective_command = true;
  for (const auto& line : run_input.lines()) {
    if (!line.tokens.empty()) {
      std::vector<std::string> tokens = line.tokens;
      if (tokens.size() >= 2 && tokens[0] == "potential") {
        tokens[1] = get_compact_nep_filename(tokens[1]);
      }
      if (has_replicate_ && first_effective_command &&
          tokens[0] == "replicate") {
        first_effective_command = false;
        continue;
      }
      first_effective_command = false;
      parse_one_keyword(tokens, run_input);
    }
  }

  if (integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("The last ensemble is not followed by a run.");
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

}

void Run::compute_force()
{
  if (is_pimd(integrate.get_type())) {
    for (int k = 0; k < integrate.get_number_of_beads(); ++k) {
      force.compute(
        box,
        atom.position_beads[k],
        atom.type,
        group,
        atom.potential_beads[k],
        atom.force_beads[k],
        atom.virial_beads[k],
        atom.velocity_beads[k],
        atom.mass);
    }
  } else {
    force.compute(box, atom, group);
  }
}

void Run::perform_a_run(const int number_of_steps)
{
  integrate.initialize(time_step, atom, box, group);
  measure.pre_run(number_of_steps, time_step, integrate, group, atom, box, force);

  // setup force for the first integrate step
  compute_force();
  atom.update_unwrapped_position(box);
  measure.setup_force(time_step, integrate, group, atom, box, force);

  double initial_time_step = time_step;

  const auto time_begin = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < number_of_steps; ++step) {

    velocity.correct_velocity(step, group, atom);

    calculate_time_step(
      max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;
    ++global_md_step;

    integrate.compute1(time_step, step, number_of_steps, group, box, atom, thermo);

    measure.post_integrate1(step, time_step, integrate, group, atom, box, force);

    force.advance_temperature();

    measure.pre_force(step, time_step, integrate, group, atom, box, force);
    compute_force();

    atom.update_unwrapped_position(box);
    measure.post_force(step, time_step, integrate, group, atom, box, force);

    integrate.compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
    atom.update_unwrapped_position(box);

    measure.set_xyz_header_step(global_md_step);
    measure.end_of_step(
      number_of_steps,
      step,
      integrate.get_fixed_group(),
      integrate.get_move_group(),
      global_time,
      integrate.get_temperature2(),
      integrate,
      box,
      group,
      thermo,
      atom,
      force);

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
      fflush(stdout);
    }
  }

  print_line_1();
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  printf("Time used for this run = %g second.\n", time_used.count());
  double run_speed = atom.number_of_atoms * (number_of_steps * 1.0 / time_used.count());
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  print_line_2();

  measure.post_run(
    atom, box, integrate, number_of_steps, time_step, integrate.get_temperature2());

  integrate.finalize(atom, box);
  velocity.finalize();
  force.finalize();
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(
  const std::vector<std::string>& tokens, const RunInput& run_input)
{
  if (tokens[0] == "replicate") {
    PRINT_INPUT_ERROR("replicate must be the first effective command.");
  }
  const int num_param = tokens.size();
  const int max_num_param = 32;
  if (num_param > max_num_param)
    PRINT_INPUT_ERROR("The number of parameters should be less than 32.\n");

#ifdef GPUMD_WPE_ENABLED
  force.wpe_process_command(tokens, atom.number_of_atoms);
#endif

  if (tokens[0] == "potential") {
    force.parse_potential(tokens, box, atom, run_input);
    has_potential_for_rerun_ = true;
  } else if (tokens[0] == "minimize") {
    Minimize minimize;
    minimize.parse_minimize(
      tokens,
      integrate.get_fixed_group(),
      integrate.get_fixed_grouping_method(),
      force,
      box,
      atom,
      group);
  } else if (tokens[0] == "compute_phonon") {
    Hessian hessian;
    hessian.parse(tokens);
    if (!has_replicate_) {
      PRINT_INPUT_ERROR("replicate keyword not found in run.in file.");
    }
    hessian.compute(force, box, atom, group, replicate_size_);
  } else if (tokens[0] == "compute_cohesive") {
    Cohesive cohesive;
    cohesive.parse(tokens, 0);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "compute_elastic") {
    Cohesive cohesive;
    cohesive.parse(tokens, 1);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "change_box") {
    parse_change_box(tokens);
  } else if (tokens[0] == "velocity") {
    parse_velocity(tokens);
  } else if (tokens[0] == "ensemble") {
    integrate.parse_ensemble(tokens, atom, box, group);
  } else if (tokens[0] == "time_step") {
    parse_time_step(tokens);
  } else if (tokens[0] == "correct_velocity") {
    parse_correct_velocity(tokens, group);
  } else if (tokens[0] == "fix") {
    integrate.parse_fix(tokens, group);
  } else if (tokens[0] == "move") {
    integrate.parse_move(tokens, group);
  } else if (tokens[0] == "kspace") {
    if (has_seen_kspace_command) {
      PRINT_INPUT_ERROR("kspace can only appear once.");
    }
    has_seen_kspace_command = true;
  } else if (tokens[0] == "dftd3") {
    if (has_seen_dftd3_command) {
      PRINT_INPUT_ERROR("dftd3 can only appear once.");
    }
    has_seen_dftd3_command = true;
  } else if (tokens[0] == "run") {
    parse_run(tokens);
  } else if (tokens[0] == "rerun") {
    parse_rerun(tokens);
  } else if (!measure.parse_action(
               tokens,
               number_of_types,
               integrate,
               group,
               atom,
               box,
               force,
               first_potential_filename_)) {
    PRINT_KEYWORD_ERROR(tokens[0].c_str());
  }
}


void Run::parse_rerun(const std::vector<std::string>& tokens)
{
  // Pseudo-run: trajectories replace integration as the source of configurations.
  // A potential must already have been loaded, even for PLUMED-only analysis.
  // All analysis/output is dispatched by Measure; rerun never creates a data file.
  if (!has_potential_for_rerun_) {
    PRINT_INPUT_ERROR("A potential must be specified before rerun.");
  }
  if (tokens.size() < 2) {
    PRINT_INPUT_ERROR("rerun trajectory.xyz [first N] [last N] [every N] [skip N]");
  }
  // Replay reads any configured temperature from the ordinary GPUMD
  // ensemble command. No separate temperature option is introduced.
  const bool has_ensemble_for_rerun = integrate.has_ensemble();
  const double simulation_temperature =
    has_ensemble_for_rerun ? integrate.get_rerun_temperature() : 0.0;

  gpumd_rerun::Selection selection;
  // The first version accepts exactly one trajectory, tokens[1].
  // Any additional positional filename is rejected as an unknown keyword.
  for (size_t i = 2; i < tokens.size(); i += 2) {
    if (i + 1 >= tokens.size()) {
      PRINT_INPUT_ERROR("rerun requires a value after each keyword.");
    }
    long long value = 0;
    try {
      size_t used = 0;
      value = std::stoll(tokens[i + 1], &used);
      if (used != tokens[i + 1].size()) throw std::invalid_argument("trailing characters");
    } catch (const std::exception&) {
      PRINT_INPUT_ERROR("rerun first/last/every/skip require integer values.");
    }
    if (tokens[i] == "first") selection.first = value;
    else if (tokens[i] == "last") selection.last = value;
    else if (tokens[i] == "every") selection.every = value;
    else if (tokens[i] == "skip") selection.skip = value;
    else PRINT_INPUT_ERROR("unsupported rerun keyword (first, last, every, skip).");
  }
  if (selection.first < 0 || selection.last < selection.first ||
      selection.every < 0 || selection.skip <= 0) {
    PRINT_INPUT_ERROR("invalid rerun selection arguments.");
  }
  bool uses_temperature_dependent_potential = false;
  for (int i = 0; i < force.get_number_of_potentials(); ++i) {
    if (force.get_potential(i).nep_model_type == 3) {
      uses_temperature_dependent_potential = true;
    }
  }
  if (uses_temperature_dependent_potential && !(simulation_temperature > 0.0)) {
    PRINT_INPUT_ERROR(
      "Temperature-dependent potential requires a positive temperature from "
      "an existing GPUMD ensemble command before rerun.");
  }
  // An MD temperature ramp cannot be reproduced by one constant replay
  // force-model temperature. Do not silently use only the first endpoint.
  if (uses_temperature_dependent_potential && has_ensemble_for_rerun &&
      integrate.get_temperature2() > 0.0 &&
      std::abs(integrate.get_temperature1() - integrate.get_temperature2()) > 1.0e-10) {
    PRINT_INPUT_ERROR(
      "Temperature-dependent rerun currently requires a constant-temperature ensemble.");
  }
  measure.validate_rerun(has_potential_for_rerun_);
  if (measure.rerun_output_conflicts(tokens[1])) {
    PRINT_INPUT_ERROR("rerun dump_xyz output must not overwrite the input trajectory.");
  }
  const bool needs_velocity = measure.rerun_needs_velocity();
  const double dt_fs = time_step * TIME_UNIT_CONVERSION;
  // Match ordinary run: initialize Force from GPUMD's ensemble temperature,
  // but never initialize or advance the integrator.
  const double previous_force_temperature = force.get_temperature();
  if (simulation_temperature > 0.0) {
    force.set_temperature_range(simulation_temperature, simulation_temperature, 1);
  }
  const Box original_box = box;
  const int N = atom.number_of_atoms;

  // Keep the full working MD configuration intact for a following run/rerun.
  std::vector<double> original_positions(3 * N);
  std::vector<double> original_velocities(3 * N);
  atom.position_per_atom.copy_to_host(original_positions.data());
  atom.velocity_per_atom.copy_to_host(original_velocities.data());

  std::vector<int> original_images(atom.position_image.size());
  if (!original_images.empty()) atom.position_image.copy_to_host(original_images.data());
  std::vector<double> original_unwrapped(atom.unwrapped_position.size());
  if (!original_unwrapped.empty())
    atom.unwrapped_position.copy_to_host(original_unwrapped.data());
  std::vector<double> original_forces(atom.force_per_atom.size());
  std::vector<double> original_potential(atom.potential_per_atom.size());
  std::vector<double> original_virial(atom.virial_per_atom.size());
  std::vector<double> original_mforce(atom.mforce_per_atom.size());
  std::vector<double> original_thermo(thermo.size());
  if (!original_forces.empty()) atom.force_per_atom.copy_to_host(original_forces.data());
  if (!original_potential.empty()) atom.potential_per_atom.copy_to_host(original_potential.data());
  if (!original_virial.empty()) atom.virial_per_atom.copy_to_host(original_virial.data());
  if (!original_mforce.empty()) atom.mforce_per_atom.copy_to_host(original_mforce.data());
  if (!original_thermo.empty()) thermo.copy_to_host(original_thermo.data());

  // A separate thermodynamic calculator avoids requiring an MD ensemble.
  Thermo thermo_calculator;
  std::vector<double> zero_forces(3 * N, 0.0);
  // The potential is mandatory, so energy/forces/virial are always recalculated.
  std::vector<double> thermo_row(thermo.size(), 0.0);

  long long processed = 0;
  long long previous_step = std::numeric_limits<long long>::min();
  try {
    gpumd_rerun::Frame frame;
    measure.pre_rerun(
      time_step, simulation_temperature, integrate, group, atom, box, force, tokens[1]);
    gpumd_rerun::Reader reader(tokens[1]);
    while (reader.next(frame)) {
      const long long step = gpumd_rerun::frame_step(frame, dt_fs);
      // Skip duplicate or out-of-order timesteps in the input trajectory.
      if (step <= previous_step) continue;
      previous_step = step;
      // Snapshots are expected in ascending order. Like LAMMPS, stop once
      // the first snapshot beyond the requested last timestep is reached.
      if (step > selection.last) break;
      if (!selection.accepts(step)) continue;
      if (frame.species.size() != static_cast<size_t>(N))
        throw std::runtime_error("rerun atom count differs from model.xyz");
      for (int i = 0; i < N; ++i) {
        if (frame.species[i] != atom.cpu_atom_symbol[i])
          throw std::runtime_error("rerun atom symbols/order differ from model.xyz");
      }
      if (needs_velocity && frame.velocities.empty())
        throw std::runtime_error(
          "rerun output requires vel:R:3 in every selected XYZ frame");
      for (int i = 0; i < 9; ++i) box.cpu_h[i] = frame.lattice[i];
      box.pbc_x = frame.pbc[0];
      box.pbc_y = frame.pbc[1];
      box.pbc_z = frame.pbc[2];
      if (!(box.get_volume() > 1.0e-12))
        throw std::runtime_error("rerun invalid simulation cell");
      box.get_inverse();
      box.set_is_orthogonal();

      atom.position_per_atom.copy_from_host(frame.positions.data());
      if (!frame.velocities.empty()) {
        std::vector<double> velocity_natural = frame.velocities;
        for (double& v : velocity_natural) v *= TIME_UNIT_CONVERSION;
        atom.velocity_per_atom.copy_from_host(velocity_natural.data());
      } else {
        // No synthetic MD temperature: a missing velocity means zeroed velocity,
        // and the thermo action above explicitly rejects such a frame.
        atom.velocity_per_atom.copy_from_host(zero_forces.data());
      }
      if (!original_images.empty()) {
        std::vector<int> zero_images(original_images.size(), 0);
        atom.position_image.copy_from_host(zero_images.data());
      }
      atom.update_unwrapped_position(box);

      // Recalculate physical forces for each snapshot without integration.
      force.compute(box, atom, group);

      if (!frame.velocities.empty()) {
        thermo_calculator.compute(
          N, box.get_volume(), atom.mass, atom.potential_per_atom,
          atom.velocity_per_atom, atom.virial_per_atom, thermo);
      } else {
        std::fill(thermo_row.begin(), thermo_row.end(), 0.0);
        // A potential energy can still be computed with no velocities.
        // For position-only replay, use NaN rather than reporting false KE or stress.
        if (frame.velocities.empty()) {
          const double nan = std::numeric_limits<double>::quiet_NaN();
          thermo_row[0] = nan; // temperature and kinetic energy are unknown
          for (int j = 2; j < 8 && j < static_cast<int>(thermo_row.size()); ++j)
            thermo_row[j] = nan; // do not report artificial stress from missing velocities
        }
        std::vector<double> per_atom_energy(N);
        atom.potential_per_atom.copy_to_host(per_atom_energy.data());
        if (thermo_row.size() > 1)
          thermo_row[1] = std::accumulate(
            per_atom_energy.begin(), per_atom_energy.end(), 0.0);
        thermo.copy_from_host(thermo_row.data());
      }

      // Actual frame timestamp (internal natural time units), not the frame index.
      measure.rerun_frame(
        step, frame.time_fs / TIME_UNIT_CONVERSION,
        integrate, group, atom, box, force, thermo);
      ++processed;
    }
    measure.post_rerun(time_step, integrate, atom, box);
  } catch (const std::exception& error) {
    fprintf(stderr, "rerun failed: %s\n", error.what());
    exit(EXIT_FAILURE);
  }

  box = original_box;
  atom.position_per_atom.copy_from_host(original_positions.data());
  atom.velocity_per_atom.copy_from_host(original_velocities.data());
  if (!original_images.empty()) atom.position_image.copy_from_host(original_images.data());
  if (!original_unwrapped.empty())
    atom.unwrapped_position.copy_from_host(original_unwrapped.data());
  if (!original_forces.empty()) atom.force_per_atom.copy_from_host(original_forces.data());
  if (!original_potential.empty()) atom.potential_per_atom.copy_from_host(original_potential.data());
  if (!original_virial.empty()) atom.virial_per_atom.copy_from_host(original_virial.data());
  if (!original_mforce.empty()) atom.mforce_per_atom.copy_from_host(original_mforce.data());
  if (!original_thermo.empty()) thermo.copy_from_host(original_thermo.data());

  if (simulation_temperature > 0.0) {
    force.set_temperature_range(previous_force_temperature, previous_force_temperature, 1);
  }
  // No MD integration occurred, so do not call the MD ensemble finalizer.
  // Consume its configuration as a normal run would consume the ensemble.
  if (has_ensemble_for_rerun) integrate.finalize_rerun();
  printf("Rerun processed %lld frames without integration.\n", processed);
}

void Run::parse_velocity(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  double initial_temperature;
  int seed = 0;
  bool use_seed = false;
  if (!(num_param == 2 || num_param == 4)) {
    PRINT_INPUT_ERROR("velocity should have 1 or 3 parameters.\n");
  }

  if (!is_valid_real(tokens[1], &initial_temperature)) {
    PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
  }
  if (initial_temperature <= 0.0) {
    PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
  }

  if (num_param == 4) {
    if (tokens[2] != "seed") {
      PRINT_INPUT_ERROR("The second parameter for velocity should be 'seed'.\n");
    }
    use_seed = true;
    if (!is_valid_int(tokens[3], &seed) || seed <= 0) {
      PRINT_INPUT_ERROR("seed should be a positive integer.\n");
    }
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz,
    initial_temperature,
    atom,
    use_seed,
    seed);
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  if (!has_velocity_in_xyz) {
    printf("Initialized velocities with input T = %g K.\n", initial_temperature);
  }
}

void Run::parse_correct_velocity(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  printf("Correct linear and angular momenta.\n");

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("correct_velocity should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(tokens[1], &velocity.velocity_correction_interval)) {
    PRINT_INPUT_ERROR("velocity correction interval should be an integer.\n");
  }
  if (velocity.velocity_correction_interval < 10) {
    PRINT_INPUT_ERROR("velocity correction interval should >= 10.\n");
  }

  printf("    every %d steps.\n", velocity.velocity_correction_interval);

  if (num_param == 3) {
    if (!is_valid_int(tokens[2], &velocity.velocity_correction_group_method)) {
      PRINT_INPUT_ERROR("velocity correction group method should be an integer.\n");
    }
    if (velocity.velocity_correction_group_method < 0) {
      PRINT_INPUT_ERROR("grouping method should >= 0.\n");
    }
    if (velocity.velocity_correction_group_method >= group.size()) {
      PRINT_INPUT_ERROR("grouping method should < maximum number of grouping methods.\n");
    }
  }

  if (velocity.velocity_correction_group_method < 0) {
    printf("    for the whole system.\n");
  } else {
    printf(
      "    for individual groups in group method %d.\n", velocity.velocity_correction_group_method);
  }

  velocity.do_velocity_correction = true;
}

void Run::parse_time_step(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("time_step should have 1 or 2 parameters.\n");
  }
  if (!is_valid_real(tokens[1], &time_step)) {
    PRINT_INPUT_ERROR("time_step should be a real number.\n");
  }
  printf("Time step for this run is %g fs.\n", time_step);
  time_step /= TIME_UNIT_CONVERSION;
  if (num_param == 3) {
    if (!is_valid_real(tokens[2], &max_distance_per_step)) {
      PRINT_INPUT_ERROR("max distance per step should be a real number.\n");
    }
    if (max_distance_per_step <= 0.0) {
      PRINT_INPUT_ERROR("max distance per step should > 0.\n");
    }
    printf("    max distance per step = %g A.\n", max_distance_per_step);
  }
}

void Run::parse_run(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  int number_of_steps;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("run should have 1 parameter.\n");
  }
  if (!is_valid_int(tokens[1], &number_of_steps)) {
    PRINT_INPUT_ERROR("number of steps should be an integer.\n");
  }
  if (number_of_steps <= 0) {
    PRINT_INPUT_ERROR("number of steps should be positive.\n");
  }
  if (!integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("An ensemble must be specified before each run.");
  }
  if (!has_potential_for_rerun_) {
    PRINT_INPUT_ERROR("A potential must be specified before an MD run.");
  }
  printf("Run %d steps.\n", number_of_steps);

  // set target temperature for temperature-dependent NEP
  force.set_temperature_range(
    integrate.get_temperature1(), integrate.get_temperature2(), number_of_steps);

  perform_a_run(number_of_steps);
}

static __global__ void gpu_deform_atom(
  int N,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    g_x[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_y[i] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_z[i] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
  }
}

void Run::parse_change_box(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 4 && num_param != 7) {
    PRINT_INPUT_ERROR("change_box can only have 1 or 3 or 6 parameters\n.");
  }

  double deformation_matrix[3][3] = {0.0};

  if (!is_valid_real(tokens[1], &deformation_matrix[0][0])) {
    PRINT_INPUT_ERROR("box change parameter in xx should be a number.");
  }
  deformation_matrix[1][1] = deformation_matrix[2][2] = deformation_matrix[0][0];

  if (num_param >= 4) {
    if (!is_valid_real(tokens[2], &deformation_matrix[1][1])) {
      PRINT_INPUT_ERROR("box change parameter in yy should be a number.");
    }
    if (!is_valid_real(tokens[3], &deformation_matrix[2][2])) {
      PRINT_INPUT_ERROR("box change parameter in zz should be a number.");
    }
  }

  if (num_param == 7) {
    if (!is_valid_real(tokens[4], &deformation_matrix[1][2])) {
      PRINT_INPUT_ERROR("box change parameter in yz should be a number.");
    }
    if (!is_valid_real(tokens[5], &deformation_matrix[0][2])) {
      PRINT_INPUT_ERROR("box change parameter in xz should be a number.");
    }
    if (!is_valid_real(tokens[6], &deformation_matrix[0][1])) {
      PRINT_INPUT_ERROR("box change parameter in xy should be a number.");
    }
    deformation_matrix[1][0] = deformation_matrix[0][1];
    deformation_matrix[2][0] = deformation_matrix[0][2];
    deformation_matrix[2][1] = deformation_matrix[1][2];
  }

  printf("Change box:\n");
  printf("    in xx by %g A.\n", deformation_matrix[0][0]);
  printf("    in yy by %g A.\n", deformation_matrix[1][1]);
  printf("    in zz by %g A.\n", deformation_matrix[2][2]);
  printf("    in yz and zy by strain %g.\n", deformation_matrix[1][2]);
  printf("    in xz and zx by strain %g.\n", deformation_matrix[0][2]);
  printf("    in xy and yz by strain %g.\n", deformation_matrix[0][1]);

  for (int d = 0; d < 3; ++d) {
    deformation_matrix[d][d] =
      (box.cpu_h[d * 3 + d] + deformation_matrix[d][d]) / box.cpu_h[d * 3 + d];
  }

  printf("    Deformation matrix =\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", deformation_matrix[d1][d2]);
    }
    printf("\n");
  }

  printf("    Original box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }

  double h_old[9];
  for (int i = 0; i < 9; ++i) {
    h_old[i] = box.cpu_h[i];
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < 3; ++k) {
        tmp += deformation_matrix[r][k] * h_old[k * 3 + c];
      }
      box.cpu_h[r * 3 + c] = tmp;
    }
  }
  box.get_inverse();

  const int number_of_atoms = atom.position_per_atom.size() / 3;
  gpu_deform_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    deformation_matrix[0][0],
    deformation_matrix[0][1],
    deformation_matrix[0][2],
    deformation_matrix[1][0],
    deformation_matrix[1][1],
    deformation_matrix[1][2],
    deformation_matrix[2][0],
    deformation_matrix[2][1],
    deformation_matrix[2][2],
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL

  printf("    Changed box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }
}
