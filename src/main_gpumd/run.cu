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
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/

#include "add_efield.cuh"
#include "add_force.cuh"
#include "add_random_force.cuh"
#include "cohesive.cuh"
#include "electron_stop.cuh"
#include "force/force.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "measure/active.cuh"
#include "measure/adf.cuh"
#include "measure/angular_rdf.cuh"
#include "measure/compute.cuh"
#include "measure/dos.cuh"
#include "measure/dump_beads.cuh"
#include "measure/dump_dipole.cuh"
#include "measure/dump_exyz.cuh"
#include "measure/dump_force.cuh"
#include "measure/dump_netcdf.cuh"
#include "measure/dump_observer.cuh"
#include "measure/dump_polarizability.cuh"
#include "measure/dump_position.cuh"
#include "measure/dump_restart.cuh"
#include "measure/dump_shock_nemd.cuh"
#include "measure/dump_thermo.cuh"
#include "measure/dump_velocity.cuh"
#include "measure/dump_xyz.cuh"
#include "measure/extrapolation.cuh"
#include "measure/hac.cuh"
#include "measure/hnemd_kappa.cuh"
#include "measure/hnemdec_kappa.cuh"
#include "measure/lsqt.cuh"
#include "measure/measure.cuh"
#include "measure/modal_analysis.cuh"
#include "measure/msd.cuh"
#include "measure/orientorder.cuh"
#include "measure/plumed.cuh"
#include "measure/property.cuh"
#include "measure/rdf.cuh"
#include "measure/sdc.cuh"
#include "measure/shc.cuh"
#include "measure/viscosity.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "phonon/hessian.cuh"
#include "replicate.cuh"
#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "velocity.cuh"
#include <chrono>
#include <cstring>

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

Run::Run()
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  initialize_position(has_velocity_in_xyz, number_of_types, box, group, atom);

  allocate_memory_gpu(group, atom, thermo);

  velocity.initialize(
    has_velocity_in_xyz,
    300,
    atom.cpu_mass,
    atom.cpu_position_per_atom,
    atom.cpu_velocity_per_atom,
    atom.velocity_per_atom,
    false,
    123);
  if (has_velocity_in_xyz) {
    printf("Initialized velocities with data in model.xyz.\n");
  } else {
    printf("Initialized velocities with default T = 300 K.\n");
  }

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  execute_run_in();
}

void Run::execute_run_in()
{
  print_line_1();
  printf("Started executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  std::ifstream input("run.in");
  if (!input.is_open()) {
    std::cout << "Failed to open run.in." << std::endl;
    exit(1);
  }

  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    std::vector<std::string> tokens_without_comments;
    for (const auto& t : tokens) {
      if (t[0] != '#') {
        tokens_without_comments.emplace_back(t);
      } else {
        break;
      }
    }
    if (tokens_without_comments.size() > 0) {
      parse_one_keyword(tokens_without_comments);
    }
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  input.close();
}

void Run::perform_a_run()
{
  integrate.initialize(time_step, atom, box, group, thermo, number_of_steps);
  mc.initialize();
  measure.initialize(number_of_steps, time_step, integrate, group, atom, box, force);

  const auto time_begin = std::chrono::high_resolution_clock::now();

  // compute force for the first integrate step
  if (integrate.type >= 31) { // PIMD
    for (int k = 0; k < integrate.number_of_beads; ++k) {
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
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.velocity_per_atom,
      atom.mass);
  }

  double initial_time_step = time_step;

  for (int step = 0; step < number_of_steps; ++step) {

    velocity.correct_velocity(
      step,
      group,
      atom.cpu_mass,
      atom.position_per_atom,
      atom.cpu_position_per_atom,
      atom.cpu_velocity_per_atom,
      atom.velocity_per_atom);

    calculate_time_step(
      max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;

    integrate.current_step = step;
    integrate.compute1(time_step, double(step) / number_of_steps, group, box, atom, thermo);

    if (integrate.type >= 31) { // PIMD
      for (int k = 0; k < integrate.number_of_beads; ++k) {
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
      force.compute(
        box,
        atom.position_per_atom,
        atom.type,
        group,
        atom.potential_per_atom,
        atom.force_per_atom,
        atom.virial_per_atom,
        atom.velocity_per_atom,
        atom.mass);
    }

    electron_stop.compute(time_step, atom);
    add_force.compute(step, group, atom);
    add_random_force.compute(step, atom);
    add_efield.compute(step, group, atom, force);

    integrate.compute2(time_step, double(step) / number_of_steps, group, box, atom, thermo, force);

    mc.compute(step, number_of_steps, atom, box, group);

    measure.process(
      number_of_steps,
      step,
      integrate.fixed_group,
      integrate.move_group,
      global_time,
      integrate.temperature2,
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

  measure.finalize(atom, box, integrate, number_of_steps, time_step, integrate.temperature2);

  electron_stop.finalize();
  add_force.finalize();
  add_random_force.finalize();
  add_efield.finalize();
  integrate.finalize();
  mc.finalize();
  velocity.finalize();
  force.finalize();
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const int max_num_param = 32;
  if (num_param > max_num_param)
    PRINT_INPUT_ERROR("The number of parameters should be less than 32.\n");
  const char* param[max_num_param];
  for (int n = 0; n < num_param; ++n) {
    param[n] = tokens[n].c_str();
  }

  if (strcmp(param[0], "potential") == 0) {
    force.parse_potential(param, num_param, box, atom.type.size());
  } else if (strcmp(param[0], "replicate") == 0) {
    Replicate(param, num_param, box, atom, group);
    allocate_memory_gpu(group, atom, thermo);
  } else if (strcmp(param[0], "minimize") == 0) {
    Minimize minimize;
    minimize.parse_minimize(
      param,
      num_param,
      force,
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom);
  } else if (strcmp(param[0], "compute_phonon") == 0) {
    Hessian hessian;
    hessian.parse(param, num_param);
    hessian.compute(
      force,
      box,
      atom.cpu_position_per_atom,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom);
  } else if (strcmp(param[0], "compute_cohesive") == 0) {
    Cohesive cohesive;
    cohesive.parse(param, num_param, 0);
    cohesive.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      force);
  } else if (strcmp(param[0], "compute_elastic") == 0) {
    Cohesive cohesive;
    cohesive.parse(param, num_param, 1);
    cohesive.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      force);
  } else if (strcmp(param[0], "change_box") == 0) {
    parse_change_box(param, num_param);
  } else if (strcmp(param[0], "velocity") == 0) {
    parse_velocity(param, num_param);
  } else if (strcmp(param[0], "ensemble") == 0) {
    integrate.parse_ensemble(param, num_param, time_step, atom, box, group, thermo);
  } else if (strcmp(param[0], "time_step") == 0) {
    parse_time_step(param, num_param);
  } else if (strcmp(param[0], "correct_velocity") == 0) {
    parse_correct_velocity(param, num_param, group);
  } else if (strcmp(param[0], "dump_thermo") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Thermo(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_position") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Position(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_netcdf") == 0) {
#ifdef USE_NETCDF
    std::unique_ptr<Property> property;
    property.reset(new DUMP_NETCDF(param, num_param));
    measure.properties.emplace_back(std::move(property));
#else
    PRINT_INPUT_ERROR("dump_netcdf is available only when USE_NETCDF flag is set.\n");
#endif
  } else if (strcmp(param[0], "plumed") == 0) {
#ifdef USE_PLUMED
    std::unique_ptr<Property> property;
    property.reset(new PLUMED(param, num_param));
    measure.properties.emplace_back(std::move(property));
#else
    PRINT_INPUT_ERROR("plumed is available only when USE_PLUMED flag is set.\n");
#endif
  } else if (strcmp(param[0], "dump_restart") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Restart(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_velocity") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Velocity(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_force") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Force(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_exyz") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_EXYZ(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_xyz") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_XYZ(param, num_param, group, atom));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_beads") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Beads(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_observer") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Observer(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_shock_nemd") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Shock_NEMD(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_dipole") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Dipole(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "dump_polarizability") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Dump_Polarizability(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "active") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Active(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_extrapolation") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Extrapolation(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_dos") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new DOS(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_sdc") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new SDC(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_msd") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new MSD(param, num_param, group, atom));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_rdf") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new RDF(param, num_param, box, number_of_types, number_of_steps));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_adf") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new ADF(param, num_param, box, number_of_types));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_orientorder") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new OrientOrder(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_angular_rdf") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new AngularRDF(param, num_param, box, number_of_types, number_of_steps));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_hac") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new HAC(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_viscosity") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Viscosity(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_hnemd") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new HNEMD(param, num_param, force));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_hnemdec") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new HNEMDEC(param, num_param, force, atom, integrate.temperature1));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_shc") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new SHC(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_gkma") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new MODAL_ANALYSIS(param, num_param, number_of_types, 0, force));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "compute_hnema") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new MODAL_ANALYSIS(param, num_param, number_of_types, 1, force));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "deform") == 0) {
    integrate.parse_deform(param, num_param);
  } else if (strcmp(param[0], "compute") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new Compute(param, num_param, group));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "fix") == 0) {
    integrate.parse_fix(param, num_param, group);
  } else if (strcmp(param[0], "move") == 0) {
    integrate.parse_move(param, num_param, group);
  } else if (strcmp(param[0], "electron_stop") == 0) {
    electron_stop.parse(param, num_param, atom.number_of_atoms, number_of_types);
  } else if (strcmp(param[0], "add_random_force") == 0) {
    add_random_force.parse(param, num_param, atom.number_of_atoms);
  } else if (strcmp(param[0], "add_force") == 0) {
    add_force.parse(param, num_param, group);
  } else if (strcmp(param[0], "add_efield") == 0) {
    add_efield.parse(param, num_param, group);
  } else if (strcmp(param[0], "mc") == 0) {
    mc.parse_mc(param, num_param, group, atom);
  } else if (strcmp(param[0], "dftd3") == 0) {
    // nothing here; will be handled elsewhere
  } else if (strcmp(param[0], "compute_lsqt") == 0) {
    std::unique_ptr<Property> property;
    property.reset(new LSQT(param, num_param));
    measure.properties.emplace_back(std::move(property));
  } else if (strcmp(param[0], "run") == 0) {
    parse_run(param, num_param);
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}

void Run::parse_velocity(const char** param, int num_param)
{
  int seed = 0;
  bool use_seed = false;
  if (!(num_param == 2 || num_param == 4)) {
    PRINT_INPUT_ERROR("velocity should have 1 or 2 parameters.\n");
  } else if (num_param == 4) {
    // See https://github.com/brucefan1983/GPUMD/pull/768
    // for the reason for putting this branch here.
    use_seed = true;
    if (!is_valid_int(param[3], &seed)) {
      PRINT_INPUT_ERROR("seed should be a positive integer.\n");
    }
  }

  if (!is_valid_real(param[1], &initial_temperature)) {
    PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
  }
  if (initial_temperature <= 0.0) {
    PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
  }

  velocity.initialize(
    has_velocity_in_xyz,
    initial_temperature,
    atom.cpu_mass,
    atom.cpu_position_per_atom,
    atom.cpu_velocity_per_atom,
    atom.velocity_per_atom,
    use_seed,
    seed);
  if (!has_velocity_in_xyz) {
    printf("Initialized velocities with input T = %g K.\n", initial_temperature);
  }
}

void Run::parse_correct_velocity(const char** param, int num_param, const std::vector<Group>& group)
{
  printf("Correct linear and angular momenta.\n");

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("correct_velocity should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &velocity.velocity_correction_interval)) {
    PRINT_INPUT_ERROR("velocity correction interval should be an integer.\n");
  }
  if (velocity.velocity_correction_interval < 10) {
    PRINT_INPUT_ERROR("velocity correction interval should >= 10.\n");
  }

  printf("    every %d steps.\n", velocity.velocity_correction_interval);

  if (num_param == 3) {
    if (!is_valid_int(param[2], &velocity.velocity_correction_group_method)) {
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

void Run::parse_time_step(const char** param, int num_param)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("time_step should have 1 or 2 parameters.\n");
  }
  if (!is_valid_real(param[1], &time_step)) {
    PRINT_INPUT_ERROR("time_step should be a real number.\n");
  }
  printf("Time step for this run is %g fs.\n", time_step);
  time_step /= TIME_UNIT_CONVERSION;
  if (num_param == 3) {
    if (!is_valid_real(param[2], &max_distance_per_step)) {
      PRINT_INPUT_ERROR("max distance per step should be a real number.\n");
    }
    if (max_distance_per_step <= 0.0) {
      PRINT_INPUT_ERROR("max distance per step should > 0.\n");
    }
    printf("    max distance per step = %g A.\n", max_distance_per_step);
  }
}

void Run::parse_run(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("run should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &number_of_steps)) {
    PRINT_INPUT_ERROR("number of steps should be an integer.\n");
  }
  printf("Run %d steps.\n", number_of_steps);

  // set target temperature for temperature-dependent NEP
  force.temperature = integrate.temperature1;
  force.delta_T = (integrate.temperature2 - integrate.temperature1) / number_of_steps;

  perform_a_run();
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

void Run::parse_change_box(const char** param, int num_param)
{
  if (num_param != 2 && num_param != 4 && num_param != 7) {
    PRINT_INPUT_ERROR("change_box can only have 1 or 3 or 6 parameters\n.");
  }

  double deformation_matrix[3][3] = {0.0};

  if (!is_valid_real(param[1], &deformation_matrix[0][0])) {
    PRINT_INPUT_ERROR("box change parameter in xx should be a number.");
  }
  deformation_matrix[1][1] = deformation_matrix[2][2] = deformation_matrix[0][0];

  if (num_param >= 4) {
    if (!is_valid_real(param[2], &deformation_matrix[1][1])) {
      PRINT_INPUT_ERROR("box change parameter in yy should be a number.");
    }
    if (!is_valid_real(param[3], &deformation_matrix[2][2])) {
      PRINT_INPUT_ERROR("box change parameter in zz should be a number.");
    }
  }

  if (num_param == 7) {
    if (!is_valid_real(param[4], &deformation_matrix[1][2])) {
      PRINT_INPUT_ERROR("box change parameter in yz should be a number.");
    }
    if (!is_valid_real(param[5], &deformation_matrix[0][2])) {
      PRINT_INPUT_ERROR("box change parameter in xz should be a number.");
    }
    if (!is_valid_real(param[6], &deformation_matrix[0][1])) {
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
