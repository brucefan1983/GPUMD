/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "cohesive.cuh"
#include "electron_stop.cuh"
#include "force/force.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "phonon/hessian.cuh"
#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "velocity.cuh"

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
  CHECK(cudaGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N,
    (N - 1) / 1024 + 1,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    gpu_v2_max);
  CUDA_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(cudaMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), cudaMemcpyDeviceToHost));
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

  initialize_position(N, has_velocity_in_xyz, number_of_types, box, group, atom);

  allocate_memory_gpu(N, group, atom, thermo);

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
  integrate.initialize(N, time_step, group, atom);
  measure.initialize(number_of_steps, time_step, integrate, group, atom, force);

#ifdef USE_PLUMED
  if (measure.plmd.use_plumed == 1) {
    measure.plmd.init(time_step, integrate.temperature);
  }
#endif

  clock_t time_begin = clock();
  double initial_time_step = time_step;

  for (int step = 0; step < number_of_steps; ++step) {

    calculate_time_step(
      max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;

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

#ifdef USE_PLUMED
    if (measure.plmd.use_plumed == 1 && (step % measure.plmd.interval) == 0) {
      measure.plmd.process(
        box, thermo, atom.position_per_atom, atom.force_per_atom, atom.virial_per_atom);
    }
#endif

    electron_stop.compute(atom);

    integrate.compute2(time_step, double(step) / number_of_steps, group, box, atom, thermo);

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

    velocity.correct_velocity(
      step,
      atom.cpu_mass,
      atom.position_per_atom,
      atom.cpu_position_per_atom,
      atom.cpu_velocity_per_atom,
      atom.velocity_per_atom);

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
      fflush(stdout);
    }
  }

  print_line_1();
  clock_t time_finish = clock();
  double time_used = (time_finish - time_begin) / (double)CLOCKS_PER_SEC;

  printf("Time used for this run = %g second.\n", time_used);
  double run_speed = N * (number_of_steps / time_used);
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  print_line_2();

  measure.finalize(integrate, number_of_steps, time_step, integrate.temperature2, box.get_volume(),atom.number_of_beads);

  electron_stop.finalize();
  integrate.finalize();
  velocity.finalize();
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const char* param[22]; // never use more than 19 parameters
  for (int n = 0; n < num_param; ++n) {
    param[n] = tokens[n].c_str();
  }

  if (strcmp(param[0], "potential") == 0) {
    force.parse_potential(param, num_param, box, atom.type.size());
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
    integrate.parse_ensemble(box, param, num_param, group);
  } else if (strcmp(param[0], "time_step") == 0) {
    parse_time_step(param, num_param);
  } else if (strcmp(param[0], "correct_velocity") == 0) {
    parse_correct_velocity(param, num_param);
  } else if (strcmp(param[0], "dump_thermo") == 0) {
    measure.dump_thermo.parse(param, num_param);
  } else if (strcmp(param[0], "dump_position") == 0) {
    measure.dump_position.parse(param, num_param, group);
  } else if (strcmp(param[0], "dump_netcdf") == 0) {
#ifdef USE_NETCDF
    measure.dump_netcdf.parse(param, num_param);
#else
    PRINT_INPUT_ERROR("dump_netcdf is available only when USE_NETCDF flag is set.\n");
#endif
  } else if (strcmp(param[0], "plumed") == 0) {
#ifdef USE_PLUMED
    measure.plmd.parse(param, num_param);
#else
    PRINT_INPUT_ERROR("plumed is available only when USE_PLUMED flag is set.\n");
#endif
  } else if (strcmp(param[0], "dump_restart") == 0) {
    measure.dump_restart.parse(param, num_param);
  } else if (strcmp(param[0], "dump_velocity") == 0) {
    measure.dump_velocity.parse(param, num_param, group);
  } else if (strcmp(param[0], "dump_force") == 0) {
    measure.dump_force.parse(param, num_param, group);
  } else if (strcmp(param[0], "dump_exyz") == 0) {
    measure.dump_exyz.parse(param, num_param);
  } else if (strcmp(param[0], "dump_beads") == 0) {
    measure.dump_beads.parse(param, num_param);
  } else if (strcmp(param[0], "dump_observer") == 0) {
    measure.dump_observer.parse(param, num_param);
  } else if (strcmp(param[0], "active") == 0) {
    measure.active.parse(param, num_param);
  } else if (strcmp(param[0], "compute_dos") == 0) {
    measure.dos.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_sdc") == 0) {
    measure.sdc.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_msd") == 0) {
    measure.msd.parse(param, num_param, group);
  }  else if (strcmp(param[0], "compute_rdf") == 0) {
    measure.rdf.parse(param, num_param, box, number_of_types, number_of_steps);
  } else if (strcmp(param[0], "compute_hac") == 0) {
    measure.hac.parse(param, num_param);
  } else if (strcmp(param[0], "compute_viscosity") == 0) {
    measure.viscosity.parse(param, num_param);
  } else if (strcmp(param[0], "compute_hnemd") == 0) {
    measure.hnemd.parse(param, num_param);
  } else if (strcmp(param[0], "compute_hnemdec") == 0) {
    measure.hnemdec.parse(param, num_param);
  } else if (strcmp(param[0], "compute_shc") == 0) {
    measure.shc.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_gkma") == 0) {
    measure.parse_compute_gkma(param, num_param, number_of_types);
  } else if (strcmp(param[0], "compute_hnema") == 0) {
    measure.parse_compute_hnema(param, num_param, number_of_types);
  } else if (strcmp(param[0], "deform") == 0) {
    integrate.parse_deform(param, num_param);
  } else if (strcmp(param[0], "compute") == 0) {
    measure.compute.parse(param, num_param, group);
  } else if (strcmp(param[0], "fix") == 0) {
    integrate.parse_fix(param, num_param, group);
  } else if (strcmp(param[0], "move") == 0) {
    integrate.parse_move(param, num_param, group);
  } else if (strcmp(param[0], "electron_stop") == 0) {
    electron_stop.parse(param, num_param, atom.number_of_atoms, number_of_types);
  } else if (strcmp(param[0], "run") == 0) {
    parse_run(param, num_param);
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}

void Run::parse_velocity(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("velocity should have 1 parameter.\n");
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
    atom.velocity_per_atom);
}

void Run::parse_correct_velocity(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("correct_velocity should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &velocity.velocity_correction_interval)) {
    PRINT_INPUT_ERROR("velocity correction interval should be an integer.\n");
  }
  if (velocity.velocity_correction_interval <= 0) {
    PRINT_INPUT_ERROR("velocity correction interval should be positive.\n");
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

  bool compute_hnemd = measure.hnemd.compute || (measure.modal_analysis.compute &&
                                                 measure.modal_analysis.method == HNEMA_METHOD);
  force.set_hnemd_parameters(
    compute_hnemd, measure.hnemd.fe_x, measure.hnemd.fe_y, measure.hnemd.fe_z);

  if (!compute_hnemd && (measure.hnemdec.compute != -1)) {
    if ((measure.hnemdec.compute > number_of_types) || (measure.hnemdec.compute < 0)) {
      PRINT_INPUT_ERROR(
        "compute for HNEMDEC should be an integer number between 0 and number_of_types.\n");
    }
    force.set_hnemdec_parameters(
      measure.hnemdec.compute,
      measure.hnemdec.fe_x,
      measure.hnemdec.fe_y,
      measure.hnemdec.fe_z,
      atom.cpu_mass,
      atom.cpu_type,
      atom.cpu_type_size,
      integrate.temperature1);
  }

  // set target temperature for temperature-dependent NEP
  force.temperature = integrate.temperature1;
  force.delta_T = (integrate.temperature2 - integrate.temperature1) / number_of_steps;

  perform_a_run();
}

static __global__ void gpu_pressure_triclinic(
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
    if (box.triclinic == 0) {
      PRINT_INPUT_ERROR("Cannot use orthogonal box with shear deformation.");
    }
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
    if (box.triclinic == 0) {
      deformation_matrix[d][d] = (box.cpu_h[d] + deformation_matrix[d][d]) / box.cpu_h[d];
    } else {
      deformation_matrix[d][d] =
        (box.cpu_h[d * 3 + d] + deformation_matrix[d][d]) / box.cpu_h[d * 3 + d];
    }
  }

  printf("    Deformation matrix =\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", deformation_matrix[d1][d2]);
    }
    printf("\n");
  }

  if (box.triclinic == 0) {
    printf("    Original box lengths are\n");
    printf("        Lx = %g A\n", box.cpu_h[0]);
    printf("        Ly = %g A\n", box.cpu_h[1]);
    printf("        Lz = %g A\n", box.cpu_h[2]);
  } else {
    printf("    Original box h = [a, b, c] is\n");
    for (int d1 = 0; d1 < 3; ++d1) {
      printf("        ");
      for (int d2 = 0; d2 < 3; ++d2) {
        printf("%g ", box.cpu_h[d1 * 3 + d2]);
      }
      printf("\n");
    }
  }

  if (box.triclinic == 0) {
    for (int d = 0; d < 3; ++d) {
      box.cpu_h[d] *= deformation_matrix[d][d];
      box.cpu_h[d + 3] = box.cpu_h[d] * 0.5;
    }
  } else {
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
  }

  const int number_of_atoms = atom.position_per_atom.size() / 3;
  gpu_pressure_triclinic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
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
  CUDA_CHECK_KERNEL

  if (box.triclinic == 0) {
    printf("    Changed box lengths are\n");
    printf("        Lx = %g A\n", box.cpu_h[0]);
    printf("        Ly = %g A\n", box.cpu_h[1]);
    printf("        Lz = %g A\n", box.cpu_h[2]);
  } else {
    printf("    Changed box h = [a, b, c] is\n");
    for (int d1 = 0; d1 < 3; ++d1) {
      printf("        ");
      for (int d2 = 0; d2 < 3; ++d2) {
        printf("%g ", box.cpu_h[d1 * 3 + d2]);
      }
      printf("\n");
    }
  }
}
