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
#include "force/force.cuh"
#include "force/validate.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/neighbor.cuh"
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
  double max_distance_per_step, GPU_Vector<double>& velocity_per_atom, double& time_step)
{
  if (max_distance_per_step <= 0.0) {
    return;
  }
  const int N = velocity_per_atom.size() / 3;
  double* gpu_v2_max;
  CHECK(cudaGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N, (N - 1) / 1024 + 1, velocity_per_atom.data(), velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2, gpu_v2_max);
  CUDA_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(cudaMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), cudaMemcpyDeviceToHost));
  double cpu_v_max = sqrt(cpu_v2_max[0]);
  double time_step_min = max_distance_per_step / cpu_v_max;

  if (time_step_min < time_step) {
    time_step = time_step_min;
  }
}

Run::Run(char* input_dir)
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  print_line_2();

  initialize_position(
    input_dir, N, has_velocity_in_xyz, number_of_types, box, neighbor, group, atom);

  allocate_memory_gpu(N, neighbor, group, atom, thermo);

#ifndef USE_FCP // the FCP does not use a neighbor list at all
  neighbor.find_neighbor(/*is_first=*/true, box, atom.position_per_atom);
#endif

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  print_line_2();

  execute_run_in(input_dir);
}

void Run::execute_run_in(char* input_dir)
{
  char file_run[200];
  strcpy(file_run, input_dir);
  strcat(file_run, "/run.in");
  char* input = get_file_contents(file_run);
  char* input_ptr = input;      // Keep the pointer in order to free later
  const int max_num_param = 20; // never use more than 19 parameters
  int num_param;
  char* param[max_num_param];

  print_line_1();
  printf("Started executing the commands in run.in.\n");
  print_line_2();

  while (input_ptr) {
    input_ptr = row_find_param(input_ptr, param, &num_param);
    if (num_param == 0) {
      continue;
    }
    parse_one_keyword(param, num_param, input_dir);
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  print_line_2();

  free(input); // Free the input file contents
}

void Run::perform_a_run(char* input_dir)
{
  integrate.initialize(N, time_step, group);
  measure.initialize(input_dir, number_of_steps, time_step, box, neighbor, group, force, atom);

  clock_t time_begin = clock();

  for (int step = 0; step < number_of_steps; ++step) {

    calculate_time_step(max_distance_per_step, atom.velocity_per_atom, time_step);
    global_time += time_step;

#ifndef USE_FCP // the FCP does not use a neighbor list at all
    if (neighbor.update) {
      neighbor.find_neighbor(/*is_first=*/false, box, atom.position_per_atom, force.rc_max);
    }
#endif

    integrate.compute1(time_step, double(step) / number_of_steps, group, box, atom, thermo);

    force.compute(
      box, atom.position_per_atom, atom.type, group, neighbor, atom.potential_per_atom,
      atom.force_per_atom, atom.virial_per_atom);

    integrate.compute2(time_step, double(step) / number_of_steps, group, box, atom, thermo);

    measure.process(
      input_dir, number_of_steps, step, integrate.fixed_group, global_time, integrate.temperature2,
      integrate.ensemble->energy_transferred, box, neighbor, group, thermo, atom);

    velocity.correct_velocity(
      step, atom.cpu_mass, atom.position_per_atom, atom.cpu_position_per_atom,
      atom.cpu_velocity_per_atom, atom.velocity_per_atom);

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
    }
  }

// only for my test
#ifdef VALIDATE_FORCE
  validate_force(
    box, atom.position_per_atom, group, atom.type, atom.potential_per_atom, atom.force_per_atom,
    atom.virial_per_atom, neighbor, &force);
#endif

  print_line_1();
  clock_t time_finish = clock();
  double time_used = (time_finish - time_begin) / (double)CLOCKS_PER_SEC;
  if (neighbor.update) {
    printf("Number of neighbor list updates for this run = %d.\n", neighbor.number_of_updates);
  } else {
    printf("!!! WARNING: You have not asked to update the neighbor list for this run. Please make "
           "sure this is what you intended.\n");
  }

  if (neighbor.number_of_updates > 0) {
    printf("    Calculated maximum number of neighbors for this run = %d.\n", neighbor.max_NN);
    printf("    The 'MN' parameter you set in xyz.in = %d.\n", neighbor.MN);
    printf("    You can consider increasing/decreasing 'MN' based on the information above.\n");
  }
  printf("Time used for this run = %g second.\n", time_used);
  double run_speed = N * (number_of_steps / time_used);
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  print_line_2();

  measure.finalize(input_dir, number_of_steps, time_step, integrate.temperature2, box.get_volume());

  integrate.finalize();
  neighbor.finalize();
  velocity.finalize();
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(char** param, int num_param, char* input_dir)
{
  if (strcmp(param[0], "potential") == 0) {
    force.parse_potential(
      param, num_param, input_dir, box, neighbor, atom.cpu_type, atom.cpu_type_size);
  } else if (strcmp(param[0], "minimize") == 0) {
    Minimize minimize;
    minimize.parse_minimize(
      param, num_param, force, box, atom.position_per_atom, atom.type, group, neighbor,
      atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
  } else if (strcmp(param[0], "compute_phonon") == 0) {
    Hessian hessian;
    hessian.parse(param, num_param);
    hessian.compute(
      input_dir, force, box, atom.cpu_position_per_atom, atom.position_per_atom, atom.type, group,
      neighbor, atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
  } else if (strcmp(param[0], "compute_cohesive") == 0) {
    Cohesive cohesive;
    cohesive.parse(param, num_param, 0);
    cohesive.compute(
      input_dir, box, atom.position_per_atom, atom.type, group, neighbor, atom.potential_per_atom,
      atom.force_per_atom, atom.virial_per_atom, force);
  } else if (strcmp(param[0], "compute_elastic") == 0) {
    Cohesive cohesive;
    cohesive.parse(param, num_param, 1);
    cohesive.compute(
      input_dir, box, atom.position_per_atom, atom.type, group, neighbor, atom.potential_per_atom,
      atom.force_per_atom, atom.virial_per_atom, force);
  } else if (strcmp(param[0], "change_box") == 0) {
    parse_change_box(param, num_param);
  } else if (strcmp(param[0], "velocity") == 0) {
    parse_velocity(param, num_param);
  } else if (strcmp(param[0], "ensemble") == 0) {
    integrate.parse_ensemble(box, param, num_param, group);
  } else if (strcmp(param[0], "time_step") == 0) {
    parse_time_step(param, num_param);
  } else if (strcmp(param[0], "neighbor") == 0) {
    parse_neighbor(param, num_param);
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
  } else if (strcmp(param[0], "dump_restart") == 0) {
    measure.dump_restart.parse(param, num_param);
  } else if (strcmp(param[0], "dump_velocity") == 0) {
    measure.dump_velocity.parse(param, num_param, group);
  } else if (strcmp(param[0], "dump_force") == 0) {
    measure.dump_force.parse(param, num_param, group);
  } else if (strcmp(param[0], "dump_exyz") == 0) {
    measure.dump_exyz.parse(param, num_param);
  } else if (strcmp(param[0], "compute_dos") == 0) {
    measure.dos.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_sdc") == 0) {
    measure.sdc.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_cvac") == 0) {
    measure.cvac.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_hac") == 0) {
    measure.hac.parse(param, num_param);
  } else if (strcmp(param[0], "compute_hnemd") == 0) {
    measure.hnemd.parse(param, num_param);
  } else if (strcmp(param[0], "compute_shc") == 0) {
    measure.shc.parse(param, num_param, group);
  } else if (strcmp(param[0], "compute_shc_harmonic") == 0) {
    measure.shc_harmonic.parse(param, num_param, group);
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
  } else if (strcmp(param[0], "run") == 0) {
    parse_run(param, num_param, input_dir);
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}

void Run::parse_velocity(char** param, int num_param)
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
    has_velocity_in_xyz, initial_temperature, atom.cpu_mass, atom.cpu_position_per_atom,
    atom.cpu_velocity_per_atom, atom.velocity_per_atom);
}

void Run::parse_correct_velocity(char** param, int num_param)
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

void Run::parse_time_step(char** param, int num_param)
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

void Run::parse_run(char** param, int num_param, char* input_dir)
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

  perform_a_run(input_dir);
}

void Run::parse_neighbor(char** param, int num_param)
{
  neighbor.update = 0;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("neighbor should have 1 parameter.\n");
  }
  if (strcmp(param[1], "off") == 0) {
    printf("Neighbor list will NOT be updated.\n");
  } else {
    PRINT_INPUT_ERROR(
      "Starting from GPUMD-v3.3, the grammar of the 'neighbor' keyword has been\n"
      "changed: The neighbor list will be updated with a skin distance of 1 A by\n"
      "default for each run and one can use 'neighbor off' to turn it off. When\n"
      "one knows there is no atom diffusion, it is better to turn off the neighbor\n"
      "list update to achieve the best computational speed.\n");
  }
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

void Run::parse_change_box(char** param, int num_param)
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
    if (!is_valid_real(param[4], &deformation_matrix[0][1])) {
      PRINT_INPUT_ERROR("box change parameter in xy should be a number.");
    }
    if (!is_valid_real(param[5], &deformation_matrix[0][2])) {
      PRINT_INPUT_ERROR("box change parameter in xz should be a number.");
    }
    if (!is_valid_real(param[6], &deformation_matrix[1][2])) {
      PRINT_INPUT_ERROR("box change parameter in yz should be a number.");
    }
    deformation_matrix[1][0] = deformation_matrix[0][1];
    deformation_matrix[2][0] = deformation_matrix[0][2];
    deformation_matrix[2][1] = deformation_matrix[1][2];
  }

  printf("Change box:\n");
  printf("    in xx by %g A.\n", deformation_matrix[0][0]);
  printf("    in yy by %g A.\n", deformation_matrix[1][1]);
  printf("    in zz by %g A.\n", deformation_matrix[2][2]);
  printf("    in xy and yx by strain %g.\n", deformation_matrix[0][1]);
  printf("    in xz and zx by strain %g.\n", deformation_matrix[0][2]);
  printf("    in yz and zy by strain %g.\n", deformation_matrix[1][2]);

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
    number_of_atoms, deformation_matrix[0][0], deformation_matrix[0][1], deformation_matrix[0][2],
    deformation_matrix[1][0], deformation_matrix[1][1], deformation_matrix[1][2],
    deformation_matrix[2][0], deformation_matrix[2][1], deformation_matrix[2][2],
    atom.position_per_atom.data(), atom.position_per_atom.data() + number_of_atoms,
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

  exit(1);
}