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
    global_time += time_step;

#ifndef USE_FCP // the FCP does not use a neighbor list at all
    if (neighbor.update) {
      neighbor.find_neighbor(/*is_first=*/false, box, atom.position_per_atom);
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
#if 0
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
  if (num_param != 2) {
    PRINT_INPUT_ERROR("time_step should have 1 parameter.\n");
  }
  if (!is_valid_real(param[1], &time_step)) {
    PRINT_INPUT_ERROR("time_step should be a real number.\n");
  }
  printf("Time step for this run is %g fs.\n", time_step);
  time_step /= TIME_UNIT_CONVERSION;
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
  neighbor.update = 1;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("neighbor should have 1 parameter.\n");
  }
  if (!is_valid_real(param[1], &neighbor.skin)) {
    PRINT_INPUT_ERROR("neighbor list skin should be a number.\n");
  }
  printf("Build neighbor list with a skin of %g A.\n", neighbor.skin);

  // change the cutoff
  neighbor.rc = force.rc_max + neighbor.skin;
}
