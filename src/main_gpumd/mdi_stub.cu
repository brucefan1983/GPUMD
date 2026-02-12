/*
  MDI integration for GPUMD
  Build with: make USE_MDI=1 (this sets -DUSE_MDI and links -lmdi)

  This file implements a functional MDI engine entry point for GPUMD.
  It allows GPUMD to act as an MD engine receiving commands from
  a driver (e.g., VASP via MDI). Supported commands:
    <NATOMS, >COORDS, <COORDS, <FORCES, >FORCES, <ENERGY, >ENERGY, EXIT

  Based on MDI developer guide:
    https://molssi-mdi.github.io/MDI_Library/developer_guide/engine_tutorial.html
*/

#include "run.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifdef USE_MDI

/* Include MDI header with fallback to stub */
#include "mdi_fallback.h"

#endif

#ifdef USE_MDI

/* Helper to extract MDI options from command line */
static char* get_mdi_options(int argc, char* argv[])
{
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--mdi") == 0 && i + 1 < argc) {
      return argv[i + 1];
    }
  }
  return nullptr;
}

extern "C" int mdi_engine_main(struct Run* run, int argc, char* argv[])
{
  printf("MDI mode requested. MDI integration compiled in.\n");

  if (!run) {
    fprintf(stderr, "ERROR: Run instance is NULL.\n");
    return EXIT_FAILURE;
  }

  /* Get MDI options from command line */
  char* mdi_options = get_mdi_options(argc, argv);
  if (!mdi_options) {
    fprintf(stderr, "ERROR: --mdi flag present but no MDI options provided.\n");
    return EXIT_FAILURE;
  }

  printf("MDI options: %s\n", mdi_options);

  /* Initialize MDI library */
  int mdi_ret = MDI_Init(&argc, &argv);
  if (mdi_ret != 0) {
    fprintf(stderr, "ERROR: MDI_Init failed with return code %d.\n", mdi_ret);
    return EXIT_FAILURE;
  }
  printf("MDI initialized.\n");

  /* Initialize GPUMD for MDI operation */
  run->mdi_initialize_for_mdi();
  printf("GPUMD initialized for MDI.\n");

  /* Register supported commands */
  MDI_Register_node("@DEFAULT");
  MDI_Register_command("@DEFAULT", "<NATOMS");
  MDI_Register_command("@DEFAULT", ">COORDS");
  MDI_Register_command("@DEFAULT", "<COORDS");
  MDI_Register_command("@DEFAULT", ">FORCES");
  MDI_Register_command("@DEFAULT", "<FORCES");
  MDI_Register_command("@DEFAULT", "<ENERGY");
  MDI_Register_command("@DEFAULT", ">ENERGY");
  MDI_Register_command("@DEFAULT", "EXIT");
  printf("Registered MDI commands.\n");

  /* Accept communicator from driver */
  MDI_Comm mdi_comm = MDI_COMM_NULL;
  mdi_ret = MDI_Accept_communicator(&mdi_comm);
  if (mdi_ret != 0) {
    fprintf(stderr, "ERROR: MDI_Accept_communicator failed with return code %d.\n", mdi_ret);
    return EXIT_FAILURE;
  }
  printf("Accepted MDI communicator from driver.\n");

  /* Main MDI command loop - loop for multiple steps */
  char command[MDI_COMMAND_LENGTH];
  bool exit_flag = false;
  int natoms = run->mdi_get_natoms();
  printf("System has %d atoms.\n", natoms);

  while (!exit_flag) {
    /* Receive command from driver */
    mdi_ret = MDI_Recv_command(command, mdi_comm);
    if (mdi_ret != 0) {
      fprintf(stderr, "Error in MDI_Recv_Command: Unable to receive command\n");
      break;
    }

    printf("[MDI] Received command: %s\n", command);

    /* Handle each command */
    if (strcmp(command, "<NATOMS") == 0) {
      /* Driver requests number of atoms */
      int expected = 1;
      mdi_ret = MDI_Send(&natoms, expected, MDI_INT, mdi_comm);
      printf("    [MDI] MDI_Send natoms expected=%d returned=%d\n", expected, mdi_ret);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Send failed for natoms (return code %d).\n", mdi_ret);
        break;
      }
      printf("  -> Sent natoms=%d\n", natoms);
    } else if (strcmp(command, ">COORDS") == 0) {
      /* Driver sends atomic coordinates (N*3 doubles) */
      std::vector<double> coords(natoms * 3);
      int expected = natoms * 3;
      mdi_ret = MDI_Recv(&coords[0], expected, MDI_DOUBLE, mdi_comm);
      printf("    [MDI] MDI_Recv >COORDS expected=%d returned=%d\n", expected, mdi_ret);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Recv failed for coordinates (return code %d).\n", mdi_ret);
        break;
      }
      run->mdi_set_positions(&coords[0]);
      printf("  -> Set positions from driver\n");
    } else if (strcmp(command, "<COORDS") == 0) {
      /* Driver requests atomic coordinates */
      std::vector<double> coords;
      run->mdi_get_positions(coords);
      int expected = natoms * 3;
      mdi_ret = MDI_Send(&coords[0], expected, MDI_DOUBLE, mdi_comm);
      printf("    [MDI] MDI_Send <COORDS expected=%d returned=%d\n", expected, mdi_ret);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Send failed for coordinates (return code %d).\n", mdi_ret);
        break;
      }
      printf("  -> Sent positions to driver\n");
    } else if (strcmp(command, "<FORCES") == 0) {
      /* Driver requests forces (compute first) */
      run->mdi_compute_forces();
      std::vector<double> forces;
      run->mdi_get_forces(forces);
      int expected = natoms * 3;
      mdi_ret = MDI_Send(&forces[0], expected, MDI_DOUBLE, mdi_comm);
      printf("    [MDI] MDI_Send <FORCES expected=%d returned=%d\n", expected, mdi_ret);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Send failed for forces (return code %d).\n", mdi_ret);
        break;
      }
      printf("  -> Sent forces to driver\n");
    } else if (strcmp(command, ">FORCES") == 0) {
      /* Driver sends forces (e.g., from QM code for hybrid MD) */
      std::vector<double> forces(natoms * 3);
      int expected = natoms * 3;
      mdi_ret = MDI_Recv(&forces[0], expected, MDI_DOUBLE, mdi_comm);
      printf("    [MDI] MDI_Recv >FORCES expected=%d returned=%d\n", expected, mdi_ret);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Recv failed for forces (return code %d).\n", mdi_ret);
        break;
      }
      printf("  -> Received external forces from driver (first 6 values):\n");
      for (int i = 0; i < std::min(natoms * 3, 6); ++i) {
        printf("    forces[%d] = %f\n", i, forces[i]);
      }
      /* Set the external forces in GPUMD */
      run->mdi_set_forces(&forces[0]);
      /* Integrate MD step with the QM forces */
      run->mdi_step_one();
      printf("  -> Integrated MD step with external forces\n");
      /* Dump updated positions to stdout for debugging */
      std::vector<double> newpos;
      run->mdi_get_positions(newpos);
      printf("  -> Positions after integration (first 6 values):\n");
      for (int i = 0; i < std::min((int)newpos.size(), 6); ++i) {
        printf("    pos[%d] = %f\n", i, newpos[i]);
      }
    } else if (strcmp(command, "<ENERGY") == 0) {
      /* Driver requests total energy */
      run->mdi_compute_forces();
      double total_energy = 0.0; /* TODO: extract real energy sum from GPUMD */
      mdi_ret = MDI_Send(&total_energy, 1, MDI_DOUBLE, mdi_comm);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Send failed for energy (return code %d).\n", mdi_ret);
        break;
      }
      printf("  -> Sent energy to driver: %f\n", total_energy);
    } else if (strcmp(command, ">ENERGY") == 0) {
      /* Driver sends energy (e.g., from QM code) */
      double energy;
      mdi_ret = MDI_Recv(&energy, 1, MDI_DOUBLE, mdi_comm);
      if (mdi_ret != 0) {
        fprintf(stderr, "ERROR: MDI_Recv failed for energy (return code %d).\n", mdi_ret);
        break;
      }
      printf("  -> Received energy from driver: %f\n", energy);
    } else if (strcmp(command, "EXIT") == 0) {
      printf("  -> EXIT command received.\n");
      exit_flag = true;
    } else {
      printf("  WARNING: Unknown command '%s', ignoring.\n", command);
    }
  }

  printf("MDI command loop exited successfully.\n");
  return EXIT_SUCCESS;
}

#else

/* If compiled without USE_MDI, provide a dummy symbol to avoid link errors */
int mdi_engine_main(int argc, char* argv[])
{
  fprintf(stderr, "mdi_engine_main called but GPUMD not built with USE_MDI.\n");
  return EXIT_FAILURE;
}

#endif
