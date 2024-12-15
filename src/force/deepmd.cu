#ifdef USE_TENSORFLOW
#include "DeepPot.h"
#include "deepmd.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <sstream>
#include <chrono>

#define MAX_STRING_LENGTH 256
#define MAX_TYPES 100


static const char cite_user_deepmd_package[] =
    "USE Deep Potential in GPUMD package, please cite:\n\n"
    "@article{Wang_ComputPhysCommun_2018_v228_p178,\n"
    "  author = {Wang, Han and Zhang, Linfeng and Han, Jiequn and E, Weinan},\n"
    "  doi = {10.1016/j.cpc.2018.03.016},\n"
    "  url = {https://doi.org/10.1016/j.cpc.2018.03.016},\n"
    "  year = 2018,\n"
    "  month = {jul},\n"
    "  publisher = {Elsevier {BV}},\n"
    "  volume = 228,\n"
    "  journal = {Comput. Phys. Commun.},\n"
    "  title = {{DeePMD-kit: A deep learning package for many-body potential "
    "energy representation and molecular dynamics}},\n"
    "  pages = {178--184}\n"
    "}\n"
    "@misc{Zeng_JChemPhys_2023_v159_p054801,\n"
    "  title  = {{DeePMD-kit v2: A software package for deep potential "
    "models}},\n"
    "  author =   {Jinzhe Zeng and Duo Zhang and Denghui Lu and Pinghui Mo and "
    "Zeyu Li\n"
    "         and Yixiao Chen and Mari{\\'a}n Rynik and Li'ang Huang and Ziyao "
    "Li and \n"
    "         Shaochen Shi and Yingze Wang and Haotian Ye and Ping Tuo and "
    "Jiabin\n"
    "         Yang and Ye Ding and Yifan Li and Davide Tisi and Qiyu Zeng and "
    "Han \n"
    "         Bao and Yu Xia and Jiameng Huang and Koki Muraoka and Yibo Wang "
    "and \n"
    "         Junhan Chang and Fengbo Yuan and Sigbj{\\o}rn L{\\o}land Bore "
    "and "
    "Chun\n"
    "         Cai and Yinnian Lin and Bo Wang and Jiayan Xu and Jia-Xin Zhu "
    "and \n"
    "         Chenxing Luo and Yuzhi Zhang and Rhys E A Goodall and Wenshuo "
    "Liang\n"
    "         and Anurag Kumar Singh and Sikai Yao and Jingchao Zhang and "
    "Renata\n"
    "         Wentzcovitch and Jiequn Han and Jie Liu and Weile Jia and Darrin "
    "M\n"
    "         York and Weinan E and Roberto Car and Linfeng Zhang and Han "
    "Wang},\n"
    "  journal =  {J. Chem. Phys.},\n"
    "  volume =   159,\n"
    "  issue =    5,  \n"
    "  year =    2023,\n"
    "  pages  =   054801,\n"
    "  doi =      {10.1063/5.0155600},\n"
    "}\n\n";


DEEPMD::DEEPMD(const char* deep_pot_file, int num_atoms)
{
  std::cout << cite_user_deepmd_package << "\n";

  ener_unit_cvt_factor=1;      // 1.0 / 8.617343e-5;
  dist_unit_cvt_factor=1;      // 1;
  force_unit_cvt_factor=1;     // ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor=1;    // ener_unit_cvt_factor
  single_model = true;
  atom_spin_flag = false;

  rc = 0.;
  numb_types = 0;
  numb_types_spin = 0;
  numb_models = 0;
  stdf_comm_buff_size = 0;
  eps = 0.;
  eps_v = 0.;
  scale = NULL;
  do_ttm = false;
  do_compute_fparam = false;
  do_compute_aparam = false;
  multi_models_mod_devi = false;
  multi_models_no_mod_devi = false;

  initialize_deepmd(deep_pot_file);
}


DEEPMD::~DEEPMD(void)
{
  // nothing
}

void DEEPMD::initialize_deepmd(const char* deep_pot_file)
{
  int num_gpus;
  CHECK(gpuGetDeviceCount(&num_gpus));
  printf("\nUse %s deepmd potential.\n\n", deep_pot_file);
  deep_pot.init(deep_pot_file, num_gpus);
  double rc = deep_pot.cutoff();
  int numb_types = deep_pot.numb_types();
  int numb_types_spin = deep_pot.numb_types_spin();
  int dim_fparam = deep_pot.dim_fparam();
  int dim_aparam = deep_pot.dim_aparam();

  char* type_map[numb_types];
  std::string type_map_str;
  deep_pot.get_type_map(type_map_str);
  // convert the string to a vector of strings
  std::istringstream iss(type_map_str);
  std::string type_name;
  int i = 0;
  while (iss >> type_name) {
    if (i >= numb_types) break;
    type_map[i] = strdup(type_name.c_str());
    i++;
  }

  printf("=======================================================\n");
  printf("  ++ cutoff: %f ++ \n", rc);
  printf("  ++ numb_types: %d ++ \n", numb_types);
  printf("  ++ numb_types_spin: %d ++ \n", numb_types_spin);
  printf("  ++ dim_fparam: %d ++ \n", dim_fparam);
  printf("  ++ dim_aparam: %d ++ \n  ++ ", dim_aparam);
  for (int i = 0; i < numb_types; ++i)
  {
    printf("%s ", type_map[i]);
  }
  printf("++\n=======================================================\n");
}

#define BLOCK_SIZE_FORCE 128
#define USE_FIXED_NEIGHBOR 1
#define BIG_ILP_CUTOFF_SQUARE 50.0

// Refs:
// pair_deepmd.cpp :: 600 line
// deep_pot.compute(dener, dforce, dvirial, deatom, dvatom, dcoord,
//                  dtype, dbox, nghost, lmp_list, ago, fparam,
//                  daparam);
// output >> dener: cell energy need to compute (double * 1)
// output >> dforce: atomic force need to compute (double * natoms*3)
// output >> dvirial: cell virial need to compute (double * 9)
// output >> deatom: atomic energy need to compute (double * natoms)
// output >> dvatom: atomic virial need to compute (double * natoms*9)
// input >> dtype: atomic type (int * natoms)
// input >> dbox: atomic force need to compute (int * 9)
// input >> nghost: cell virial need to compute (ghost number of atoms: ref from lammps)
// input >> lmp_list: atomic energy need to compute (lammps neighbor list structure)
// input >> ago: 0 to compute (maybe always should to be 0)
// input >> fparam: not need
// input >> daparam: not need

void DEEPMD::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{

  const int ghost_z_cell_num = box.pbc_z ? 3 : 1;
  const int ghost_y_cell_num = box.pbc_y ? 3 : 1;
  const int ghost_x_cell_num = box.pbc_x ? 3 : 1;
  const int ghost_cell_num = ghost_x_cell_num * ghost_y_cell_num * ghost_z_cell_num;

  const int real_num_of_atoms = type.size();
  const double rc = deep_pot.cutoff(); // Use cutoff from DeepPot
  const int all_num_of_atoms = real_num_of_atoms * ghost_cell_num;

  deepmd_data.ghost_type.resize(all_num_of_atoms, 0);
  deepmd_data.ghost_position.resize(all_num_of_atoms * 3, -1);

  const int z_lim = box.pbc_z ? 1 : 0;
  const int y_lim = box.pbc_y ? 1 : 0;
  const int x_lim = box.pbc_x ? 1 : 0;
  int half_const_cell;
  if (z_lim + y_lim + x_lim == 3) half_const_cell = 13;
  if (z_lim + y_lim + x_lim == 2) half_const_cell = 4;
  if (z_lim + y_lim + x_lim == 1) half_const_cell = 1;
  if (z_lim + y_lim + x_lim == 0) half_const_cell = 0;

  find_neighbor_deepmd(N1, N2, half_const_cell, rc, box, type, position_per_atom,
                       deepmd_data.ghost_type, deepmd_data.ghost_position);

  std::vector<int> gpumd_cpu_ghost_type(all_num_of_atoms);
  cudaMemcpy(gpumd_cpu_ghost_type.data(), deepmd_data.ghost_type.data(),
             all_num_of_atoms * sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<double> gpumd_cpu_ghost_position(all_num_of_atoms * 3);
  cudaMemcpy(gpumd_cpu_ghost_position.data(), deepmd_data.ghost_position.data(),
             all_num_of_atoms * 3 * sizeof(double), cudaMemcpyDeviceToHost);

  // Put the real atom to the first
  if (z_lim + y_lim + x_lim > 0) {
    for (int i = 0; i < real_num_of_atoms; i++) {  // read atom position from position
      int ti = half_const_cell * real_num_of_atoms + i;
      int select_type = gpumd_cpu_ghost_type[ti];
      gpumd_cpu_ghost_type[ti] = gpumd_cpu_ghost_type[i];
      gpumd_cpu_ghost_type[i] = select_type;
      for (int j = 0; j < 3; ++j) {
        double select_py = gpumd_cpu_ghost_position[all_num_of_atoms*j + ti];
        gpumd_cpu_ghost_position[all_num_of_atoms*j + ti] = gpumd_cpu_ghost_position[all_num_of_atoms*j + i];
        gpumd_cpu_ghost_position[all_num_of_atoms*j + i] = select_py;
      }
    }
  }
  deepmd_ghost_data.NN.resize(all_num_of_atoms);
  deepmd_ghost_data.NL.resize(all_num_of_atoms * 512); // 1024 is the largest supported by CUDA
  deepmd_ghost_data.cell_count.resize(all_num_of_atoms);
  deepmd_ghost_data.cell_count_sum.resize(all_num_of_atoms);
  deepmd_ghost_data.cell_contents.resize(all_num_of_atoms);
  deepmd_ghost_data.ghost_type.resize(all_num_of_atoms);
  deepmd_ghost_data.ghost_position.resize(all_num_of_atoms*3);
  deepmd_ghost_data.ghost_type.copy_from_host(gpumd_cpu_ghost_type.data());
  deepmd_ghost_data.ghost_position.copy_from_host(gpumd_cpu_ghost_position.data());

  // defind a new box include the ghost cell.
  for (int i = 0; i < 9; ++i) ghost_box.cpu_h[i] = box.cpu_h[i];
  ghost_box.pbc_x = box.pbc_x;
  ghost_box.pbc_y = box.pbc_y;
  ghost_box.pbc_z = box.pbc_z;
  ghost_box.triclinic = box.triclinic;
  if (box.triclinic) {
    printf("The deepmd in gpumd is not support the triclinic box.\n");
  } else {
    if (ghost_box.pbc_x) ghost_box.cpu_h[0] *= 3;
    if (ghost_box.pbc_y) ghost_box.cpu_h[1] *= 3;
    if (ghost_box.pbc_z) ghost_box.cpu_h[2] *= 3;
  }

  find_neighbor(N1, N1+all_num_of_atoms, rc, ghost_box, deepmd_ghost_data.ghost_type, deepmd_ghost_data.ghost_position,
                deepmd_ghost_data.cell_count, deepmd_ghost_data.cell_count_sum,
                deepmd_ghost_data.cell_contents, deepmd_ghost_data.NN, deepmd_ghost_data.NL);

  // Creating LAMMPS-compatible data structures
  int num_of_all_atoms = deepmd_ghost_data.NN.size();
  int total_all_neighs = deepmd_ghost_data.NL.size();

  // Allocate lmp_ilist and lmp_numneigh
  int* lmp_ilist = (int*)malloc(num_of_all_atoms*sizeof(int));
  int* lmp_numneigh = (int*)malloc(num_of_all_atoms*sizeof(int));
  cudaMemcpy(lmp_numneigh, deepmd_ghost_data.NN.data(), num_of_all_atoms*sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<int> cpu_NL(total_all_neighs);
  cudaMemcpy(cpu_NL.data(), deepmd_ghost_data.NL.data(), total_all_neighs * sizeof(int), cudaMemcpyDeviceToHost);

  int* neigh_storage = (int*)malloc(total_all_neighs*sizeof(int));
  int** lmp_firstneigh = (int**)malloc(num_of_all_atoms*sizeof(int*));
  int offset = 0;
  for (int i = 0; i < num_of_all_atoms; i++) {
    lmp_ilist[i] = i;
    lmp_firstneigh[i] = &neigh_storage[offset];
    for (int j = 0; j < lmp_numneigh[i]; j++) {
        neigh_storage[offset + j] = cpu_NL[i + j * num_of_all_atoms]; // Copy in column-major order
    }
    offset += lmp_numneigh[i];
  }
  // Constructing a neighbor list in LAMMPS format
  deepmd_compat::InputNlist lmp_list(num_of_all_atoms, lmp_ilist, lmp_numneigh, lmp_firstneigh);

  // Initialize DeepPot computation variables
  std::vector<double> dp_ene_all(1, 0.0);
  std::vector<double> dp_ene_atom(all_num_of_atoms, 0.0);
  std::vector<double> dp_force(all_num_of_atoms * 3, 0.0);
  std::vector<double> dp_vir_all(9, 0.0);
  std::vector<double> dp_vir_atom(all_num_of_atoms * 9, 0.0);

  // Prepare the box dimensions
  std::vector<double> dp_box(9, 0);
  if (ghost_box.triclinic == 0) {
    dp_box[0] = ghost_box.cpu_h[0];
    dp_box[4] = ghost_box.cpu_h[1];
    dp_box[8] = ghost_box.cpu_h[2];
  } else {
    dp_box[0] = ghost_box.cpu_h[0];
    dp_box[4] = ghost_box.cpu_h[1];
    dp_box[8] = ghost_box.cpu_h[2];
    dp_box[7] = ghost_box.cpu_h[7];
    dp_box[6] = ghost_box.cpu_h[6];
    dp_box[3] = ghost_box.cpu_h[3];
  }

  // Allocate and copy position_per_atom to CPU
  std::vector<double> dp_cpu_ghost_position(num_of_all_atoms*3);
  for (int i = 0; i < num_of_all_atoms; i++) {
    dp_cpu_ghost_position.data()[i * 3] = gpumd_cpu_ghost_position.data()[i]; // + box.thickness_x;
    dp_cpu_ghost_position.data()[i * 3 + 1] = gpumd_cpu_ghost_position.data()[i + num_of_all_atoms]; // + box.thickness_y;
    dp_cpu_ghost_position.data()[i * 3 + 2] = gpumd_cpu_ghost_position.data()[i + 2 * num_of_all_atoms]; // + box.thickness_z;
  }

  // to calculate the atomic force and energy from deepot
  bool do_ghost = true;
  if (do_ghost)  {
    if (single_model) {
      if (! atom_spin_flag) {
          const int reduced_ghost_num_of_atom = all_num_of_atoms - real_num_of_atoms;
	  //deep_pot.compute(dp_ene_all, dp_force, dp_vir_all,dp_cpu_ghost_position, gpumd_cpu_ghost_type,dp_box);
          deep_pot.compute(dp_ene_all, dp_force, dp_vir_all, dp_ene_atom, dp_vir_atom, 
              dp_cpu_ghost_position, gpumd_cpu_ghost_type, dp_box,
              reduced_ghost_num_of_atom, lmp_list, 0);
      }
    }
  }

  std::vector<double> gpumd_ene_atom(real_num_of_atoms, 0.0);
  std::vector<double> gpumd_force(real_num_of_atoms * 3, 0.0);
  std::vector<double> virial_per_atom_cpu(real_num_of_atoms * 6, 0.0);
  const int const_cell = half_const_cell * 2 + 1;
  for (int i = 0; i < const_cell; i++) {  // read atom position from position
    for (int g = 0; g < real_num_of_atoms; g++) {
      gpumd_ene_atom[g] += dp_ene_atom[i * real_num_of_atoms + g] * ener_unit_cvt_factor;
      for (int o = 0; o < 3; o++)
        gpumd_force.data()[g + o * real_num_of_atoms] += dp_force.data()[i*real_num_of_atoms*3+3*g+o] * force_unit_cvt_factor;
      virial_per_atom_cpu.data()[g + 0 * real_num_of_atoms] += dp_vir_atom.data()[i*real_num_of_atoms*9+9*g+0] * virial_unit_cvt_factor;
      virial_per_atom_cpu.data()[g + 1 * real_num_of_atoms] += dp_vir_atom.data()[i*real_num_of_atoms*9+9*g+4] * virial_unit_cvt_factor;
      virial_per_atom_cpu.data()[g + 2 * real_num_of_atoms] += dp_vir_atom.data()[i*real_num_of_atoms*9+9*g+8] * virial_unit_cvt_factor;
      virial_per_atom_cpu.data()[g + 3 * real_num_of_atoms] += dp_vir_atom.data()[i*real_num_of_atoms*9+9*g+1] * virial_unit_cvt_factor;
      virial_per_atom_cpu.data()[g + 4 * real_num_of_atoms] += dp_vir_atom.data()[i*real_num_of_atoms*9+9*g+2] * virial_unit_cvt_factor;
      virial_per_atom_cpu.data()[g + 5 * real_num_of_atoms] += dp_vir_atom.data()[i*real_num_of_atoms*9+9*g+5] * virial_unit_cvt_factor;
    }
  }

  potential_per_atom.copy_from_host(gpumd_ene_atom.data());
  force_per_atom.copy_from_host(gpumd_force.data());
  virial_per_atom.copy_from_host(virial_per_atom_cpu.data());

  free(lmp_ilist);
  free(lmp_numneigh);
  free(neigh_storage);
  free(lmp_firstneigh);
}
#endif
