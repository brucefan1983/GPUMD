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

/**
 * NNAP interface for GPUMD
 * @author liqa
 */

#ifdef USE_NNAP
#include "nnap.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define BLOCK_SIZE_NNAP 256

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

static jboolean initJVM_(JNIEnv **rEnv) {
  JavaVM *tJVM = NULL;
  jsize tNVMs;
  JNI_GetCreatedJavaVMs(&tJVM, 1, &tNVMs);
  if (tJVM != NULL) {
    (tJVM)->AttachCurrentThreadAsDaemon((void**)rEnv, NULL);
    return JNI_TRUE;
  }
  JavaVMInitArgs tVMArgs;
  JavaVMOption tOptions[4];
  tOptions[0].optionString = (char *)JVM_CLASS_PATH; // here, path/to/jse/lib/jse-all.jar
  tOptions[1].optionString = (char *)"-Xmx1g";
  tOptions[2].optionString = (char *)"--enable-native-access=ALL-UNNAMED";
  tOptions[3].optionString = (char *)"-Djdk.lang.processReaperUseDefaultStackSize=true"; // avoid fail of thread create
  tVMArgs.version = JNI_VERSION_1_6;
  tVMArgs.nOptions = 4;
  tVMArgs.options = tOptions;
  tVMArgs.ignoreUnrecognized = JNI_TRUE;
  
  jint tOut = JNI_CreateJavaVM(&tJVM, (void**)rEnv, &tVMArgs);
  return tOut==JNI_OK ? JNI_TRUE : JNI_FALSE;
}
static jboolean exceptionCheck_(JNIEnv *aEnv) {
  if (aEnv->ExceptionCheck()) {
    aEnv->ExceptionDescribe();
    aEnv->ExceptionClear();
    return JNI_TRUE;
  }
  return JNI_FALSE;
}

static jclass NNAP_CLAZZ = NULL;

static jboolean cacheJClass_(JNIEnv *aEnv) {
  if (NNAP_CLAZZ == NULL) {
    jclass clazz = aEnv->FindClass("jsex/nnap/NNAP");
    if (aEnv->ExceptionCheck()) return JNI_FALSE;
    NNAP_CLAZZ = (jclass)aEnv->NewGlobalRef(clazz);
    aEnv->DeleteLocalRef(clazz);
  }
  return JNI_TRUE;
}
static void uncacheJClass_(JNIEnv *aEnv) {
  if (NNAP_CLAZZ != NULL) {
    aEnv->DeleteGlobalRef(NNAP_CLAZZ);
    NNAP_CLAZZ = NULL;
  }
}

static jmethodID sInit = 0;
static jmethodID sClose = 0;
static jmethodID sRcutMax = 0;
static jmethodID sComputeGPUMD = 0;

static jobject newJObject_(JNIEnv *aEnv, const char* filename) {
  jobject rOut = NULL;
  
  jstring tJFileName = aEnv->NewStringUTF(filename);
  jstring tArchStr = aEnv->NewStringUTF("cuda");
  if (sInit || (sInit = aEnv->GetMethodID(NNAP_CLAZZ, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V"))) {
    rOut = aEnv->NewObject(NNAP_CLAZZ, sInit, tJFileName, tArchStr);
  }
  aEnv->DeleteLocalRef(tJFileName);
  aEnv->DeleteLocalRef(tArchStr);
  
  return rOut;
}
static double rcutMax_(JNIEnv *aEnv, jobject aSelf) {
  if (sRcutMax || (sRcutMax = aEnv->GetMethodID(NNAP_CLAZZ, "rcutMax", "()D"))) {
    return aEnv->CallDoubleMethod(aSelf, sRcutMax);
  }
  return 0.0;
}
static void close_(JNIEnv *aEnv, jobject aSelf) {
  if (sClose || (sClose = aEnv->GetMethodID(NNAP_CLAZZ, "close", "()V"))) {
    aEnv->CallVoidMethod(aSelf, sClose);
  }
}
static void computeGPUMD_(JNIEnv *aEnv, jobject aSelf,
    const int number_of_particles,
    const int N1,
    const int N2,
    const int nn_max,
    const int *g_neighbor_number,
    const int *g_neighbor_list,
    const float *nl_dx,
    const float *nl_dy,
    const float *nl_dz,
    const int *g_type,
    double *g_fx,
    double *g_fy,
    double *g_fz,
    double *g_virial,
    double *g_potential) {
  if (sComputeGPUMD || (sComputeGPUMD = aEnv->GetMethodID(NNAP_CLAZZ, "computeGPUMD", "(IIIIJJJJJJJJJJJ)V"))) {
    return aEnv->CallVoidMethod(aSelf, sComputeGPUMD,
      (jint)number_of_particles,
      (jint)N1,
      (jint)N2,
      (jint)nn_max,
      (jlong)(intptr_t)g_neighbor_number,
      (jlong)(intptr_t)g_neighbor_list,
      (jlong)(intptr_t)nl_dx,
      (jlong)(intptr_t)nl_dy,
      (jlong)(intptr_t)nl_dz,
      (jlong)(intptr_t)g_type,
      (jlong)(intptr_t)g_fx,
      (jlong)(intptr_t)g_fy,
      (jlong)(intptr_t)g_fz,
      (jlong)(intptr_t)g_virial,
      (jlong)(intptr_t)g_potential
    );
  }
}

static bool check_if_small_box(const double rc, const Box& box)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.0 * rc) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.0 * rc) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.0 * rc) {
    is_small_box = true;
  }
  return is_small_box;
}


NNAP::NNAP(const char* setting_file, const char* nnap_file, int num_atoms)
{
  // init GPUMD settings, like zbl
  std::ifstream input(setting_file);
  if (!input.is_open()) {
    std::cout << "Failed to open " << setting_file << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nnap.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nnap") {
    zbl.enabled = false;
  } else if (tokens[0] == "nnap_zbl") {
    zbl.enabled = true;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NNAP model. Expected: nnap, nnap_zbl"
              << std::endl;
    exit(1);
  }
  num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (num_types == 1) {
    printf("Use the NNAP potential with %d atom type.\n", num_types);
  } else {
    printf("Use the NNAP potential with %d atom types.\n", num_types);
  }
  // zbl
  if (zbl.enabled) {
    if (tokens.size() != 2 + num_types) {
      std::cout << "The first line of nnap.txt should have " << num_types << " atom symbols."
                << std::endl;
      exit(1);
    }
    for (int n = 0; n < num_types; ++n) {
      int atomic_number = 0;
      for (int m = 0; m < NUM_ELEMENTS; ++m) {
        if (tokens[2 + n] == ELEMENTS[m]) {
          atomic_number = m + 1;
          break;
        }
      }
      zbl.atomic_numbers[n] = atomic_number;
      printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
    }
    // zbl params
    tokens = get_tokens(input);
    if ((tokens.size() != 3 && tokens.size() != 4) || tokens[0] != "zbl") {
      std::cout << "This line should be zbl <rc_inner> <rc_outer> [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      std::cout << "Flexible ZBL is invalid for nnap_zbl" << std::endl;
      exit(1);
    }
    if (tokens.size() == 4) {
      zbl.typewise_cutoff_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
      zbl.use_typewise_cutoff = true;
      printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
        zbl.typewise_cutoff_factor);
    } else {
      printf(
        "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
        zbl.rc_inner,
        zbl.rc_outer);
    }
  }
  // nn_max
  tokens = get_tokens(input);
  if (tokens.size() > 0 && tokens[0] == "nn_max") {
    if (tokens.size() != 2) {
      std::cout << "This line should be nn_max <nn_max>." << std::endl;
      exit(1);
    }
    int nn_max_ = get_int_from_token(tokens[1], __FILE__, __LINE__);
    printf("    nn_max = %d.\n", nn_max_);
    if (nn_max_ > 819) {
      std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
                << std::endl;
      exit(1);
    }
    nn_max = int(ceil(nn_max_ * 1.25));
    printf("    enlarged nn_max = %d.\n", nn_max);
  }
  input.close();


  // init jni env
  if (mEnv == NULL) {
    jboolean tSuc = initJVM_(&mEnv);
    if (!tSuc) PRINT_INPUT_ERROR("Fail to init jvm");
    if (mEnv == NULL) PRINT_INPUT_ERROR("Fail to get jni env");
  }
  // init java NNAP object
  jboolean tSuc = cacheJClass_(mEnv);
  if (exceptionCheck_(mEnv) || !tSuc) PRINT_INPUT_ERROR("Fail to cache class of java NNAP");
  jobject tObj = newJObject_(mEnv, nnap_file);
  if (exceptionCheck_(mEnv) || tObj==NULL) PRINT_INPUT_ERROR("Fail to create java NNAP object");
  if (mCore != NULL) mEnv->DeleteGlobalRef(mCore);
  mCore = mEnv->NewGlobalRef(tObj);
  mEnv->DeleteLocalRef(tObj);
  
  // get rcut
  rc = rcutMax_(mEnv, mCore);
  if (exceptionCheck_(mEnv)) PRINT_INPUT_ERROR("Fail to get rcutMax");
  neighbor.initialize(rc, num_atoms, nn_max);
  if (!(std::isfinite(rc) && rc > 0.0)) {
    PRINT_INPUT_ERROR("Invalid NNAP cutoff returned by rcutMax()");
  }
  printf("    max cutoff = %g A.\n", rc);
  
  // init nl cache
  nl_dx.resize(num_atoms * nn_max);
  nl_dy.resize(num_atoms * nn_max);
  nl_dz.resize(num_atoms * nn_max);
}

NNAP::~NNAP(void)
{
  if (mCore != NULL && mEnv != NULL) {
    close_(mEnv, mCore);
    // only check, no error on destructor
    exceptionCheck_(mEnv);
    
    mEnv->DeleteGlobalRef(mCore);
    mCore = NULL;
    uncacheJClass_(mEnv);
  }
}


static __global__ void valid_nl_(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float* __restrict__ nl_dx,
  float* __restrict__ nl_dy,
  float* __restrict__ nl_dz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      
      nl_dx[n1 + number_of_particles * i1] = x12;
      nl_dy[n1 + number_of_particles * i1] = y12;
      nl_dz[n1 + number_of_particles * i1] = z12;
    }
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NNAP::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* __restrict__ nl_dx,
  const float* __restrict__ nl_dy,
  const float* __restrict__ nl_dz,
  const int* __restrict__ g_type,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
    float s_fx = 0.0f, s_fy = 0.0f, s_fz = 0.0f;
    float s_sxx = 0.0f, s_sxy = 0.0f, s_sxz = 0.0f;
    float s_syx = 0.0f, s_syy = 0.0f, s_syz = 0.0f;
    float s_szx = 0.0f, s_szy = 0.0f, s_szz = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = nl_dx[n1 + N * i1];
      float y12 = nl_dy[n1 + N * i1];
      float z12 = nl_dz[n1 + N * i1];
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      float rc_inner = zbl.rc_inner;
      float rc_outer = zbl.rc_outer;
      if (zbl.use_typewise_cutoff) {
        // zi and zj start from 1, so need to minus 1 here
        rc_outer = min(
          (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * zbl.typewise_cutoff_factor,
          rc_outer);
        rc_inner = 0.0f;
      }
      find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}


void NNAP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  if (check_if_small_box(rc, box)) {
    printf("Cannot use small box for NNAP.\n");
    exit(1);
  }

  const int number_of_atoms = type.size();
  
  // build neighbor list here
  neighbor.find_neighbor_global(
    rc,
    box, 
    type, 
    position
  );
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_NNAP + 1;
  valid_nl_<<<grid_size, BLOCK_SIZE_NNAP>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    neighbor.NN.data(),
    neighbor.NL.data(),
    position.data(),
    position.data() + number_of_atoms,
    position.data() + number_of_atoms * 2,
    nl_dx.data(),
    nl_dy.data(),
    nl_dz.data()
  );

  // invoke NNAP_cuda.computeGPUMD(...)
  computeGPUMD_(mEnv, mCore,
    number_of_atoms,
    N1,
    N2,
    nn_max,
    neighbor.NN.data(),
    neighbor.NL.data(),
    nl_dx.data(),
    nl_dy.data(),
    nl_dz.data(),
    type.data(),
    force.data(),
    force.data() + number_of_atoms,
    force.data() + 2 * number_of_atoms,
    virial.data(),
    potential.data()
  );
  if (exceptionCheck_(mEnv)) PRINT_INPUT_ERROR("Fail when call computeGPUMD");

  // zbl support
  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE_NNAP>>>(
      number_of_atoms,
      zbl,
      N1,
      N2,
      box,
      neighbor.NN.data(),
      neighbor.NL.data(),
      nl_dx.data(),
      nl_dy.data(),
      nl_dz.data(),
      type.data(),
      force.data(),
      force.data() + number_of_atoms,
      force.data() + 2 * number_of_atoms,
      virial.data(),
      potential.data());
    GPU_CHECK_KERNEL
  }
}
#endif
