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
#include <cstdint>

#define BLOCK_SIZE_NL_NNAP 256
#define MAX_NEIGH_NUM_NNAP 512

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
    jclass clazz = aEnv->FindClass("jsex/nnap/NNAP_cuda"); // Interim version developed for gpu version 
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
static jmethodID sShutdown = 0;
static jmethodID sRcutMax = 0;
static jmethodID sComputeGPUMD = 0;

static jobject newJObject_(JNIEnv *aEnv, const char* filename) {
  jobject rOut = NULL;
  
  jstring tJFileName = aEnv->NewStringUTF(filename);
  if (sInit || (sInit = aEnv->GetMethodID(NNAP_CLAZZ, "<init>", "(Ljava/lang/String;)V"))) {
    rOut = aEnv->NewObject(NNAP_CLAZZ, sInit, tJFileName);
  }
  aEnv->DeleteLocalRef(tJFileName);
  
  return rOut;
}
static double rcutMax_(JNIEnv *aEnv, jobject aSelf) {
  if (sRcutMax || (sRcutMax = aEnv->GetMethodID(NNAP_CLAZZ, "rcutMax", "()D"))) {
    return aEnv->CallDoubleMethod(aSelf, sRcutMax);
  }
  return 0.0;
}
static void shutdown_(JNIEnv *aEnv, jobject aSelf) {
  if (sShutdown || (sShutdown = aEnv->GetMethodID(NNAP_CLAZZ, "shutdown", "()V"))) {
    aEnv->CallVoidMethod(aSelf, sShutdown);
  }
}
static void computeGPUMD_(JNIEnv *aEnv, jobject aSelf,
    const int number_of_particles,
    const int N1,
    const int N2,
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
      (jint)MAX_NEIGH_NUM_NNAP,
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

NNAP::NNAP(const char* filename, int num_atoms)
{
  // init jni env
  if (mEnv == NULL) {
    jboolean tSuc = initJVM_(&mEnv);
    if (!tSuc) PRINT_INPUT_ERROR("Fail to init jvm");
    if (mEnv == NULL) PRINT_INPUT_ERROR("Fail to get jni env");
  }
  // init java NNAP object
  jboolean tSuc = cacheJClass_(mEnv);
  if (exceptionCheck_(mEnv) || !tSuc) PRINT_INPUT_ERROR("Fail to cache class of java NNAP");
  jobject tObj = newJObject_(mEnv, filename);
  if (exceptionCheck_(mEnv) || tObj==NULL) PRINT_INPUT_ERROR("Fail to create java NNAP object");
  if (mCore != NULL) mEnv->DeleteGlobalRef(mCore);
  mCore = mEnv->NewGlobalRef(tObj);
  mEnv->DeleteLocalRef(tObj);
  
  // get rcut
  rc = rcutMax_(mEnv, mCore);
  if (exceptionCheck_(mEnv)) PRINT_INPUT_ERROR("Fail to get rcutMax");
  neighbor.initialize(rc, num_atoms, MAX_NEIGH_NUM_NNAP); // TODO: ?
  if (!(std::isfinite(rc) && rc > 0.0)) {
    PRINT_INPUT_ERROR("Invalid NNAP cutoff returned by rcutMax()");
  }
  
  // init nl cache
  nl_dx.resize(num_atoms * MAX_NEIGH_NUM_NNAP);
  nl_dy.resize(num_atoms * MAX_NEIGH_NUM_NNAP);
  nl_dz.resize(num_atoms * MAX_NEIGH_NUM_NNAP);
}

NNAP::~NNAP(void)
{
  if (mCore != NULL && mEnv != NULL) {
    shutdown_(mEnv, mCore);
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

void NNAP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int number_of_atoms = type.size();
  
  // build neighbor list here
  neighbor.find_neighbor_global(
    rc,
    box, 
    type, 
    position
  );
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_NL_NNAP + 1;
  valid_nl_<<<grid_size, BLOCK_SIZE_NL_NNAP>>>(
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
  
  // TODO: type map? anyway
  // invoke NNAP_cuda.computeGPUMD(...)
  computeGPUMD_(mEnv, mCore,
    number_of_atoms,
    N1,
    N2,
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
}
#endif
