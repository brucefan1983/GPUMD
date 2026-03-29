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
#include "nnap.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"


static jboolean initJVM_(JNIEnv **rEnv) {
  JavaVM *tJVM = NULL;
  jsize tNVMs;
  JNI_GetCreatedJavaVMs(&tJVM, 1, &tNVMs);
  if (tJVM != NULL) {
    (tJVM)->AttachCurrentThreadAsDaemon((void**)rEnv, NULL);
    return JNI_TRUE;
  }
  JavaVMInitArgs tVMArgs;
  JavaVMOption tOptions[3];
  tOptions[0].optionString = (char *)JVM_CLASS_PATH; // here, path/to/jse/lib/jse-all.jar
  tOptions[1].optionString = (char *)"-Xmx1g";
  tOptions[2].optionString = (char *)"--enable-native-access=ALL-UNNAMED";
  tVMArgs.version = JNI_VERSION_1_6;
  tVMArgs.nOptions = 3;
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
    jclass clazz = aEnv->FindClass("jsex/nnap/NNAP2"); // Interim version developed for gpu version 
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
  if (exceptionCheck_(mEnv) || !tSuc) PRINT_INPUT_ERROR("Fail to cache class of java LmpFix");
  jobject tObj = newJObject_(mEnv, filename);
  if (exceptionCheck_(mEnv) || tObj==NULL) PRINT_INPUT_ERROR("Fail to create java LmpFix object");
  if (mCore != NULL) mEnv->DeleteGlobalRef(mCore);
  mCore = mEnv->NewGlobalRef(tObj);
  mEnv->DeleteLocalRef(tObj);
  
  // get rcut
  rc = rcutMax_(mEnv, mCore);
  if (exceptionCheck_(mEnv)) PRINT_INPUT_ERROR("Fail to get rcutMax");
  neighbor.initialize(rc, num_atoms, 700); // 700?
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
    position_per_atom);
  
  // TODO: invoke NNAP.computeGPUMD(...)
}
