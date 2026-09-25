/*
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Portions copyright 2017-2026 Zheyong Fan and respective contributors.
 * Copyright (c) 2026 Wisevolve.
 *
 * This license applies only to this public WPE interface header. It does not
 * grant rights to the Wisevolve Potential Engine implementation.
 */

#pragma once

/*
 * Wisevolve Potential Engine (WPE) C API.
 *
 * This interface is the dedicated GPUMD/WPE integration boundary. It is not a
 * generic GPUMD potential-backend ABI.
 *
 * Binary compatibility is carried by the shared-library SONAME
 * (libwisevolve_potential.so.1). Public records keep struct_size so fields can
 * be appended without adding a separate API/ABI version field to each call.
 *
 * Contract:
 *   - pure C ABI exported by libwisevolve_potential.so
 *   - opaque WPE context
 *   - no GPUMD C++ types cross this header
 *   - stage activation receives only high-level run facts
 *   - bind provides stable GPUMD-owned input/output surfaces
 *   - WPE owns execution policy, including virial routing
 *   - compute(context) is the hot path
 */

#include <stdint.h>
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define WPE_API __attribute__((visibility("default")))
#else
#define WPE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WpeContext WpeContext;

typedef enum WpeStatus {
  WPE_OK = 0,
  WPE_ERR_INVALID_ARGUMENT = 1,
  WPE_ERR_UNSUPPORTED_MODEL = 2,
  WPE_ERR_CUDA = 3,
  WPE_ERR_NEIGHBOR_CAPACITY = 4,
  WPE_ERR_INTERNAL = 5
} WpeStatus;

typedef struct WpeInfo {
  uint32_t struct_size;
  uint32_t reserved0;
  const char* name;
  const char* version;
  const char* provider;
  const char* library_soname;

  /* Optional release metadata. NULL means not published for this release. */
  const char* project_url;
  const char* citation;
  const char* citation_doi;
} WpeInfo;

typedef enum WpeQnepKspaceMethod {
  WPE_QNEP_KSPACE_PPPM = 0,
  WPE_QNEP_KSPACE_EWALD = 1
} WpeQnepKspaceMethod;

typedef enum WpeStageKind {
  WPE_STAGE_RUN = 1,
  WPE_STAGE_MINIMIZE = 2,
  WPE_STAGE_PHONON = 3,
  WPE_STAGE_COHESIVE = 4,
  WPE_STAGE_ELASTIC = 5
} WpeStageKind;

enum WpeStageFactBits {
  WPE_STAGE_HAS_SYSTEM_VIRIAL_CONSUMER = 1ull << 0,
  WPE_STAGE_HAS_ATOM_VIRIAL_CONSUMER = 1ull << 1,
  WPE_STAGE_HAS_HNEMD = 1ull << 2,
  WPE_STAGE_HAS_HNEMDEC = 1ull << 3,
  WPE_STAGE_NEED_CHARGE_OUTPUT = 1ull << 4,
  WPE_STAGE_NEED_BEC_OUTPUT = 1ull << 5,
  WPE_STAGE_HAS_MC = 1ull << 6,
  WPE_STAGE_MINIMIZE_BOX_CHANGE = 1ull << 7,
  WPE_STAGE_PIMD_ECO_CONFIGURATION = 1ull << 8
};

typedef struct WpeStageInfo {
  uint32_t struct_size;
  uint32_t stage_kind;
  uint32_t reserved0;
  uint32_t reserved1;
  const char* ensemble_name;
  uint64_t facts;
  int32_t number_of_atoms;
  uint32_t pimd_beads;
  uint32_t qnep_kspace_method;
  uint32_t dftd3_enabled;
  const char* dftd3_functional;
  float dftd3_rc_potential;
  float dftd3_rc_coordination_number;
  uint32_t reserved2;
  uint32_t reserved3;
} WpeStageInfo;

typedef struct WpeCreateInfo {
  uint32_t struct_size;
  uint32_t reserved0;
  const char* potential_path;
  int32_t number_of_atoms;
  uint32_t reserved1;
} WpeCreateInfo;

/* Generic model metadata required by the open Potential facade. Execution
 * routing and capabilities remain private to the closed backend. */
typedef struct WpeCreateResult {
  uint32_t struct_size;
  uint32_t reserved0;
  double cutoff;
} WpeCreateResult;

typedef struct WpeBinding {
  uint32_t struct_size;
  int32_t number_of_atoms;
  uint32_t reserved0;
  uint32_t reserved1;

  const void* position_device;       /* double[3N], GPUMD SoA */
  const void* type_device;           /* int[N] */

  void* force_device;                /* double[3N], host-prepared output */
  void* potential_device;            /* double[N], host-prepared output */
  void* atomic_virial_device;        /* double[9N], host-prepared/zeroed output */
  void* total_virial_device;         /* optional double[6]; NULL = WPE-owned scratch */

  const double* box_h18_host;        /* borrowed Box::cpu_h */
  const int32_t* pbc_x_host;
  const int32_t* pbc_y_host;
  const int32_t* pbc_z_host;

  void* charge_device;               /* float[N] or NULL */
  void* bec_device;                  /* float[9N] or NULL */
} WpeBinding;

/*
 * ABI 1 minimum accepted struct sizes. These values are frozen for SONAME 1.
 * Future ABI-1-compatible fields may only be appended; these V1 minimums
 * must never be advanced to include appended fields.
 */
#define WPE_INFO_V1_MIN_SIZE \
  ((uint32_t)(offsetof(WpeInfo, library_soname) + sizeof(const char*)))
#define WPE_CREATE_INFO_V1_MIN_SIZE \
  ((uint32_t)(offsetof(WpeCreateInfo, reserved1) + sizeof(uint32_t)))
#define WPE_CREATE_RESULT_V1_MIN_SIZE \
  ((uint32_t)(offsetof(WpeCreateResult, cutoff) + sizeof(double)))
#define WPE_STAGE_INFO_V1_MIN_SIZE \
  ((uint32_t)(offsetof(WpeStageInfo, reserved3) + sizeof(uint32_t)))
#define WPE_BINDING_V1_MIN_SIZE \
  ((uint32_t)(offsetof(WpeBinding, bec_device) + sizeof(void*)))

WPE_API WpeStatus wpe_get_info(WpeInfo* info);

WPE_API WpeStatus wpe_create(
  const WpeCreateInfo* info,
  WpeCreateResult* result,
  WpeContext** context);

WPE_API WpeStatus wpe_activate_stage(
  WpeContext* context,
  const WpeStageInfo* stage);

WPE_API WpeStatus wpe_bind(
  WpeContext* context,
  const WpeBinding* binding);

WPE_API WpeStatus wpe_compute(WpeContext* context);

WPE_API void wpe_destroy(WpeContext* context);

#ifdef __cplusplus
}
#endif
