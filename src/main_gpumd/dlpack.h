/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief C header of DLPack
 */
#ifndef DLPACK_H_
#define DLPACK_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief The current version of DLPack */
#define DLPACK_VERSION 80

/*! \brief DLPack version: 0.8.x */
#define DLPACK_VERSION_MAJOR 0
#define DLPACK_VERSION_MINOR 8
#define DLPACK_VERSION_PATCH 0

/*! \brief The default device type for CPU */
#define DLPACK_DEFAULT_DEVICE_TYPE kDLCPU

#ifdef __cplusplus
#define DLPACK_INLINE inline
#else
#ifdef _MSC_VER
#define DLPACK_INLINE __inline
#else
#define DLPACK_INLINE static inline
#endif
#endif

/*!
 * \brief The device type in DLDevice.
 */
typedef enum {
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*! \brief Pinned CUDA CPU memory by cudaMallocHost */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*! \brief ROCm host pinned memory for AMD GPU */
  kDLROCMHost = 11,
  /*! \brief Reserved for extension, used for quick test */
  kDLExtDev = 12,
  /*! \brief CUDA managed/unified memory allocated by cudaMallocManaged */
  kDLCUDAManaged = 13,
  /*! \brief Unified shared memory allocated on a oneAPI non-partititioned
   * device. Call to oneAPI runtime is required to determine the device
   * type, the USM pointer type, and the allocation type
   * (device, host, or shared). It is compatible with DPC++ and Level Zero
   * interfaces.
   */
  kDLOneAPI = 14,
  /*! \brief GPU support for next generation WebGPU standard. */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
} DLDeviceType;

/*!
 * \brief The device information.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*! \brief The device index. For CPU device, this should be set to 0,
   * and the implementation can use the index to select the physical
   * device or stream.
   */
  int device_id;
} DLDevice;

/*!
 * \brief The data type code.
 */
typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  /*! \brief Opaque handle type, reserved for testing purposes.
   *  The handle can have an arbitrary value, and the implementation
   *  may use the type as an alias for a user-defined type that is
   *  not part of the DLPack specification.
   */
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
} DLDataTypeCode;

/*!
 * \brief The data type.
 *
 *  This is the type of the elements stored in the tensor.
 *  The index refers to the DLDataTypeCode.
 *  The bits refers to the number of bits for each element.
 *  The lanes refers to the number of elements in a vector.
 */
typedef struct {
  uint8_t code;    // DLDataTypeCode value cast to uint8_t
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

/*!
 * \brief The shape of the tensor.
 */
typedef int64_t* DLShape;

/*!
 * \brief A struct that represents a tensor.
 */
typedef struct {
  /*! \brief The data pointer.
   *
   *  It is the responsibility of the user to ensure that the data pointer
   *  is valid and points to the correct device memory.
   */
  void* data;
  /*! \brief The device context. */
  DLDevice device;
  /*! \brief The number of dimensions. */
  int ndim;
  /*! \brief The data type of the elements. */
  DLDataType dtype;
  /*! \brief The shape of the tensor. */
  DLShape shape;
  /*! \brief The strides of the tensor (in number of elements, not bytes).
   *
   *  If the strides are NULL, the tensor is compact (dense) and the strides
   *  are inferred from the shape.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data.
   *
   *  This is used when the data pointer points to a non-zero offset
   *  of the allocated memory.
   */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief A structure to hold the managed tensor.
 *
 *  This is the entry point for the DLPack protocol.
 *  The deleter is responsible for freeing the DLManagedTensor and the
 *  associated memory.
 */
typedef struct DLManagedTensor DLManagedTensor;

/*!
 * \brief The destructor function for DLManagedTensor.
 */
typedef void (*DLManagedTensorDeleter)(DLManagedTensor* self);

struct DLManagedTensor {
  /*! \brief The tensor data. */
  DLTensor dl_tensor;
  /*! \brief The context of the tensor.
   *
   *  The context can be used to store additional information about the
   *  tensor, such as the memory manager or the device context.
   */
  void* manager_ctx;
  /*! \brief The destructor function for the managed tensor.
   *
   *  The deleter is called when the tensor is no longer needed.
   *  It is the responsibility of the user to ensure that the deleter
   *  is called exactly once.
   */
  DLManagedTensorDeleter deleter;
};

/*!
 * \brief The version of the DLPack C API.
 */
typedef struct {
  int major;
  int minor;
  int patch;
} DLPackVersion;

/*!
 * \brief Get the DLPack version.
 *
 * \return The DLPack version.
 */
DLPACK_INLINE DLPackVersion DLPackGetVersion(void) {
  DLPackVersion version;
  version.major = DLPACK_VERSION_MAJOR;
  version.minor = DLPACK_VERSION_MINOR;
  version.patch = DLPACK_VERSION_PATCH;
  return version;
}

#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif  // DLPACK_H_
