/*
    gpumd_python.cpp

    pybind11 wrapper for GPUMD that exposes simulation data as DLPack
    capsules and supports a per-step Python callback.
    Compile this file with a C++ compiler (g++/clang++) and link it with
    the GPUMD object files to produce a loadable Python module named
    ``gpumd``.

    Copyright 2026 Jaafar Mehrez
    (Shanghai Jiao Tong University, Shanghai, China;
     HPQC Labs, Waterloo, Canada;
     jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

    SPDX-License-Identifier: MIT
*/

// IMPORTANT: All CUDA / standard library headers must come BEFORE any
// GPUMD .cuh file.  The GPUMD headers use CUDA host types (cudaError_t,
// __global__, etc.) that are only visible when cuda_runtime.h has been
// processed first.
#include <cuda_runtime.h>

#include <cctype>   // std::tolower
#include <cstring>  // std::memset
#include <fstream>  // std::ifstream
#include <memory>   // std::unique_ptr, std::make_unique

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "dlpack.h"
#include "gpumd_python_kernels.cuh"
#include "run.cuh"
#include "utilities/gpu_vector.cuh"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: wrap a GPU_Vector as a DLPack capsule.
//
// The capsule owns a small DLManagedTensor wrapper but does NOT own the
// underlying GPU memory (GPUMD does).  When the consumer (JAX) calls the
// deleter, we only free the wrapper struct.
// ---------------------------------------------------------------------------
template <typename T>
py::capsule gpu_vector_to_dlpack(
  T* data,
  int64_t size,
  int64_t components,                     // 3 for per-atom vectors, 1 for scalars
  DLDataTypeCode type_code,
  uint8_t bits)
{
  // Build shape / strides so that the tensor is in compact C-order,
  // which JAX/XLA accepts without the "non-default layout" error.
  //
  // GPUMD stores per-atom vectors in SOA layout:
  //   [x0..xN-1, y0..yN-1, z0..zN-1]
  // This is compact C-order when viewed as shape (3, N) with strides (N, 1).
  // The PySAGES backend will transpose back to (N, 3) via .T.
  int ndim = (components > 1) ? 2 : 1;
  int64_t* shape = new int64_t[ndim];
  int64_t* strides = new int64_t[ndim];
  if (ndim == 2) {
    shape[0] = components;   // 3
    shape[1] = size;         // N
    strides[0] = size;       // N
    strides[1] = 1;          // 1
  } else {
    shape[0] = size;
    strides[0] = 1;
  }

  auto* manager = new DLManagedTensor;
  // Zero the struct to avoid garbage in padding bytes that JAX might read.
  std::memset(manager, 0, sizeof(DLManagedTensor));
  manager->dl_tensor.data = static_cast<void*>(data);
  manager->dl_tensor.device = DLDevice{kDLCUDA, 0};
  manager->dl_tensor.ndim = ndim;
  // Explicitly set each field of DLDataType to avoid C++ brace-init issues
  // with enum + mixed-size integer fields.
  manager->dl_tensor.dtype.code = static_cast<uint8_t>(type_code);
  manager->dl_tensor.dtype.bits = bits;
  manager->dl_tensor.dtype.lanes = 1;
  manager->dl_tensor.shape = shape;
  manager->dl_tensor.strides = strides;
  manager->dl_tensor.byte_offset = 0;
  manager->manager_ctx = nullptr;
  manager->deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete self;
  };

  // Return a pybind11 capsule with the DLPack protocol name.
  // pybind11 will call the capsule destructor when the Python object
  // is garbage-collected, but the *real* cleanup happens via the
  // DLManagedTensor deleter after JAX (or another consumer) calls it.
  py::capsule cap(manager, "dltensor", [](PyObject* /*capsule*/) {
    // No-op: JAX is responsible for calling the DLManagedTensor deleter.
  });
  return cap;
}

// ---------------------------------------------------------------------------
// Convenience overloads for the common GPUMD types.
// ---------------------------------------------------------------------------
py::capsule dlpack_from_double_vector(double* data, int64_t n, int64_t comp = 1)
{
  return gpu_vector_to_dlpack(data, n, comp, kDLFloat, 64);
}

py::capsule dlpack_from_int_vector(int* data, int64_t n, int64_t comp = 1)
{
  return gpu_vector_to_dlpack(data, n, comp, kDLInt, 32);
}

// ---------------------------------------------------------------------------
// PySimulation: high-level wrapper exposed to Python.
// ---------------------------------------------------------------------------
class PySimulation
{
  // Heap-allocate Run so there is exactly one owner.  Copying Run
  // (which contains GPU_Vector with raw CUDA pointers) would cause a
  // double-free when the copy's destructor also calls gpuFree.
  std::unique_ptr<Run> run_;
  std::string run_input_path_;

public:
  explicit PySimulation(const std::string& run_input_path = "run.in")
    : run_input_path_(run_input_path)
  {
    // Construct GPUMD in "skip-run" mode so that run.in is parsed but
    // perform_a_run() is not executed immediately.
    run_ = std::make_unique<Run>(true, run_input_path);

    // Synchronize to ensure allocations are done before Python touches them.
    cudaDeviceSynchronize();
  }

  // -----------------------------------------------------------------------
  // Query whether the simulation box is guaranteed to remain constant.
  //
  // This reads the run.in file and checks for keywords that modify the box
  // (npt ensemble, change_box, puff).  If none are found, the box is
  // treated as constant for the entire run, allowing the PySAGES backend
  // to skip per-step box refreshes.
  // -----------------------------------------------------------------------
  bool is_box_constant() const
  {
    std::ifstream file(run_input_path_);
    if (!file.is_open()) {
      // If we cannot read the file, be conservative and assume the box
      // may change (NPT / etc.).
      return false;
    }
    std::string line;
    while (std::getline(file, line)) {
      // Skip comments and blank lines.
      auto first_non_space = line.find_first_not_of(" \t\r\n");
      if (first_non_space == std::string::npos) continue;
      if (line[first_non_space] == '#') continue;

      // Check for box-changing keywords (case-insensitive).
      std::string lower;
      for (char c : line) lower += std::tolower(c);

      if (lower.find("ensemble npt") != std::string::npos) return false;
      if (lower.find("change_box") != std::string::npos) return false;
      if (lower.find("puff") != std::string::npos) return false;
    }
    return true;
  }

  // -----------------------------------------------------------------------
  // DLPack accessors
  // -----------------------------------------------------------------------
  py::capsule get_positions_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().position_per_atom.data(),
      run_->get_atom().number_of_atoms,
      3);
  }

  py::capsule get_velocities_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().velocity_per_atom.data(),
      run_->get_atom().number_of_atoms,
      3);
  }

  py::capsule get_forces_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().force_per_atom.data(),
      run_->get_atom().number_of_atoms,
      3);
  }

  py::capsule get_masses_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().mass.data(),
      run_->get_atom().number_of_atoms,
      1);
  }

  py::capsule get_types_dlpack()
  {
    return dlpack_from_int_vector(
      run_->get_atom().type.data(),
      run_->get_atom().number_of_atoms,
      1);
  }

  // -----------------------------------------------------------------------
  // Box & timestep
  // -----------------------------------------------------------------------
  py::tuple get_box()
  {
    // GPUMD stores the 3x3 affine transform in cpu_h[0..8] (row-major).
    py::list h(9);
    for (int i = 0; i < 9; ++i) {
      h[i] = run_->get_box().cpu_h[i];
    }
    // GPUMD does not store an explicit origin; it is always (0,0,0).
    py::tuple origin = py::make_tuple(0.0, 0.0, 0.0);
    return py::make_tuple(h, origin);
  }

  double get_timestep() const
  {
    // GPUMD internal natural time units.
    return run_->get_time_step();
  }

  int get_number_of_atoms() const
  {
    return run_->get_atom().number_of_atoms;
  }

  // -----------------------------------------------------------------------
  // Callback registration
  // -----------------------------------------------------------------------
  void set_step_callback(std::function<void(int)> cb)
  {
    run_->step_callback = cb;
  }

  // -----------------------------------------------------------------------
  // Execution
  // -----------------------------------------------------------------------
  void run(int steps)
  {
    run_->set_number_of_steps(steps);
    run_->execute_run();
  }

  void synchronize()
  {
    cudaDeviceSynchronize();
  }

  // -----------------------------------------------------------------------
  // Direct bias write (optional)
  //
  // Accepts a DLPack capsule from Python (e.g. a CuPy array) and copies
  // its contents into GPUMD's external_bias_per_atom buffer.
  // -----------------------------------------------------------------------
  void set_external_bias(py::capsule dlpack_cap)
  {
    DLManagedTensor* dlm = static_cast<DLManagedTensor*>(dlpack_cap.get_pointer());
    if (!dlm) {
      throw std::runtime_error("Invalid DLPack capsule");
    }

    const DLTensor& dt = dlm->dl_tensor;
    if (dt.device.device_type != kDLCUDA && dt.device.device_type != kDLCUDAManaged) {
      throw std::runtime_error("set_external_bias expects a CUDA DLPack tensor");
    }

    int64_t expected = run_->get_atom().number_of_atoms * 3;
    int64_t actual = 1;
    for (int i = 0; i < dt.ndim; ++i) {
      actual *= dt.shape[i];
    }
    if (actual != expected) {
      throw std::runtime_error("Bias tensor size mismatch");
    }

    // Validate dtype: GPUMD forces are double, so we need float64 bias
    if (dt.dtype.code != static_cast<uint8_t>(kDLFloat) || dt.dtype.bits != 64) {
      char msg[128];
      snprintf(
        msg,
        sizeof(msg),
        "set_external_bias: expected float64 (kDLFloat,64) bias, got code=%u bits=%u",
        dt.dtype.code,
        dt.dtype.bits);
      throw std::runtime_error(msg);
    }

    cudaMemcpy(
      run_->external_bias_per_atom.data(),
      dt.data,
      expected * sizeof(double),
      cudaMemcpyDeviceToDevice);

    // NOTE: We do NOT call dlm->deleter here. JAX's to_dlpack() capsule
    // already has a capsule destructor that calls the deleter when the
    // Python capsule object is garbage-collected. Calling it manually would
    // cause a double-free / use-after-free and segfault.
  }

  // -----------------------------------------------------------------------
  // Zero the external bias buffer.
  // -----------------------------------------------------------------------
  void clear_external_bias()
  {
    int64_t n = run_->external_bias_per_atom.size();
    if (n > 0) {
      cudaMemset(run_->external_bias_per_atom.data(), 0, n * sizeof(double));
    }
  }

  // -----------------------------------------------------------------------
  // Direct in-place bias add via custom CUDA kernel.
  // -----------------------------------------------------------------------
  void add_aos_bias_to_forces(py::capsule dlpack_cap)
  {
    DLManagedTensor* dlm = static_cast<DLManagedTensor*>(dlpack_cap.get_pointer());
    if (!dlm) {
      throw std::runtime_error("Invalid DLPack capsule");
    }

    const DLTensor& dt = dlm->dl_tensor;
    if (dt.device.device_type != kDLCUDA && dt.device.device_type != kDLCUDAManaged) {
      throw std::runtime_error("add_aos_bias_to_forces expects a CUDA DLPack tensor");
    }

    int64_t expected = run_->get_atom().number_of_atoms * 3;
    int64_t actual = 1;
    for (int i = 0; i < dt.ndim; ++i) {
      actual *= dt.shape[i];
    }
    if (actual != expected) {
      throw std::runtime_error("Bias tensor size mismatch");
    }

    if (dt.dtype.code != static_cast<uint8_t>(kDLFloat) || dt.dtype.bits != 64) {
      char msg[128];
      snprintf(
        msg,
        sizeof(msg),
        "add_aos_bias_to_forces: expected float64 bias, got code=%u bits=%u",
        dt.dtype.code,
        dt.dtype.bits);
      throw std::runtime_error(msg);
    }

    const int N = run_->get_atom().number_of_atoms;
    const double* bias_aos = static_cast<const double*>(dt.data);
    double* forces_soa = run_->get_atom().force_per_atom.data();

    gpu_add_aos_bias_to_soa_forces(N, forces_soa, bias_aos);

    // Synchronize so GPUMD's subsequent integrate step sees the updated forces.
    cudaDeviceSynchronize();
  }
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(gpumd, m)
{
  m.doc() = "GPUMD Python wrapper for PySAGES integration";

  py::class_<PySimulation>(m, "Simulation")
    .def(py::init<const std::string&>(), py::arg("run_input_path") = "run.in")
    .def("get_positions_dlpack", &PySimulation::get_positions_dlpack)
    .def("get_velocities_dlpack", &PySimulation::get_velocities_dlpack)
    .def("get_forces_dlpack", &PySimulation::get_forces_dlpack)
    .def("get_masses_dlpack", &PySimulation::get_masses_dlpack)
    .def("get_types_dlpack", &PySimulation::get_types_dlpack)
    .def("get_box", &PySimulation::get_box)
    .def("get_timestep", &PySimulation::get_timestep)
    .def("get_number_of_atoms", &PySimulation::get_number_of_atoms)
    .def("set_step_callback", &PySimulation::set_step_callback)
    .def("run", &PySimulation::run)
    .def("synchronize", &PySimulation::synchronize)
    .def("set_external_bias", &PySimulation::set_external_bias)
    .def("clear_external_bias", &PySimulation::clear_external_bias)
    .def("add_aos_bias_to_forces", &PySimulation::add_aos_bias_to_forces)
    .def("is_box_constant", &PySimulation::is_box_constant,
         "Returns True if the simulation box is guaranteed to remain constant (NVT/NVE).");
}
