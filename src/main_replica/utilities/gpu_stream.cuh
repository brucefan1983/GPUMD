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

#pragma once

#include "error.cuh"

namespace gpu_detail
{
class Device_Guard
{
public:
  explicit Device_Guard(const int device_id) : restore_(false)
  {
    CHECK(gpuGetDevice(&previous_device_));
    if (previous_device_ != device_id) {
      CHECK(gpuSetDevice(device_id));
      restore_ = true;
    }
  }

  ~Device_Guard()
  {
    if (restore_)
      CHECK(gpuSetDevice(previous_device_));
  }

  Device_Guard(const Device_Guard&) = delete;
  Device_Guard& operator=(const Device_Guard&) = delete;

private:
  int previous_device_;
  bool restore_;
};
} // namespace gpu_detail

class GPU_Stream;

// Move-only owners for streams and events created on a specific device.
class GPU_Event
{
public:
  GPU_Event() = default;

  explicit GPU_Event(const int device_id, const bool enable_timing = false)
  {
    create(device_id, enable_timing);
  }

  ~GPU_Event() { destroy(); }

  GPU_Event(const GPU_Event&) = delete;
  GPU_Event& operator=(const GPU_Event&) = delete;

  GPU_Event(GPU_Event&& other) noexcept
    : event_(other.event_), device_id_(other.device_id_)
  {
    other.event_ = nullptr;
    other.device_id_ = -1;
  }

  GPU_Event& operator=(GPU_Event&& other) noexcept
  {
    if (this != &other) {
      destroy();
      event_ = other.event_;
      device_id_ = other.device_id_;
      other.event_ = nullptr;
      other.device_id_ = -1;
    }
    return *this;
  }

  void create(const int device_id, const bool enable_timing = false)
  {
    destroy();
    gpu_detail::Device_Guard guard(device_id);
    const unsigned int flags = enable_timing ? gpuEventDefault : gpuEventDisableTiming;
    CHECK(gpuEventCreateWithFlags(&event_, flags));
    device_id_ = device_id;
  }

  void destroy()
  {
    if (event_ != nullptr) {
      gpu_detail::Device_Guard guard(device_id_);
      CHECK(gpuEventDestroy(event_));
      event_ = nullptr;
      device_id_ = -1;
    }
  }

  void record(const gpuStream_t stream = nullptr)
  {
    gpu_detail::Device_Guard guard(device_id_);
    CHECK(gpuEventRecord(event_, stream));
  }

  void record(const GPU_Stream& stream);

  void synchronize() const
  {
    gpu_detail::Device_Guard guard(device_id_);
    CHECK(gpuEventSynchronize(event_));
  }

  bool ready() const
  {
    gpu_detail::Device_Guard guard(device_id_);
    const gpuError_t status = gpuEventQuery(event_);
    if (status == gpuSuccess)
      return true;
    if (status == gpuErrorNotReady)
      return false;
    CHECK(status);
    return false;
  }

  float elapsed_time_since(const GPU_Event& start) const
  {
    gpu_detail::Device_Guard guard(device_id_);
    float milliseconds = 0.0f;
    CHECK(gpuEventElapsedTime(&milliseconds, start.event_, event_));
    return milliseconds;
  }

  bool valid() const { return event_ != nullptr; }
  int device_id() const { return device_id_; }
  gpuEvent_t get() const { return event_; }

private:
  gpuEvent_t event_ = nullptr;
  int device_id_ = -1;
};

class GPU_Stream
{
public:
  GPU_Stream() = default;

  explicit GPU_Stream(
    const int device_id, const unsigned int flags = gpuStreamNonBlocking)
  {
    create(device_id, flags);
  }

  ~GPU_Stream() { destroy(); }

  GPU_Stream(const GPU_Stream&) = delete;
  GPU_Stream& operator=(const GPU_Stream&) = delete;

  GPU_Stream(GPU_Stream&& other) noexcept
    : stream_(other.stream_), device_id_(other.device_id_)
  {
    other.stream_ = nullptr;
    other.device_id_ = -1;
  }

  GPU_Stream& operator=(GPU_Stream&& other) noexcept
  {
    if (this != &other) {
      destroy();
      stream_ = other.stream_;
      device_id_ = other.device_id_;
      other.stream_ = nullptr;
      other.device_id_ = -1;
    }
    return *this;
  }

  void create(const int device_id, const unsigned int flags = gpuStreamNonBlocking)
  {
    destroy();
    gpu_detail::Device_Guard guard(device_id);
    CHECK(gpuStreamCreateWithFlags(&stream_, flags));
    device_id_ = device_id;
  }

  void destroy()
  {
    if (stream_ != nullptr) {
      gpu_detail::Device_Guard guard(device_id_);
      CHECK(gpuStreamDestroy(stream_));
      stream_ = nullptr;
      device_id_ = -1;
    }
  }

  void synchronize() const
  {
    gpu_detail::Device_Guard guard(device_id_);
    CHECK(gpuStreamSynchronize(stream_));
  }

  void wait(const GPU_Event& event) const
  {
    gpu_detail::Device_Guard guard(device_id_);
    CHECK(gpuStreamWaitEvent(stream_, event.get(), 0));
  }

  bool ready() const
  {
    gpu_detail::Device_Guard guard(device_id_);
    const gpuError_t status = gpuStreamQuery(stream_);
    if (status == gpuSuccess)
      return true;
    if (status == gpuErrorNotReady)
      return false;
    CHECK(status);
    return false;
  }

  bool valid() const { return stream_ != nullptr; }
  int device_id() const { return device_id_; }
  gpuStream_t get() const { return stream_; }

private:
  gpuStream_t stream_ = nullptr;
  int device_id_ = -1;
};

inline void GPU_Event::record(const GPU_Stream& stream)
{
  record(stream.get());
}
