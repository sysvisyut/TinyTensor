#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace tinytensor::runtime {

struct LaunchDimensions {
  unsigned int x{1};
  unsigned int y{1};
  unsigned int z{1};

  LaunchDimensions() = default;
  constexpr LaunchDimensions(unsigned int x_, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

class CUDAException : public std::runtime_error {
 public:
  CUDAException(std::string_view call, int errorCode,
                std::string_view errorName, std::string_view errorDesc,
                std::string_view filename, int line,
                std::string_view jitLog = "");

  int errorCode() const noexcept { return errorCode_; }
  const std::string& errorName() const noexcept { return errorName_; }
  const std::string& errorDesc() const noexcept { return errorDesc_; }
  const std::string& jitLog() const noexcept { return jitLog_; }

 private:
  int errorCode_;
  std::string errorName_;
  std::string errorDesc_;
  std::string jitLog_;
};

class CUDADriver {
 public:
  static bool isAvailable() noexcept;
  static std::size_t deviceCount() noexcept;
  static std::string getDeviceName(std::size_t deviceIndex);
  static std::pair<int, int> getComputeCapability(std::size_t deviceIndex);
  static std::string getDeviceArch(std::size_t deviceIndex); // e.g. "sm_75"
};

// Internal shared states forward-declared to prevent leaking CUDA types into public headers
struct CUDAContextState;
struct CUDAModuleState;

class CUDAModule;

class CUDAFunction {
 public:
  ~CUDAFunction() = default;
  CUDAFunction(CUDAFunction&&) noexcept = default;
  CUDAFunction& operator=(CUDAFunction&&) noexcept = default;
  CUDAFunction(const CUDAFunction&) = default;
  CUDAFunction& operator=(const CUDAFunction&) = default;

  void launch(LaunchDimensions gridDim, LaunchDimensions blockDim,
              void** kernelParams, unsigned int sharedMemBytes = 0);
  void* rawFunction() const noexcept { return rawFunction_; }

 private:
  friend class CUDAModule;
  CUDAFunction(std::shared_ptr<CUDAModuleState> moduleState, void* rawFunction);

  std::shared_ptr<CUDAModuleState> moduleState_; // Prevents module unload while function exists
  void* rawFunction_{nullptr};
};

class CUDAContext {
 public:
  explicit CUDAContext(std::size_t deviceIndex = 0);
  ~CUDAContext() = default;
  CUDAContext(CUDAContext&&) noexcept = default;
  CUDAContext& operator=(CUDAContext&&) noexcept = default;
  CUDAContext(const CUDAContext&) = delete;
  CUDAContext& operator=(const CUDAContext&) = delete;

  void makeCurrent();
  void synchronize();
  std::shared_ptr<CUDAContextState> state() const noexcept { return state_; }

 private:
  std::shared_ptr<CUDAContextState> state_;
};

class CUDAModule {
 public:
  CUDAModule(std::shared_ptr<CUDAContextState> ctx, std::string_view ptx);
  ~CUDAModule() = default;
  CUDAModule(CUDAModule&&) noexcept = default;
  CUDAModule& operator=(CUDAModule&&) noexcept = default;
  CUDAModule(const CUDAModule&) = delete;
  CUDAModule& operator=(const CUDAModule&) = delete;

  CUDAFunction getFunction(std::string_view kernelName);

 private:
  std::shared_ptr<CUDAModuleState> state_;
};

class CUDABuffer {
 public:
  CUDABuffer(std::shared_ptr<CUDAContextState> ctx, std::size_t sizeBytes);
  ~CUDABuffer();
  CUDABuffer(CUDABuffer&&) noexcept;
  CUDABuffer& operator=(CUDABuffer&&) noexcept;
  CUDABuffer(const CUDABuffer&) = delete;
  CUDABuffer& operator=(const CUDABuffer&) = delete;

  void copyFromHost(const void* hostPtr, std::size_t bytes);
  void copyToHost(void* hostPtr, std::size_t bytes);

  std::size_t sizeBytes() const noexcept { return sizeBytes_; }
  std::uintptr_t deviceAddress() const noexcept { return deviceAddress_; }

 private:
  std::shared_ptr<CUDAContextState> ctxState_; // Prevents context destroy while buffer exists
  std::uintptr_t deviceAddress_{0};
  std::size_t sizeBytes_{0};
};

} // namespace tinytensor::runtime
