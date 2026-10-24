#include "tinytensor/runtime/cuda_driver.hpp"

#include <iostream>
#include <mutex>
#include <vector>

namespace tinytensor::runtime {

CUDAException::CUDAException(std::string_view call, int errorCode,
                             std::string_view errorName, std::string_view errorDesc,
                             std::string_view filename, int line,
                             std::string_view jitLog)
    : std::runtime_error([&]() {
        std::string msg = "[CUDA Driver Error] in ";
        msg += filename;
        msg += ":";
        msg += std::to_string(line);
        msg += " during ";
        msg += call;
        msg += ": ";
        msg += errorName;
        msg += " (code ";
        msg += std::to_string(errorCode);
        msg += ") - ";
        msg += errorDesc;
        if (!jitLog.empty()) {
          msg += "\nJIT Error Log:\n";
          msg += jitLog;
        }
        return msg;
      }()),
      errorCode_(errorCode),
      errorName_(errorName),
      errorDesc_(errorDesc),
      jitLog_(jitLog) {}

#if defined(TINYTENSOR_HAS_CUDA)

#include <cuda.h>

namespace {

std::string getCUDAResultName(CUresult res) {
  const char* name = nullptr;
  if (cuGetErrorName(res, &name) == CUDA_SUCCESS && name) {
    return std::string(name);
  }
  return "CUDA_ERROR_UNKNOWN";
}

std::string getCUDAResultString(CUresult res) {
  const char* str = nullptr;
  if (cuGetErrorString(res, &str) == CUDA_SUCCESS && str) {
    return std::string(str);
  }
  return "Unknown CUDA driver error";
}

#define TINYTENSOR_CHECK_CUDA_DRIVER(call)                                       \
  do {                                                                           \
    CUresult _res = (call);                                                      \
    if (_res != CUDA_SUCCESS) {                                                  \
      throw ::tinytensor::runtime::CUDAException(                                \
          #call, static_cast<int>(_res), getCUDAResultName(_res),               \
          getCUDAResultString(_res), __FILE__, __LINE__);                        \
    }                                                                            \
  } while (0)

} // namespace

struct CUDAContextState {
  explicit CUDAContextState(CUcontext ctx) : ctx_(ctx) {}
  ~CUDAContextState() {
    if (ctx_) {
      cuCtxDestroy(ctx_);
      ctx_ = nullptr;
    }
  }

  CUDAContextState(const CUDAContextState&) = delete;
  CUDAContextState& operator=(const CUDAContextState&) = delete;

  CUcontext handle() const noexcept { return ctx_; }

 private:
  CUcontext ctx_{nullptr};
};

namespace {

class CUDAContextScope {
 public:
  explicit CUDAContextScope(const std::shared_ptr<CUDAContextState>& state) {
    if (state && state->handle()) {
      cuCtxGetCurrent(&prevCtx_);
      if (prevCtx_ != state->handle()) {
        cuCtxSetCurrent(state->handle());
        swapped_ = true;
      }
    }
  }

  ~CUDAContextScope() {
    if (swapped_) {
      cuCtxSetCurrent(prevCtx_);
    }
  }

  CUDAContextScope(const CUDAContextScope&) = delete;
  CUDAContextScope& operator=(const CUDAContextScope&) = delete;

 private:
  CUcontext prevCtx_{nullptr};
  bool swapped_{false};
};

} // namespace

struct CUDAModuleState {
  CUDAModuleState(std::shared_ptr<CUDAContextState> ctxState, CUmodule module)
      : ctxState_(std::move(ctxState)), module_(module) {}

  ~CUDAModuleState() {
    if (module_ && ctxState_) {
      CUDAContextScope scope(ctxState_);
      cuModuleUnload(module_);
      module_ = nullptr;
    }
  }

  CUDAModuleState(const CUDAModuleState&) = delete;
  CUDAModuleState& operator=(const CUDAModuleState&) = delete;

  CUmodule handle() const noexcept { return module_; }
  const std::shared_ptr<CUDAContextState>& contextState() const noexcept {
    return ctxState_;
  }

 private:
  std::shared_ptr<CUDAContextState> ctxState_;
  CUmodule module_{nullptr};
};

bool CUDADriver::isAvailable() noexcept {
  static std::once_flag initFlag;
  static bool available = false;
  std::call_once(initFlag, []() {
    if (cuInit(0) != CUDA_SUCCESS) {
      available = false;
      return;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
      available = false;
      return;
    }
    available = true;
  });
  return available;
}

std::size_t CUDADriver::deviceCount() noexcept {
  if (!isAvailable()) {
    return 0;
  }
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count < 0) {
    return 0;
  }
  return static_cast<std::size_t>(count);
}

std::string CUDADriver::getDeviceName(std::size_t deviceIndex) {
  if (!isAvailable() || deviceIndex >= deviceCount()) {
    throw CUDAException("CUDADriver::getDeviceName", -1,
                        "CUDA_ERROR_INVALID_DEVICE",
                        "Device index out of range or CUDA unavailable",
                        __FILE__, __LINE__);
  }
  CUdevice dev;
  TINYTENSOR_CHECK_CUDA_DRIVER(cuDeviceGet(&dev, static_cast<int>(deviceIndex)));
  char nameBuf[256];
  TINYTENSOR_CHECK_CUDA_DRIVER(cuDeviceGetName(nameBuf, sizeof(nameBuf), dev));
  return std::string(nameBuf);
}

std::pair<int, int> CUDADriver::getComputeCapability(std::size_t deviceIndex) {
  if (!isAvailable() || deviceIndex >= deviceCount()) {
    throw CUDAException("CUDADriver::getComputeCapability", -1,
                        "CUDA_ERROR_INVALID_DEVICE",
                        "Device index out of range or CUDA unavailable",
                        __FILE__, __LINE__);
  }
  CUdevice dev;
  TINYTENSOR_CHECK_CUDA_DRIVER(cuDeviceGet(&dev, static_cast<int>(deviceIndex)));
  int major = 0;
  int minor = 0;
  TINYTENSOR_CHECK_CUDA_DRIVER(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  TINYTENSOR_CHECK_CUDA_DRIVER(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  return {major, minor};
}

std::string CUDADriver::getDeviceArch(std::size_t deviceIndex) {
  auto [major, minor] = getComputeCapability(deviceIndex);
  return "sm_" + std::to_string(major) + std::to_string(minor);
}

CUDAContext::CUDAContext(std::size_t deviceIndex) {
  if (!CUDADriver::isAvailable() || deviceIndex >= CUDADriver::deviceCount()) {
    throw CUDAException("CUDAContext::CUDAContext", -1,
                        "CUDA_ERROR_NO_DEVICE",
                        "No CUDA device available or index out of range",
                        __FILE__, __LINE__);
  }
  CUdevice dev;
  TINYTENSOR_CHECK_CUDA_DRIVER(cuDeviceGet(&dev, static_cast<int>(deviceIndex)));
  CUcontext rawCtx = nullptr;
  TINYTENSOR_CHECK_CUDA_DRIVER(cuCtxCreate(&rawCtx, 0, dev));
  state_ = std::make_shared<CUDAContextState>(rawCtx);
}

void CUDAContext::makeCurrent() {
  if (!state_) {
    throw CUDAException("CUDAContext::makeCurrent", -1,
                        "CUDA_ERROR_INVALID_CONTEXT",
                        "Context state is null", __FILE__, __LINE__);
  }
  TINYTENSOR_CHECK_CUDA_DRIVER(cuCtxSetCurrent(state_->handle()));
}

void CUDAContext::synchronize() {
  if (!state_) {
    throw CUDAException("CUDAContext::synchronize", -1,
                        "CUDA_ERROR_INVALID_CONTEXT",
                        "Context state is null", __FILE__, __LINE__);
  }
  CUDAContextScope scope(state_);
  TINYTENSOR_CHECK_CUDA_DRIVER(cuCtxSynchronize());
}

CUDAModule::CUDAModule(std::shared_ptr<CUDAContextState> ctx, std::string_view ptx) {
  if (!ctx || !ctx->handle()) {
    throw CUDAException("CUDAModule::CUDAModule", -1,
                        "CUDA_ERROR_INVALID_CONTEXT",
                        "Null context supplied for module loading",
                        __FILE__, __LINE__);
  }

  CUDAContextScope scope(ctx);

  constexpr unsigned int logBufferSize = 8192;
  std::vector<char> infoLog(logBufferSize, 0);
  std::vector<char> errorLog(logBufferSize, 0);

  CUjit_option options[] = {
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_LOG_VERBOSE,
  };
  void* optionValues[] = {
      infoLog.data(),
      reinterpret_cast<void*>(static_cast<uintptr_t>(logBufferSize)),
      errorLog.data(),
      reinterpret_cast<void*>(static_cast<uintptr_t>(logBufferSize)),
      reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
  };

  CUmodule rawModule = nullptr;
  CUresult res = cuModuleLoadDataEx(
      &rawModule, ptx.data(), sizeof(options) / sizeof(options[0]), options, optionValues);

  if (res != CUDA_SUCCESS) {
    std::string jitErr(errorLog.data());
    throw CUDAException("cuModuleLoadDataEx", static_cast<int>(res),
                        getCUDAResultName(res), getCUDAResultString(res),
                        __FILE__, __LINE__, jitErr);
  }

  state_ = std::make_shared<CUDAModuleState>(std::move(ctx), rawModule);
}

CUDAFunction CUDAModule::getFunction(std::string_view kernelName) {
  if (!state_ || !state_->handle()) {
    throw CUDAException("CUDAModule::getFunction", -1,
                        "CUDA_ERROR_INVALID_IMAGE",
                        "Module state is invalid or unloaded",
                        __FILE__, __LINE__);
  }

  CUDAContextScope scope(state_->contextState());
  CUfunction rawFn = nullptr;
  std::string nameStr(kernelName);
  CUresult res = cuModuleGetFunction(&rawFn, state_->handle(), nameStr.c_str());
  if (res != CUDA_SUCCESS) {
    throw CUDAException("cuModuleGetFunction", static_cast<int>(res),
                        getCUDAResultName(res), getCUDAResultString(res),
                        __FILE__, __LINE__);
  }

  return CUDAFunction(state_, static_cast<void*>(rawFn));
}

CUDAFunction::CUDAFunction(std::shared_ptr<CUDAModuleState> moduleState, void* rawFunction)
    : moduleState_(std::move(moduleState)), rawFunction_(rawFunction) {}

void CUDAFunction::launch(LaunchDimensions gridDim, LaunchDimensions blockDim,
                          void** kernelParams, unsigned int sharedMemBytes) {
  if (!moduleState_ || !rawFunction_) {
    throw CUDAException("CUDAFunction::launch", -1,
                        "CUDA_ERROR_INVALID_HANDLE",
                        "CUDAFunction is not associated with a valid module",
                        __FILE__, __LINE__);
  }

  CUDAContextScope scope(moduleState_->contextState());
  TINYTENSOR_CHECK_CUDA_DRIVER(cuLaunchKernel(
      static_cast<CUfunction>(rawFunction_), gridDim.x, gridDim.y, gridDim.z,
      blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, nullptr,
      kernelParams, nullptr));
}

CUDABuffer::CUDABuffer(std::shared_ptr<CUDAContextState> ctx, std::size_t sizeBytes)
    : ctxState_(std::move(ctx)), sizeBytes_(sizeBytes) {
  if (!ctxState_ || !ctxState_->handle()) {
    throw CUDAException("CUDABuffer::CUDABuffer", -1,
                        "CUDA_ERROR_INVALID_CONTEXT",
                        "Null context supplied for buffer allocation",
                        __FILE__, __LINE__);
  }

  CUDAContextScope scope(ctxState_);
  CUdeviceptr dptr = 0;
  TINYTENSOR_CHECK_CUDA_DRIVER(cuMemAlloc(&dptr, sizeBytes));
  deviceAddress_ = static_cast<std::uintptr_t>(dptr);
}

CUDABuffer::~CUDABuffer() {
  if (deviceAddress_ && ctxState_) {
    CUDAContextScope scope(ctxState_);
    cuMemFree(static_cast<CUdeviceptr>(deviceAddress_));
    deviceAddress_ = 0;
    sizeBytes_ = 0;
  }
}

CUDABuffer::CUDABuffer(CUDABuffer&& other) noexcept
    : ctxState_(std::move(other.ctxState_)),
      deviceAddress_(other.deviceAddress_),
      sizeBytes_(other.sizeBytes_) {
  other.deviceAddress_ = 0;
  other.sizeBytes_ = 0;
}

CUDABuffer& CUDABuffer::operator=(CUDABuffer&& other) noexcept {
  if (this != &other) {
    if (deviceAddress_ && ctxState_) {
      CUDAContextScope scope(ctxState_);
      cuMemFree(static_cast<CUdeviceptr>(deviceAddress_));
    }
    ctxState_ = std::move(other.ctxState_);
    deviceAddress_ = other.deviceAddress_;
    sizeBytes_ = other.sizeBytes_;
    other.deviceAddress_ = 0;
    other.sizeBytes_ = 0;
  }
  return *this;
}

void CUDABuffer::copyFromHost(const void* hostPtr, std::size_t bytes) {
  if (!deviceAddress_ || !ctxState_) {
    throw CUDAException("CUDABuffer::copyFromHost", -1,
                        "CUDA_ERROR_INVALID_VALUE",
                        "Buffer is unallocated or moved-from",
                        __FILE__, __LINE__);
  }
  if (bytes > sizeBytes_) {
    throw CUDAException("CUDABuffer::copyFromHost", -1,
                        "CUDA_ERROR_INVALID_VALUE",
                        "Copy byte count exceeds allocated buffer size",
                        __FILE__, __LINE__);
  }
  CUDAContextScope scope(ctxState_);
  TINYTENSOR_CHECK_CUDA_DRIVER(
      cuMemcpyHtoD(static_cast<CUdeviceptr>(deviceAddress_), hostPtr, bytes));
}

void CUDABuffer::copyToHost(void* hostPtr, std::size_t bytes) {
  if (!deviceAddress_ || !ctxState_) {
    throw CUDAException("CUDABuffer::copyToHost", -1,
                        "CUDA_ERROR_INVALID_VALUE",
                        "Buffer is unallocated or moved-from",
                        __FILE__, __LINE__);
  }
  if (bytes > sizeBytes_) {
    throw CUDAException("CUDABuffer::copyToHost", -1,
                        "CUDA_ERROR_INVALID_VALUE",
                        "Copy byte count exceeds allocated buffer size",
                        __FILE__, __LINE__);
  }
  CUDAContextScope scope(ctxState_);
  TINYTENSOR_CHECK_CUDA_DRIVER(
      cuMemcpyDtoH(hostPtr, static_cast<CUdeviceptr>(deviceAddress_), bytes));
}

#else // !defined(TINYTENSOR_HAS_CUDA)

struct CUDAContextState {};
struct CUDAModuleState {};

bool CUDADriver::isAvailable() noexcept { return false; }
std::size_t CUDADriver::deviceCount() noexcept { return 0; }

std::string CUDADriver::getDeviceName(std::size_t) {
  throw CUDAException("CUDADriver::getDeviceName", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

std::pair<int, int> CUDADriver::getComputeCapability(std::size_t) {
  throw CUDAException("CUDADriver::getComputeCapability", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

std::string CUDADriver::getDeviceArch(std::size_t) {
  throw CUDAException("CUDADriver::getDeviceArch", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

CUDAContext::CUDAContext(std::size_t) {
  throw CUDAException("CUDAContext::CUDAContext", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

void CUDAContext::makeCurrent() {}
void CUDAContext::synchronize() {}

CUDAModule::CUDAModule(std::shared_ptr<CUDAContextState>, std::string_view) {
  throw CUDAException("CUDAModule::CUDAModule", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

CUDAFunction CUDAModule::getFunction(std::string_view) {
  throw CUDAException("CUDAModule::getFunction", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

CUDAFunction::CUDAFunction(std::shared_ptr<CUDAModuleState> moduleState, void* rawFunction)
    : moduleState_(std::move(moduleState)), rawFunction_(rawFunction) {}

void CUDAFunction::launch(LaunchDimensions, LaunchDimensions, void**, unsigned int) {
  throw CUDAException("CUDAFunction::launch", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

CUDABuffer::CUDABuffer(std::shared_ptr<CUDAContextState>, std::size_t) {
  throw CUDAException("CUDABuffer::CUDABuffer", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

CUDABuffer::~CUDABuffer() = default;

CUDABuffer::CUDABuffer(CUDABuffer&& other) noexcept
    : ctxState_(std::move(other.ctxState_)),
      deviceAddress_(other.deviceAddress_),
      sizeBytes_(other.sizeBytes_) {
  other.deviceAddress_ = 0;
  other.sizeBytes_ = 0;
}

CUDABuffer& CUDABuffer::operator=(CUDABuffer&& other) noexcept {
  if (this != &other) {
    ctxState_ = std::move(other.ctxState_);
    deviceAddress_ = other.deviceAddress_;
    sizeBytes_ = other.sizeBytes_;
    other.deviceAddress_ = 0;
    other.sizeBytes_ = 0;
  }
  return *this;
}

void CUDABuffer::copyFromHost(const void*, std::size_t) {
  throw CUDAException("CUDABuffer::copyFromHost", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

void CUDABuffer::copyToHost(void*, std::size_t) {
  throw CUDAException("CUDABuffer::copyToHost", -1,
                      "CUDA_NOT_COMPILED",
                      "CUDA support not compiled into this build",
                      __FILE__, __LINE__);
}

#endif // TINYTENSOR_HAS_CUDA

} // namespace tinytensor::runtime
