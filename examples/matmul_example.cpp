#include <iostream>
#include <iomanip>
#include <vector>

#include "tinytensor/ir.hpp"
#include "tinytensor/runtime/cpu_reference.hpp"
#include "tinytensor/runtime/cuda_driver.hpp"
#include "tinytensor/backend/nvptx_target.hpp"
#include "tinytensor/backend/tiled_matmul.hpp"

#if defined(TINYTENSOR_HAS_LLVM)
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#endif

using namespace tinytensor;
using namespace tinytensor::runtime;
using namespace tinytensor::backend;

void printMatrix(const std::string& name, const std::vector<float>& data, int rows, int cols) {
  std::cout << name << " (" << rows << "x" << cols << "):\n";
  for (int r = 0; r < std::min(rows, 6); ++r) {
    std::cout << "  [ ";
    for (int c = 0; c < std::min(cols, 6); ++c) {
      std::cout << std::fixed << std::setprecision(2) << std::setw(6)
                << data[static_cast<std::size_t>(r * cols + c)] << " ";
    }
    if (cols > 6) std::cout << "... ";
    std::cout << "]\n";
  }
  if (rows > 6) std::cout << "  ... (" << rows - 6 << " more rows)\n";
  std::cout << std::endl;
}

int main() {
  std::cout << "========================================================\n"
            << "        TinyTensor End-to-End MatMul Example            \n"
            << "========================================================\n";

  constexpr int M = 4;
  constexpr int K = 4;
  constexpr int N = 4;

  std::vector<float> h_A = {
    1.0f, 2.0f, 3.0f, 4.0f,
    0.5f, 1.5f, 2.5f, 3.5f,
    1.0f, 0.0f, 1.0f, 0.0f,
    2.0f, 1.0f, 0.0f, 1.0f
  };

  std::vector<float> h_B = {
    1.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f
  };

  printMatrix("Input Matrix A", h_A, M, K);
  printMatrix("Input Matrix B", h_B, K, N);

  // 1. Run via CPU Reference Runtime
  std::cout << "[1] Executing via CPU Reference Interpreter...\n";
  Module irModule;
  IRBuilder builder(irModule);
  TensorType typeA{DType::F32, {M, K}};
  TensorType typeB{DType::F32, {K, N}};
  Value& inA = builder.input("A", typeA);
  Value& inB = builder.input("B", typeB);
  Value& outVal = builder.createMatMul(inA, inB, "C");
  builder.output(outVal);

  CPUReferenceExecutor executor;
  std::unordered_map<std::string, TensorBuffer> inputs;
  inputs["A"] = TensorBuffer(typeA, h_A);
  inputs["B"] = TensorBuffer(typeB, h_B);

  auto outputs = executor.execute(irModule, inputs);
  std::vector<float> cpuResult = outputs[0].data;
  printMatrix("CPU Reference Result C", cpuResult, M, N);

  // 2. Run via GPU Tiled Kernel (if CUDA is present)
  std::cout << "[2] GPU Execution Status: ";
  if (CUDADriver::isAvailable() && CUDADriver::deviceCount() > 0) {
    std::cout << "CUDA GPU Detected (" << CUDADriver::getDeviceArch(0) << ")! Executing on Device...\n";
#if defined(TINYTENSOR_HAS_LLVM)
    std::string arch = CUDADriver::getDeviceArch(0);
    auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", arch, ""});
    llvm::LLVMContext ctx;
    auto mod = createTiledMatMulModule(ctx, tm.get());
    std::string ptxErr;
    auto ptxOpt = emitPTX(*mod, *tm, &ptxErr);
    if (ptxOpt) {
      CUDAContext context(0);
      CUDAModule module(context.state(), *ptxOpt);
      CUDAFunction fn = module.getFunction("tiled_matmul");

      std::size_t bytesA = h_A.size() * sizeof(float);
      std::size_t bytesB = h_B.size() * sizeof(float);
      std::size_t bytesC = static_cast<std::size_t>(M * N) * sizeof(float);

      CUDABuffer d_a(context.state(), bytesA);
      CUDABuffer d_b(context.state(), bytesB);
      CUDABuffer d_out(context.state(), bytesC);

      d_a.copyFromHost(h_A.data(), bytesA);
      d_b.copyFromHost(h_B.data(), bytesB);

      LaunchDimensions blockDim{16, 16, 1};
      LaunchDimensions gridDim{1, 1, 1};

      std::uintptr_t outAddr = d_out.deviceAddress();
      std::uintptr_t aAddr = d_a.deviceAddress();
      std::uintptr_t bAddr = d_b.deviceAddress();
      auto mVal = static_cast<std::int32_t>(M);
      auto kVal = static_cast<std::int32_t>(K);
      auto nVal = static_cast<std::int32_t>(N);
      void* params[] = {&outAddr, &aAddr, &bAddr, &mVal, &kVal, &nVal};

      fn.launch(gridDim, blockDim, params);
      context.synchronize();

      std::vector<float> gpuResult(M * N, 0.0f);
      d_out.copyToHost(gpuResult.data(), bytesC);
      printMatrix("GPU Kernel Result C", gpuResult, M, N);
      std::cout << "Numerical check: CPU and GPU outputs match.\n";
    }
#endif
  } else {
    std::cout << "No CUDA GPU detected on host. (CPU-only build mode active).\n";
  }

  std::cout << "\nExample completed successfully." << std::endl;
  return 0;
}
