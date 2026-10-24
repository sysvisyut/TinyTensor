#include <gtest/gtest.h>

#include "tinytensor/backend/nvptx_target.hpp"
#include "tinytensor/ir.hpp"
#include "tinytensor/runtime/cpu_reference.hpp"
#include "tinytensor/runtime/cuda_driver.hpp"

#if defined(TINYTENSOR_HAS_LLVM)
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#endif

#include <cmath>
#include <vector>

using namespace tinytensor;
using namespace tinytensor::backend;
using namespace tinytensor::runtime;

// ----------------------------------------------------------------------------
// Test 1: Fallback behavior split by build mode
// ----------------------------------------------------------------------------
TEST(CUDADriverFallbackTest, UnavailableInCpuOnlyBuild) {
#if !defined(TINYTENSOR_HAS_CUDA)
  EXPECT_FALSE(CUDADriver::isAvailable());
  EXPECT_EQ(CUDADriver::deviceCount(), 0u);
  EXPECT_THROW(CUDAContext(0), CUDAException);
#else
  GTEST_SKIP() << "Compiled with TINYTENSOR_HAS_CUDA=1; CPU-only fallback test skipped.";
#endif
}

// ----------------------------------------------------------------------------
// Test 2: End-to-end vector add execution and CPU reference comparison
// ----------------------------------------------------------------------------
TEST(CUDAIntegrationTest, VectorAddF32EndToEnd) {
  if (!CUDADriver::isAvailable() || CUDADriver::deviceCount() == 0) {
    GTEST_SKIP() << "CUDA GPU not available on this host. Skipping GPU execution test.";
  }

#if defined(TINYTENSOR_HAS_LLVM)
  // 1. Build TinyTensor IR module representing elementwise Add with non-multiple shape [17, 23] (N = 391)
  Module irModule;
  IRBuilder builder(irModule);
  TensorType tensorType{DType::F32, {17, 23}};
  Value& inA = builder.input("a", tensorType);
  Value& inB = builder.input("b", tensorType);
  Value& outVal = builder.createBinary(Op::Add, inA, inB, "out");
  builder.output(outVal);

  // 2. Query target GPU architecture (e.g. sm_75 for T4, sm_80 for A100)
  std::string arch = CUDADriver::getDeviceArch(0);
  ASSERT_FALSE(arch.empty());

  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", arch, ""});
  ASSERT_NE(tm, nullptr) << "Failed to construct TargetMachine for " << arch;

  llvm::LLVMContext ctx;
  auto llvmMod = createVectorAddModule(ctx, tm.get(), "mod_vector_add", "vector_add");
  ASSERT_NE(llvmMod, nullptr);

  std::string verifyErr;
  ASSERT_TRUE(verifyLLVMModule(*llvmMod, &verifyErr)) << "LLVM verification failed: " << verifyErr;

  std::string emitErr;
  auto ptxOpt = emitPTX(*llvmMod, *tm, &emitErr);
  ASSERT_TRUE(ptxOpt.has_value()) << "PTX emission failed: " << emitErr;

  // 3. Initialize CUDA context, module, and function
  CUDAContext context(0);
  CUDAModule module(context.state(), *ptxOpt);
  CUDAFunction fn = module.getFunction("vector_add");

  // 4. Prepare host inputs and allocate device buffers
  constexpr std::size_t numElements = 391;
  constexpr std::size_t sizeBytes = numElements * sizeof(float);

  std::vector<float> h_a(numElements);
  std::vector<float> h_b(numElements);
  for (std::size_t i = 0; i < numElements; ++i) {
    auto fi = static_cast<float>(i);
    h_a[i] = 1.5f + 0.01f * fi;
    h_b[i] = 2.0f - 0.005f * fi;
  }

  CUDABuffer d_out(context.state(), sizeBytes);
  CUDABuffer d_a(context.state(), sizeBytes);
  CUDABuffer d_b(context.state(), sizeBytes);

  d_a.copyFromHost(h_a.data(), sizeBytes);
  d_b.copyFromHost(h_b.data(), sizeBytes);

  // 5. Marshall kernel parameters in stable host storage
  std::uintptr_t outAddr = d_out.deviceAddress();
  std::uintptr_t aAddr = d_a.deviceAddress();
  std::uintptr_t bAddr = d_b.deviceAddress();
  auto nElem = static_cast<std::int32_t>(numElements);

  void* kernelParams[] = {&outAddr, &aAddr, &bAddr, &nElem};

  // Launch with block size 256 and grid size ceil(391/256) = 2
  LaunchDimensions blockDim(256, 1, 1);
  LaunchDimensions gridDim(
      static_cast<unsigned int>((numElements + blockDim.x - 1) / blockDim.x), 1, 1);

  fn.launch(gridDim, blockDim, kernelParams);
  context.synchronize();

  // 6. Copy back GPU result
  std::vector<float> h_gpu_out(numElements, 0.0f);
  d_out.copyToHost(h_gpu_out.data(), sizeBytes);

  // 7. Evaluate CPU reference oracle on identical TinyTensor IR
  CPUReferenceExecutor cpu;
  TensorBuffer tb_a{tensorType, h_a};
  TensorBuffer tb_b{tensorType, h_b};
  auto cpuOutputs = cpu.execute(irModule, {{"a", tb_a}, {"b", tb_b}});
  ASSERT_EQ(cpuOutputs.size(), 1u);
  ASSERT_EQ(cpuOutputs[0].data.size(), numElements);

  // 8. Assert GPU output matches CPU reference within tolerance 1e-3
  for (std::size_t i = 0; i < numElements; ++i) {
    EXPECT_NEAR(h_gpu_out[i], cpuOutputs[0].data[i], 1e-3f)
        << "Mismatch at index " << i << " (GPU: " << h_gpu_out[i]
        << ", CPU: " << cpuOutputs[0].data[i] << ")";
  }
#endif
}

// ----------------------------------------------------------------------------
// Test 3: Malformed PTX produces JIT compiler diagnostics
// ----------------------------------------------------------------------------
TEST(CUDAIntegrationTest, MalformedPTXThrowsWithJITDiagnostics) {
  if (!CUDADriver::isAvailable() || CUDADriver::deviceCount() == 0) {
    GTEST_SKIP() << "CUDA GPU not available on this host. Skipping GPU error test.";
  }

  CUDAContext context(0);
  std::string malformedPTX =
      ".version 6.0\n"
      ".target sm_9999_invalid_arch\n"
      ".address_size 64\n"
      ".visible .entry bad_kernel() { ret; }\n";

  try {
    CUDAModule badModule(context.state(), malformedPTX);
    FAIL() << "Expected CUDAException when loading malformed PTX";
  } catch (const CUDAException& ex) {
    EXPECT_NE(ex.errorCode(), 0);
    // Ensure actionable driver diagnostic or JIT log is reported
    EXPECT_FALSE(ex.errorName().empty());
  }
}

// ----------------------------------------------------------------------------
// Test 4: Querying nonexistent kernel name throws informative error
// ----------------------------------------------------------------------------
TEST(CUDAIntegrationTest, NonexistentKernelNameThrows) {
  if (!CUDADriver::isAvailable() || CUDADriver::deviceCount() == 0) {
    GTEST_SKIP() << "CUDA GPU not available on this host. Skipping GPU error test.";
  }

#if defined(TINYTENSOR_HAS_LLVM)
  CUDAContext context(0);
  std::string arch = CUDADriver::getDeviceArch(0);
  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", arch, ""});
  ASSERT_NE(tm, nullptr);

  llvm::LLVMContext ctx;
  auto llvmMod = createSmokeTestModule(ctx, tm.get(), "mod_smoke", "tiny_smoke");
  auto ptxOpt = emitPTX(*llvmMod, *tm);
  ASSERT_TRUE(ptxOpt.has_value());

  CUDAModule module(context.state(), *ptxOpt);

  try {
    module.getFunction("nonexistent_kernel_symbol_xyz");
    FAIL() << "Expected CUDAException when looking up nonexistent kernel";
  } catch (const CUDAException& ex) {
    EXPECT_EQ(ex.errorName(), "CUDA_ERROR_NOT_FOUND");
  }
#endif
}
