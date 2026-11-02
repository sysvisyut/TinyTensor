#include <gtest/gtest.h>

#include "tinytensor/backend/nvptx_target.hpp"
#include "tinytensor/backend/tiled_matmul.hpp"
#include "tinytensor/runtime/cuda_driver.hpp"

#if defined(TINYTENSOR_HAS_LLVM)
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#endif

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace tinytensor;
using namespace tinytensor::backend;
using namespace tinytensor::runtime;

#if defined(TINYTENSOR_HAS_LLVM)

// ----------------------------------------------------------------------------
// Test 1: LLVM IR Generation and Module Verification
// ----------------------------------------------------------------------------
TEST(TiledMatMulLLVMTest, ModuleVerificationAndIRStructure) {
  llvm::LLVMContext ctx;
  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", "sm_70", ""});
  ASSERT_NE(tm, nullptr);

  auto mod = createTiledMatMulModule(ctx, tm.get(), "mod_tiled_matmul", "tiled_matmul");
  ASSERT_NE(mod, nullptr);

  std::string verifyErr;
  bool verified = verifyLLVMModule(*mod, &verifyErr);
  EXPECT_TRUE(verified) << "Module verification failed: " << verifyErr;

  std::string irStr = emitLLVMIR(*mod);
  EXPECT_FALSE(irStr.empty());

  // Check required structural elements in generated LLVM IR
  EXPECT_NE(irStr.find("@tileA = internal addrspace(3) global [256 x float]"), std::string::npos)
      << "Expected @tileA shared memory array in AddressSpace 3";
  EXPECT_NE(irStr.find("@tileB = internal addrspace(3) global [256 x float]"), std::string::npos)
      << "Expected @tileB shared memory array in AddressSpace 3";
  EXPECT_NE(irStr.find("@llvm.nvvm.barrier"), std::string::npos)
      << "Expected barrier intrinsic calls in IR";
  EXPECT_NE(irStr.find("@llvm.nvvm.read.ptx.sreg.tid.x"), std::string::npos)
      << "Expected tid.x register read";
  EXPECT_NE(irStr.find("@llvm.nvvm.read.ptx.sreg.tid.y"), std::string::npos)
      << "Expected tid.y register read";
  EXPECT_NE(irStr.find("@llvm.nvvm.read.ptx.sreg.ctaid.x"), std::string::npos)
      << "Expected ctaid.x register read";
  EXPECT_NE(irStr.find("@llvm.nvvm.read.ptx.sreg.ctaid.y"), std::string::npos)
      << "Expected ctaid.y register read";
}

// ----------------------------------------------------------------------------
// Test 2: PTX Assembly Emission and Directives Verification
// ----------------------------------------------------------------------------
TEST(TiledMatMulLLVMTest, PTXEmissionContainsSharedMemoryAndBarriers) {
  llvm::LLVMContext ctx;
  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", "sm_70", ""});
  ASSERT_NE(tm, nullptr);

  auto mod = createTiledMatMulModule(ctx, tm.get(), "mod_tiled_matmul", "tiled_matmul");
  ASSERT_NE(mod, nullptr);

  std::string emitErr;
  auto ptxOpt = emitPTX(*mod, *tm, &emitErr);
  ASSERT_TRUE(ptxOpt.has_value()) << "PTX emission failed: " << emitErr;

  const std::string& ptx = *ptxOpt;
  EXPECT_NE(ptx.find(".entry tiled_matmul"), std::string::npos);
  EXPECT_NE(ptx.find(".shared .align 16"), std::string::npos);
  EXPECT_NE(ptx.find("bar.sync"), std::string::npos) << "Emitted PTX:\n" << ptx;
  EXPECT_NE(ptx.find("%tid.x"), std::string::npos);
  EXPECT_NE(ptx.find("%tid.y"), std::string::npos);
  EXPECT_NE(ptx.find("%ctaid.x"), std::string::npos);
  EXPECT_NE(ptx.find("%ctaid.y"), std::string::npos);
}

#endif // TINYTENSOR_HAS_LLVM

// ----------------------------------------------------------------------------
// Helper: CPU reference matrix multiplication
// ----------------------------------------------------------------------------
static std::vector<float> cpuMatMulReference(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int K, int N) {
  std::vector<float> C(static_cast<std::size_t>(M * N), 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int p = 0; p < K; ++p) {
      float aVal = A[static_cast<std::size_t>(i * K + p)];
      for (int j = 0; j < N; ++j) {
        C[static_cast<std::size_t>(i * N + j)] += aVal * B[static_cast<std::size_t>(p * N + j)];
      }
    }
  }
  return C;
}

#if defined(TINYTENSOR_HAS_LLVM)

// Helper: Run GPU tiled matmul and compare with CPU reference
static void runAndValidateTiledMatMul(int M, int K, int N, float tolerance = 1e-3f) {
  if (!CUDADriver::isAvailable() || CUDADriver::deviceCount() == 0) {
    GTEST_SKIP() << "CUDA GPU not available on this host. Skipping GPU execution test.";
  }

  // 1. Detect target GPU architecture
  std::string arch = CUDADriver::getDeviceArch(0);
  ASSERT_FALSE(arch.empty());

  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", arch, ""});
  ASSERT_NE(tm, nullptr) << "Failed to construct TargetMachine for " << arch;

  llvm::LLVMContext ctx;
  auto llvmMod = createTiledMatMulModule(ctx, tm.get(), "mod_tiled_matmul", "tiled_matmul");
  ASSERT_NE(llvmMod, nullptr);

  std::string verifyErr;
  ASSERT_TRUE(verifyLLVMModule(*llvmMod, &verifyErr)) << "LLVM verification failed: " << verifyErr;

  std::string emitErr;
  auto ptxOpt = emitPTX(*llvmMod, *tm, &emitErr);
  ASSERT_TRUE(ptxOpt.has_value()) << "PTX emission failed: " << emitErr;

  // 2. Load module into CUDA driver
  CUDAContext context(0);
  CUDAModule module(context.state(), *ptxOpt);
  CUDAFunction fn = module.getFunction("tiled_matmul");

  // 3. Initialize host data
  auto numA = static_cast<std::size_t>(M * K);
  auto numB = static_cast<std::size_t>(K * N);
  auto numC = static_cast<std::size_t>(M * N);

  std::vector<float> h_A(numA);
  std::vector<float> h_B(numB);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < numA; ++i) {
    h_A[i] = dist(rng);
  }
  for (std::size_t i = 0; i < numB; ++i) {
    h_B[i] = dist(rng);
  }

  // 4. Compute CPU reference
  std::vector<float> h_expected = cpuMatMulReference(h_A, h_B, M, K, N);

  // 5. Allocate device buffers
  CUDABuffer d_out(context.state(), numC * sizeof(float));
  CUDABuffer d_a(context.state(), numA * sizeof(float));
  CUDABuffer d_b(context.state(), numB * sizeof(float));

  d_a.copyFromHost(h_A.data(), numA * sizeof(float));
  d_b.copyFromHost(h_B.data(), numB * sizeof(float));

  // 6. Launch configuration: 16x16 threads per block
  constexpr int kTile = 16;
  LaunchDimensions blockDim{static_cast<unsigned int>(kTile), static_cast<unsigned int>(kTile), 1};
  auto gridX = static_cast<unsigned int>((N + kTile - 1) / kTile);
  auto gridY = static_cast<unsigned int>((M + kTile - 1) / kTile);
  LaunchDimensions gridDim{gridX > 0 ? gridX : 1, gridY > 0 ? gridY : 1, 1};

  std::uintptr_t outAddr = d_out.deviceAddress();
  std::uintptr_t aAddr = d_a.deviceAddress();
  std::uintptr_t bAddr = d_b.deviceAddress();
  auto mVal = static_cast<std::int32_t>(M);
  auto kVal = static_cast<std::int32_t>(K);
  auto nVal = static_cast<std::int32_t>(N);

  void* kernelParams[] = {&outAddr, &aAddr, &bAddr, &mVal, &kVal, &nVal};

  auto t0 = std::chrono::high_resolution_clock::now();
  fn.launch(gridDim, blockDim, kernelParams);
  context.synchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // 7. Copy result back and verify against CPU reference
  std::vector<float> h_gpu(numC, 0.0f);
  d_out.copyToHost(h_gpu.data(), numC * sizeof(float));

  float maxDiff = 0.0f;
  for (std::size_t i = 0; i < numC; ++i) {
    float diff = std::abs(h_gpu[i] - h_expected[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }

  std::cout << "[TiledMatMul] Shape (" << M << "x" << K << " * " << K << "x" << N
            << ") -> Grid: (" << gridDim.x << ", " << gridDim.y << "), Block: ("
            << blockDim.x << ", " << blockDim.y << ") MaxDiff: " << maxDiff
            << ", Observed Time: " << elapsedMs << " ms" << std::endl;

  EXPECT_LE(maxDiff, tolerance)
      << "GPU result diverged from CPU reference for shape ("
      << M << "x" << K << " * " << K << "x" << N << ")";
}

// ----------------------------------------------------------------------------
// Test 3: Square tile-aligned matrices (16x16, 32x32)
// ----------------------------------------------------------------------------
TEST(TiledMatMulGPUTest, SquareTileAligned16x16And32x32) {
  runAndValidateTiledMatMul(16, 16, 16);
  runAndValidateTiledMatMul(32, 32, 32);
}

// ----------------------------------------------------------------------------
// Test 4: Rectangular non-tile-aligned matrix (17x19 * 19x23) from task matrix
// ----------------------------------------------------------------------------
TEST(TiledMatMulGPUTest, NonTileAlignedRectangular17x19x23) {
  runAndValidateTiledMatMul(17, 19, 23);
}

// ----------------------------------------------------------------------------
// Test 5: Boundary shapes (1x1, 1x17 * 17x1, 33x1 * 1x33)
// ----------------------------------------------------------------------------
TEST(TiledMatMulGPUTest, BoundaryShapes) {
  runAndValidateTiledMatMul(1, 1, 1);
  runAndValidateTiledMatMul(1, 17, 1);
  runAndValidateTiledMatMul(33, 1, 33);
}

// ----------------------------------------------------------------------------
// Test 6: Scaled sizes (256x256, 512x256 * 256x512) from task matrix
// ----------------------------------------------------------------------------
TEST(TiledMatMulGPUTest, ScaledMatrices256And512) {
  runAndValidateTiledMatMul(256, 256, 256, /*tolerance=*/5e-3f);
  runAndValidateTiledMatMul(512, 256, 512, /*tolerance=*/5e-3f);
}

#endif // TINYTENSOR_HAS_LLVM
