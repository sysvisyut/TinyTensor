#include "tinytensor/backend/nvptx_target.hpp"
#include "tinytensor/backend/naive_matmul.hpp"
#include "tinytensor/backend/tiled_matmul.hpp"
#include "tinytensor/ir.hpp"
#include "tinytensor/pass.hpp"
#include "tinytensor/passes/fusion.hpp"
#include "tinytensor/runtime/cpu_reference.hpp"
#include "tinytensor/runtime/cuda_driver.hpp"

#if defined(TINYTENSOR_HAS_LLVM)
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace tinytensor;
using namespace tinytensor::backend;
using namespace tinytensor::runtime;

namespace {

// ----------------------------------------------------------------------------
// Benchmark Statistical Summary
// ----------------------------------------------------------------------------
struct BenchmarkStats {
  std::size_t iterations{0};
  double minMs{0.0};
  double medianMs{0.0};
  double meanMs{0.0};
  double maxMs{0.0};
  double stddevMs{0.0};
  double gflops{0.0};

  static BenchmarkStats compute(std::vector<double> samplesMs, double totalFlops = 0.0) {
    BenchmarkStats s;
    if (samplesMs.empty()) return s;

    s.iterations = samplesMs.size();
    std::sort(samplesMs.begin(), samplesMs.end());

    s.minMs = samplesMs.front();
    s.maxMs = samplesMs.back();

    if (s.iterations % 2 == 1) {
      s.medianMs = samplesMs[s.iterations / 2];
    } else {
      s.medianMs = 0.5 * (samplesMs[s.iterations / 2 - 1] + samplesMs[s.iterations / 2]);
    }

    double sum = std::accumulate(samplesMs.begin(), samplesMs.end(), 0.0);
    s.meanMs = sum / static_cast<double>(s.iterations);

    double varSum = 0.0;
    for (double v : samplesMs) {
      double diff = v - s.meanMs;
      varSum += diff * diff;
    }
    s.stddevMs = std::sqrt(varSum / static_cast<double>(s.iterations));

    if (totalFlops > 0.0 && s.medianMs > 0.0) {
      // GFLOPS = (FLOP / 1e9) / (seconds) = FLOP / (medianMs * 1e6)
      s.gflops = totalFlops / (s.medianMs * 1.0e6);
    }

    return s;
  }
};

void printStatsRow(std::string_view label, const BenchmarkStats& stats) {
  std::cout << "  " << std::left << std::setw(34) << label << " | "
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(8) << stats.medianMs << " ms | "
            << std::setw(8) << stats.minMs << " ms | "
            << std::setw(8) << stats.maxMs << " ms | "
            << std::setw(8) << stats.stddevMs << " ms | "
            << std::setprecision(2)
            << std::setw(8) << stats.gflops << " GFLOPS"
            << std::endl;
}

// ----------------------------------------------------------------------------
// CPU Matrix Multiplication Baseline
// ----------------------------------------------------------------------------
void cpuMatMul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N) {
  std::fill(C.begin(), C.end(), 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int p = 0; p < K; ++p) {
      float aVal = A[static_cast<std::size_t>(i * K + p)];
      for (int j = 0; j < N; ++j) {
        C[static_cast<std::size_t>(i * N + j)] += aVal * B[static_cast<std::size_t>(p * N + j)];
      }
    }
  }
}

// ----------------------------------------------------------------------------
// 1. Compilation and JIT Benchmark (Separated from Execution)
// ----------------------------------------------------------------------------
void benchmarkCompilationAndJIT() {
  std::cout << "\n================================================================================"
            << "\n 1. Compilation & JIT Time (Separated from Execution)"
            << "\n================================================================================\n"
            << "  Target Architecture: sm_70 / nvptx64-nvidia-cuda (10 iterations per kernel)\n"
            << "  " << std::left << std::setw(34) << "Kernel Generator" << " | "
            << std::right
            << std::setw(11) << "Median (ms)" << " | "
            << std::setw(11) << "Min (ms)" << " | "
            << std::setw(11) << "Max (ms)" << " | "
            << std::setw(11) << "StdDev (ms)"
            << "\n--------------------------------------------------------------------------------\n";

#if defined(TINYTENSOR_HAS_LLVM)
  auto runCompileBench = [](const std::string& name, auto generator) {
    constexpr int kIterations = 10;
    std::vector<double> samples;
    samples.reserve(kIterations);

    for (int i = 0; i < kIterations; ++i) {
      auto t0 = std::chrono::high_resolution_clock::now();
      llvm::LLVMContext ctx;
      auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", "sm_70", ""});
      auto mod = generator(ctx, tm.get());
      std::string ptxErr;
      auto ptx = emitPTX(*mod, *tm, &ptxErr);
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      samples.push_back(ms);
    }
    BenchmarkStats stats = BenchmarkStats::compute(samples);
    std::cout << "  " << std::left << std::setw(34) << name << " | "
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(8) << stats.medianMs << " ms | "
              << std::setw(8) << stats.minMs << " ms | "
              << std::setw(8) << stats.maxMs << " ms | "
              << std::setw(8) << stats.stddevMs << " ms"
              << std::endl;
  };

  runCompileBench("VectorAdd (Elementwise)", [](llvm::LLVMContext& c, llvm::TargetMachine* tm) {
    return createVectorAddModule(c, tm);
  });
  runCompileBench("Naive MatMul (Global)", [](llvm::LLVMContext& c, llvm::TargetMachine* tm) {
    return createNaiveMatMulModule(c, tm);
  });
  runCompileBench("Tiled MatMul (16x16 AS3)", [](llvm::LLVMContext& c, llvm::TargetMachine* tm) {
    return createTiledMatMulModule(c, tm);
  });
#else
  std::cout << "  LLVM not compiled in; compilation benchmark skipped.\n";
#endif
}

// ----------------------------------------------------------------------------
// 2. CPU Reference vs GPU Tiled MatMul (Meaningful Subset)
// ----------------------------------------------------------------------------
void benchmarkCpuVsGpuMatMul() {
  std::cout << "\n================================================================================"
            << "\n 2. CPU Reference vs GPU Tiled MatMul (Meaningful Comparison Subset)"
            << "\n================================================================================\n"
            << "  Warmup: 5 passes | Timed: 20 iterations | Reported: Median & GFLOPS\n"
            << "  Note: CPU Reference is O(N^3) single-threaded C++ and skipped for >= 512.\n"
            << "  " << std::left << std::setw(34) << "Workload & Device" << " | "
            << std::right
            << std::setw(11) << "Median (ms)" << " | "
            << std::setw(11) << "Min (ms)" << " | "
            << std::setw(11) << "Max (ms)" << " | "
            << std::setw(11) << "StdDev (ms)" << " | "
            << std::setw(11) << "Throughput"
            << "\n--------------------------------------------------------------------------------\n";

  struct Shape { int m; int k; int n; std::string label; };
  std::vector<Shape> shapes = {
    {17, 19, 23, "Rectangular (17x19 * 19x23)"},
    {64, 64, 64, "Square (64x64)"},
    {128, 128, 128, "Square (128x128)"},
    {256, 256, 256, "Square (256x256)"}
  };

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (const auto& sh : shapes) {
    double flops = 2.0 * static_cast<double>(sh.m) * static_cast<double>(sh.k) * static_cast<double>(sh.n);
    std::vector<float> h_A(static_cast<std::size_t>(sh.m * sh.k));
    std::vector<float> h_B(static_cast<std::size_t>(sh.k * sh.n));
    std::vector<float> h_C(static_cast<std::size_t>(sh.m * sh.n), 0.0f);
    for (float& x : h_A) x = dist(rng);
    for (float& x : h_B) x = dist(rng);

    // CPU Reference Timing
    constexpr int kCpuRuns = 15;
    for (int w = 0; w < 3; ++w) cpuMatMul(h_A, h_B, h_C, sh.m, sh.k, sh.n);

    std::vector<double> cpuSamples;
    cpuSamples.reserve(kCpuRuns);
    for (int r = 0; r < kCpuRuns; ++r) {
      auto t0 = std::chrono::high_resolution_clock::now();
      cpuMatMul(h_A, h_B, h_C, sh.m, sh.k, sh.n);
      auto t1 = std::chrono::high_resolution_clock::now();
      cpuSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    printStatsRow("CPU Reference: " + sh.label, BenchmarkStats::compute(cpuSamples, flops));

    // GPU Tiled MatMul Timing (if CUDA available)
    if (CUDADriver::isAvailable() && CUDADriver::deviceCount() > 0) {
#if defined(TINYTENSOR_HAS_LLVM)
      try {
        std::string arch = CUDADriver::getDeviceArch(0);
        auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", arch, ""});
        llvm::LLVMContext ctx;
        auto mod = createTiledMatMulModule(ctx, tm.get());
        std::string ptxErr;
        auto ptx = emitPTX(*mod, *tm, &ptxErr);
        if (ptx) {
          CUDAContext context(0);
          CUDAModule module(context.state(), *ptx);
          CUDAFunction fn = module.getFunction("tiled_matmul");

          std::size_t bytesA = h_A.size() * sizeof(float);
          std::size_t bytesB = h_B.size() * sizeof(float);
          std::size_t bytesC = h_C.size() * sizeof(float);

          CUDABuffer d_a(context.state(), bytesA);
          CUDABuffer d_b(context.state(), bytesB);
          CUDABuffer d_out(context.state(), bytesC);
          d_a.copyFromHost(h_A.data(), bytesA);
          d_b.copyFromHost(h_B.data(), bytesB);

          constexpr int kTile = 16;
          LaunchDimensions blockDim{static_cast<unsigned int>(kTile), static_cast<unsigned int>(kTile), 1};
          auto gridX = static_cast<unsigned int>((sh.n + kTile - 1) / kTile);
          auto gridY = static_cast<unsigned int>((sh.m + kTile - 1) / kTile);
          LaunchDimensions gridDim{gridX > 0 ? gridX : 1, gridY > 0 ? gridY : 1, 1};

          std::uintptr_t outAddr = d_out.deviceAddress();
          std::uintptr_t aAddr = d_a.deviceAddress();
          std::uintptr_t bAddr = d_b.deviceAddress();
          auto mVal = static_cast<std::int32_t>(sh.m);
          auto kVal = static_cast<std::int32_t>(sh.k);
          auto nVal = static_cast<std::int32_t>(sh.n);
          void* params[] = {&outAddr, &aAddr, &bAddr, &mVal, &kVal, &nVal};

          // Warmup
          for (int w = 0; w < 5; ++w) {
            fn.launch(gridDim, blockDim, params);
          }
          context.synchronize();

          // Timed runs
          constexpr int kGpuRuns = 30;
          std::vector<double> gpuSamples;
          gpuSamples.reserve(kGpuRuns);
          for (int r = 0; r < kGpuRuns; ++r) {
            context.synchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            fn.launch(gridDim, blockDim, params);
            context.synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            gpuSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
          }
          printStatsRow("GPU Tiled:     " + sh.label, BenchmarkStats::compute(gpuSamples, flops));
        }
      } catch (const std::exception& e) {
        std::cout << "  GPU benchmark exception: " << e.what() << "\n";
      }
#endif
    } else {
      std::cout << "  GPU Tiled:     " << std::left << std::setw(20) << sh.label
                << " | [SKIPPED - No CUDA GPU Available on Host]\n";
    }
  }
}

// ----------------------------------------------------------------------------
// 3. Unfused vs Fused Execution
// ----------------------------------------------------------------------------
void benchmarkUnfusedVsFused() {
  std::cout << "\n================================================================================"
            << "\n 3. Unfused vs Fused Execution (CPU Reference & Memory Reduction)"
            << "\n================================================================================\n"
            << "  Workload: MatMul(M, K, N) + Bias(N) -> ReLU\n"
            << "  Comparison: Materialized intermediate buffers vs fused epilogue evaluation\n"
            << "  " << std::left << std::setw(34) << "Pipeline Mode" << " | "
            << std::right
            << std::setw(11) << "Median (ms)" << " | "
            << std::setw(11) << "Min (ms)" << " | "
            << std::setw(11) << "Max (ms)" << " | "
            << std::setw(11) << "StdDev (ms)" << " | "
            << std::setw(11) << "Speedup"
            << "\n--------------------------------------------------------------------------------\n";

  constexpr int M = 128;
  constexpr int K = 128;
  constexpr int N = 128;
  constexpr double flops = 2.0 * M * K * N + 2.0 * M * N;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_Bias(N);
  for (float& x : h_A) x = dist(rng);
  for (float& x : h_B) x = dist(rng);
  for (float& x : h_Bias) x = dist(rng);

  // Unfused evaluation: Matmul to C_tmp -> Add bias to Add_tmp -> ReLU to Out
  auto runUnfused = [&]() {
    std::vector<float> cTmp(M * N, 0.0f);
    cpuMatMul(h_A, h_B, cTmp, M, K, N);
    std::vector<float> addTmp(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        addTmp[static_cast<std::size_t>(i * N + j)] =
            cTmp[static_cast<std::size_t>(i * N + j)] + h_Bias[static_cast<std::size_t>(j)];
      }
    }
    std::vector<float> out(M * N, 0.0f);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = std::max(0.0f, addTmp[i]);
    }
    return out;
  };

  // Fused evaluation: Matmul with epilogue (+bias and relu) in single pass
  auto runFused = [&]() {
    std::vector<float> out(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float accum = 0.0f;
        for (int p = 0; p < K; ++p) {
          accum += h_A[static_cast<std::size_t>(i * K + p)] * h_B[static_cast<std::size_t>(p * N + j)];
        }
        accum += h_Bias[static_cast<std::size_t>(j)];
        out[static_cast<std::size_t>(i * N + j)] = std::max(0.0f, accum);
      }
    }
    return out;
  };

  // Warmup
  for (int w = 0; w < 3; ++w) {
    runUnfused();
    runFused();
  }

  constexpr int kRuns = 25;
  std::vector<double> unfusedSamples;
  std::vector<double> fusedSamples;
  unfusedSamples.reserve(kRuns);
  fusedSamples.reserve(kRuns);

  for (int r = 0; r < kRuns; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto resUnfused = runUnfused();
    auto t1 = std::chrono::high_resolution_clock::now();
    unfusedSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

    auto t2 = std::chrono::high_resolution_clock::now();
    auto resFused = runFused();
    auto t3 = std::chrono::high_resolution_clock::now();
    fusedSamples.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
  }

  BenchmarkStats unfusedStats = BenchmarkStats::compute(unfusedSamples, flops);
  BenchmarkStats fusedStats = BenchmarkStats::compute(fusedSamples, flops);

  double speedup = (fusedStats.medianMs > 0.0) ? (unfusedStats.medianMs / fusedStats.medianMs) : 1.0;

  printStatsRow("Unfused (3 Pass + 2 Temp Buffers)", unfusedStats);
  std::cout << "  " << std::left << std::setw(34) << "Fused (Single-Pass Epilogue)" << " | "
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(8) << fusedStats.medianMs << " ms | "
            << std::setw(8) << fusedStats.minMs << " ms | "
            << std::setw(8) << fusedStats.maxMs << " ms | "
            << std::setw(8) << fusedStats.stddevMs << " ms | "
            << std::setprecision(2)
            << std::setw(8) << speedup << "x speedup"
            << std::endl;
  std::cout << "  Intermediate Memory Saved: " << (2 * M * N * sizeof(float)) / 1024 << " KiB per launch.\n";
}

// ----------------------------------------------------------------------------
// 4. Naive vs Tiled GPU MatMul (Scaling and Memory Hierarchy)
// ----------------------------------------------------------------------------
void benchmarkNaiveVsTiledGpu() {
  std::cout << "\n================================================================================"
            << "\n 4. Naive vs Tiled GPU MatMul (Memory Hierarchy & Tiling Impact)"
            << "\n================================================================================\n";

  if (!CUDADriver::isAvailable() || CUDADriver::deviceCount() == 0) {
    std::cout << "  [CUDA GPU NOT AVAILABLE ON CURRENT HOST]\n"
              << "  To execute this benchmark on an NVIDIA GPU environment, run:\n"
              << "    cmake -B build-cuda -S . -DTINYTENSOR_ENABLE_CUDA=ON\n"
              << "    cmake --build build-cuda --target tinytensor_benchmark\n"
              << "    ./build-cuda/bench/tinytensor_benchmark\n"
              << "--------------------------------------------------------------------------------\n";
    return;
  }

#if defined(TINYTENSOR_HAS_LLVM)
  std::cout << "  " << std::left << std::setw(26) << "Matrix Shape" << " | "
            << std::right
            << std::setw(12) << "Naive (ms)" << " | "
            << std::setw(12) << "Tiled (ms)" << " | "
            << std::setw(14) << "Tiled GFLOPS" << " | "
            << std::setw(10) << "Speedup"
            << "\n--------------------------------------------------------------------------------\n";

  std::vector<int> dimensions = {128, 256, 512, 1024};
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::string arch = CUDADriver::getDeviceArch(0);
  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", arch, ""});

  llvm::LLVMContext ctxNaive;
  auto modNaive = createNaiveMatMulModule(ctxNaive, tm.get());
  std::string ptxNaiveErr;
  auto ptxNaive = emitPTX(*modNaive, *tm, &ptxNaiveErr);

  llvm::LLVMContext ctxTiled;
  auto modTiled = createTiledMatMulModule(ctxTiled, tm.get());
  std::string ptxTiledErr;
  auto ptxTiled = emitPTX(*modTiled, *tm, &ptxTiledErr);

  if (!ptxNaive || !ptxTiled) {
    std::cout << "  Failed to compile PTX for naive or tiled module.\n";
    return;
  }

  CUDAContext context(0);
  CUDAModule moduleNaive(context.state(), *ptxNaive);
  CUDAFunction fnNaive = moduleNaive.getFunction("naive_matmul");

  CUDAModule moduleTiled(context.state(), *ptxTiled);
  CUDAFunction fnTiled = moduleTiled.getFunction("tiled_matmul");

  for (int dim : dimensions) {
    int M = dim, K = dim, N = dim;
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);

    std::vector<float> h_A(static_cast<std::size_t>(M * K));
    std::vector<float> h_B(static_cast<std::size_t>(K * N));
    for (float& x : h_A) x = dist(rng);
    for (float& x : h_B) x = dist(rng);

    std::size_t bytesA = h_A.size() * sizeof(float);
    std::size_t bytesB = h_B.size() * sizeof(float);
    std::size_t bytesC = static_cast<std::size_t>(M * N) * sizeof(float);

    CUDABuffer d_a(context.state(), bytesA);
    CUDABuffer d_b(context.state(), bytesB);
    CUDABuffer d_out(context.state(), bytesC);
    d_a.copyFromHost(h_A.data(), bytesA);
    d_b.copyFromHost(h_B.data(), bytesB);

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
    void* params[] = {&outAddr, &aAddr, &bAddr, &mVal, &kVal, &nVal};

    // Warmup Naive
    for (int w = 0; w < 5; ++w) fnNaive.launch(gridDim, blockDim, params);
    context.synchronize();

    constexpr int kRuns = 25;
    std::vector<double> naiveSamples;
    naiveSamples.reserve(kRuns);
    for (int r = 0; r < kRuns; ++r) {
      context.synchronize();
      auto t0 = std::chrono::high_resolution_clock::now();
      fnNaive.launch(gridDim, blockDim, params);
      context.synchronize();
      auto t1 = std::chrono::high_resolution_clock::now();
      naiveSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Warmup Tiled
    for (int w = 0; w < 5; ++w) fnTiled.launch(gridDim, blockDim, params);
    context.synchronize();

    std::vector<double> tiledSamples;
    tiledSamples.reserve(kRuns);
    for (int r = 0; r < kRuns; ++r) {
      context.synchronize();
      auto t0 = std::chrono::high_resolution_clock::now();
      fnTiled.launch(gridDim, blockDim, params);
      context.synchronize();
      auto t1 = std::chrono::high_resolution_clock::now();
      tiledSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    BenchmarkStats statsNaive = BenchmarkStats::compute(naiveSamples, flops);
    BenchmarkStats statsTiled = BenchmarkStats::compute(tiledSamples, flops);

    double speedup = (statsTiled.medianMs > 0.0) ? (statsNaive.medianMs / statsTiled.medianMs) : 1.0;

    std::string shapeLabel = std::to_string(dim) + "x" + std::to_string(dim) + " Square";
    std::cout << "  " << std::left << std::setw(26) << shapeLabel << " | "
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(9) << statsNaive.medianMs << " ms | "
              << std::setw(9) << statsTiled.medianMs << " ms | "
              << std::setprecision(2)
              << std::setw(11) << statsTiled.gflops << " GFLOPS | "
              << std::setw(7) << speedup << "x"
              << std::endl;
  }
#endif
}

// ----------------------------------------------------------------------------
// 5. System Profiling and Header
// ----------------------------------------------------------------------------
void printSystemHeader() {
  std::cout << "================================================================================\n"
            << "                       TinyTensor Benchmark Suite (Task 13)                     \n"
            << "================================================================================\n"
            << "System & Toolchain Profile:\n"
#if defined(__APPLE__)
            << "  Host OS:            macOS Darwin (arm64)\n"
#elif defined(__linux__)
            << "  Host OS:            Linux (x86_64/arm64)\n"
#else
            << "  Host OS:            Generic Host\n"
#endif
#if defined(__clang__)
            << "  Compiler:           Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n"
#elif defined(__GNUC__)
            << "  Compiler:           GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n"
#endif
            << "  C++ Standard:       " << __cplusplus << "\n"
#if defined(LLVM_VERSION_STRING)
            << "  LLVM Version:       " << LLVM_VERSION_STRING << " (Target: nvptx64-nvidia-cuda)\n"
#endif
            << "  CUDA Driver API:    ";

  if (CUDADriver::isAvailable() && CUDADriver::deviceCount() > 0) {
    std::cout << "Enabled (Devices Found: " << CUDADriver::deviceCount()
              << ", Arch: " << CUDADriver::getDeviceArch(0) << ")\n";
  } else {
    std::cout << "Disabled / Unavailable (Running in CPU-Only Build Mode)\n";
  }

  std::cout << "--------------------------------------------------------------------------------\n"
            << "Measurement Principles:\n"
            << "  - Compilation/JIT time separated from execution time.\n"
            << "  - Warmup iterations (5 passes) performed before active timing.\n"
            << "  - GPU launches synchronized before timer start and after kernel completion.\n"
            << "  - Multi-iteration sampling (15-30 runs) reporting Median to reject noise/spikes.\n"
            << "================================================================================\n";
}

} // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  printSystemHeader();
  benchmarkCompilationAndJIT();
  benchmarkCpuVsGpuMatMul();
  benchmarkUnfusedVsFused();
  benchmarkNaiveVsTiledGpu();

  std::cout << "\n================================================================================"
            << "\n 5. Bottleneck Analysis & Next Optimization Opportunities"
            << "\n================================================================================\n"
            << "Key Observations & Likely Bottlenecks:\n"
            << "  1. Shared Memory Bank Conflicts:\n"
            << "     With a 16x16 tile of 32-bit floats, threads within a warp access elements\n"
            << "     with stride 16 or consecutive rows, causing 2-way bank conflicts on 32-bank\n"
            << "     shared memory. Adding 1-element padding [16x17] eliminates this conflict.\n"
            << "  2. Fixed 16x16 Tile Size & Arithmetic Intensity:\n"
            << "     16x16 tiles achieve an arithmetic intensity of 8 FLOP/byte. Expanding to\n"
            << "     32x32 or 64x64 tiles with thread-level register tiling (e.g., 4x4 elements\n"
            << "     computed per thread) dramatically raises compute density.\n"
            << "  3. Global Memory Read Coalescing:\n"
            << "     Matrix A loads currently strided across K; transposing A or packing into\n"
            << "     float4 vectorized loads (128-bit) would saturate memory bus bandwidth.\n"
            << "  4. Software Pipelining / Asynchronous Copies:\n"
            << "     Shared memory tiles are currently loaded synchronously between full barriers.\n"
            << "     Double-buffering allows overlapping global memory loads of phase (p+1)\n"
            << "     with shared-memory math computation of phase p.\n"
            << "================================================================================\n"
            << std::endl;

  return 0;
}
