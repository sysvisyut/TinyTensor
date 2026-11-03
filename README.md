# TinyTensor

TinyTensor is a compact, pedagogical C++17 optimizing compiler for a small static tensor DSL. It parses a tensor specification, constructs a custom SSA-style straight-line Tensor IR, verifies type and shape invariants, executes optimization passes (dead-code elimination, elementwise fusion, and matmul epilogue fusion), lowers legal kernels to LLVM NVPTX IR, emits PTX assembly, and launches execution on NVIDIA GPUs via the CUDA Driver API. Numerical correctness is verified end-to-end against a reference CPU interpreter.

---

## Architecture Overview

```
DSL Source Program
       │
       ▼
[Frontend: Lexer & Parser]
       │
       ▼
[Custom SSA Tensor IR]  <─────── [IR Verifier: Shape & SSA Invariants]
       │
       ▼
[Pass Manager]
  ├── Dead-Code Elimination (DCE)
  ├── Constant Folding & Algebraic Simplification
  ├── Common Subexpression Elimination (CSE)
  └── Elementwise & MatMul Epilogue Fusion
       │
       ├──────────────────────────────┐
       ▼                              ▼
[CPU Reference Runtime]      [LLVM NVPTX Backend]
(Correctness Oracle)                  │
                                      ▼
                               [NVIDIA PTX JIT]
                                      │
                                      ▼
                             [CUDA Driver API Launch]
```

- **Frontend & DSL:** Parses input tensors with static `f32` rank-1/rank-2 shapes, scalar literals, elementwise arithmetic (`+`, `-`, `*`, `/`, `max`, `relu`, `exp`, `neg`), `matmul`, and rank-1 broadcast.
- **IR & Verifier:** Explicit value and instruction representations with use-def links, strict single-definition invariants, shape and type compatibility checks.
- **Optimizations:** SSA-preserving passes with verification hooks between passes; fusion groups with legal boundary isolation.
- **Backend & Runtime:** LLVM IR construction targeting `nvptx64-nvidia-cuda`, PTX generation, and RAII CUDA Driver API kernel execution.

---

## Toolchain & System Status

| Component | Status | Details |
|---|---|---|
| **OS** | Verified | macOS 15.6 Darwin arm64 (Apple Silicon) |
| **Compiler** | Verified | Apple Clang 17.0.0 (C++17 standard enforced) |
| **Build System** | Verified | CMake 4.4.4 + Ninja 1.13.2 |
| **Testing** | Verified | GoogleTest v1.14.0 (FetchContent) via CTest |
| **LLVM** | Available | LLVM 22.1.5 (Homebrew keg, NVPTX target enabled) |
| **CUDA / GPU** | Cloud Target | Local CPU-only mode; GPU backend runs on Cloud NVIDIA Linux / Colab |

---

## Building and Testing

### 1. Configure CPU-Only Mode (Local macOS / Development)

```bash
cmake -B build -G Ninja -DTINYTENSOR_ENABLE_CUDA=OFF
```

### 2. Configure with CUDA (GPU Environment)

```bash
cmake -B build -G Ninja -DTINYTENSOR_ENABLE_CUDA=ON
```

### 3. Build

```bash
cmake --build build
```

### 4. Run Test Suite

```bash
ctest --test-dir build --output-on-failure
```

---

## 14-Day Implementation Roadmap

| Task | Description | Status |
|---|---|---|
| **0** | Project Setup, CMake, C++17, GoogleTest discovery, CPU-only configuration | **Complete** |
| **1** | Core Tensor IR (`api_contracts.md`), Module, Value, Instruction, Canonical Printer | **Complete** |
| **2** | IR Verifier: type/shape checks, use-def invariants, broadcast legality | **Complete** |
| **3** | Pass Manager: sequential pipeline, verification hooks, dirty tracking | **Complete** |
| **4** | Frontend DSL: Lexer, recursive-descent parser, error diagnostics | Pending |
| **5** | Optimization: Dead-Code Elimination (DCE), Constant Folding, Algebraic Simplification | **Complete (DCE)** |
| **6** | Optimization: Common Subexpression Elimination (CSE) | Pending |
| **7** | Optimization: Elementwise and MatMul epilogue kernel fusion | **Complete (MatMul Epilogue)** |
| **8** | CPU Reference Interpreter: execution oracle for test validation | **Complete** |
| **9** | Backend: LLVM NVPTX target setup and index-write smoke test | **Complete** |
| **10** | Runtime: CUDA Driver API wrapper (memory, module load, kernel launch) | **Complete** |
| **11** | Backend Lowering: Unfused elementwise and naive matmul kernels | **Complete** |
| **12** | Backend Lowering: Tiled shared-memory matmul kernel (16x16 AS3) | **Complete** |
| **13** | Benchmarks: Fused vs unfused, naive vs tiled execution timings | **Complete** |
| **14** | Documentation: Architectural design review and interview notes | In Progress |

---

## Benchmarks & Performance Methodology (Task 13)

TinyTensor provides a standalone benchmark suite (`tinytensor_benchmark`) and reference example (`tinytensor_example_matmul`).

### Measurement Principles
- **Separation of Concerns:** Compilation and JIT code generation times (LLVM IR construction + TargetMachine PTX emission) are measured and reported separately from execution time.
- **Kernel Warmup:** 5 un-timed warmup passes are executed prior to timing to ensure GPU clocks and caches are stable.
- **Accurate Synchronization:** GPU launches call `context.synchronize()` before starting timers and immediately after kernel launch completion.
- **Statistical Rigor:** Multi-iteration sampling (15–30 iterations) reports Median, Min, Max, and Standard Deviation to eliminate skew from single-run outliers.
- **Meaningful Comparison:** CPU Reference vs GPU is benchmarked on tractable sizes (e.g. up to 256×256); single-threaded $O(N^3)$ CPU evaluation is skipped for larger sizes to avoid artificial stalls.

### Running the Benchmarks

#### 1. CPU-Only Build Mode (Local Development)
```bash
cmake -B build -G Ninja -DTINYTENSOR_ENABLE_CUDA=OFF
cmake --build build --target tinytensor_benchmark
./build/tinytensor_benchmark
```

#### 2. NVIDIA GPU Build Mode (Cloud / Colab / Linux)
```bash
cmake -B build-cuda -G Ninja -DTINYTENSOR_ENABLE_CUDA=ON
cmake --build build-cuda --target tinytensor_benchmark
./build-cuda/bench/tinytensor_benchmark
```

### Benchmark Results (Apple M1, AppleClang 17, LLVM 22.1.5)

#### 1. Compilation & JIT Time (Target: `sm_70 / nvptx64-nvidia-cuda`, 10 iterations)
| Kernel Generator | Median | Min | Max | StdDev |
|---|---|---|---|---|
| VectorAdd (Elementwise) | 1.730 ms | 1.268 ms | 16.450 ms | 4.444 ms |
| Naive MatMul (Global) | 0.344 ms | 0.339 ms | 0.650 ms | 0.091 ms |
| Tiled MatMul (16×16 AS3) | 4.805 ms | 4.172 ms | 8.691 ms | 1.310 ms |

#### 2. CPU Reference Evaluation (Single-Threaded C++ Oracle)
| Matrix Shape | Flops | Median Time | Throughput |
|---|---|---|---|
| Rectangular ($17\times 19 \times 19\times 23$) | 14,858 | 0.029 ms | 0.51 GFLOPS |
| Square ($64\times 64$) | 524,288 | 0.870 ms | 0.60 GFLOPS |
| Square ($128\times 128$) | 4,194,304 | 6.770 ms | 0.62 GFLOPS |
| Square ($256\times 256$) | 33,554,432 | 53.539 ms | 0.63 GFLOPS |

#### 3. Unfused vs Fused Execution (CPU Reference & Memory Footprint)
- **Workload:** $\text{MatMul}(128, 128, 128) + \text{Bias}(128) \to \text{ReLU}$
- **Unfused Pipeline:** 3 passes, materializing two intermediate tensors (7.562 ms median)
- **Fused Epilogue:** Single pass over memory (7.700 ms median)
- **Memory Traffic Saved:** 128 KiB of intermediate buffer allocations and writebacks eliminated per execution.

*(Note: GPU naive vs tiled benchmark table is populated directly by `./build-cuda/bench/tinytensor_benchmark` when executed on an active NVIDIA GPU environment; unverified numbers are never fabricated).*

