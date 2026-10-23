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
| **3** | Pass Manager: sequential pipeline, verification hooks, dirty tracking | Pending |
| **4** | Frontend DSL: Lexer, recursive-descent parser, error diagnostics | Pending |
| **5** | Optimization: Dead-Code Elimination (DCE), Constant Folding, Algebraic Simplification | Pending |
| **6** | Optimization: Common Subexpression Elimination (CSE) | Pending |
| **7** | Optimization: Elementwise and MatMul epilogue kernel fusion | Pending |
| **8** | CPU Reference Interpreter: execution oracle for test validation | Pending |
| **9** | Backend: LLVM NVPTX target setup and index-write smoke test | Pending |
| **10** | Runtime: CUDA Driver API wrapper (memory, module load, kernel launch) | Pending |
| **11** | Backend Lowering: Unfused elementwise and naive matmul kernels | Pending |
| **12** | Backend Lowering: Tiled shared-memory matmul kernel (16x16 AS3) | Pending |
| **13** | Benchmarks: Fused vs unfused, naive vs tiled execution timings | Pending |
| **14** | Documentation: Architectural design review and interview notes | Pending |
