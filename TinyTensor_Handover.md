# TinyTensor — 14-Day Execution Handover

**Planning start:** 23 October 2026  
**Owner:** Sharvaj Vidyut  
**Purpose:** build a credible, demoable compiler project before an NVIDIA System Software / Compiler Intern interview. This document is the complete working brief for a technical mentor LLM or collaborator.

## The honest project goal

TinyTensor is a compact C++17 optimizing compiler for a small, static tensor DSL. It parses a program, constructs a custom SSA-flavoured Tensor IR, verifies and optimizes it, lowers supported kernels to LLVM IR, emits NVIDIA PTX via LLVM NVPTX, and launches kernels through the CUDA Driver API. Correctness is checked against a CPU interpreter; performance is measured on a cloud NVIDIA GPU (for example, Google Colab).

The project is **not** intended to rival cuBLAS, TVM, XLA, Triton, or TensorRT. Its value is a clean end-to-end story: compiler representation and invariants, legality-preserving transformations, GPU mapping, real executable code generation, measurement, and honest engineering trade-offs.

### Target interview narrative

> I built a small optimizing tensor compiler to learn the boundary between compiler IR design and GPU performance. Its Tensor IR represents static f32 tensors; passes verify shapes and SSA-like use/def invariants, then eliminate dead work and fuse legal producer–consumer chains. The backend emits PTX through LLVM NVPTX and uses the CUDA Driver API to launch kernels. I used a CPU interpreter as the oracle and benchmarked unfused versus fused elementwise kernels and naïve versus tiled matmul. The results are hardware- and shape-dependent, so I report the exact GPU, median timing, and limitations rather than claiming production performance.

## Fixed constraints and decisions

| Item | Decision |
| --- | --- |
| Deadline | 14 calendar days from 2026-10-23; optimize for a working demo first. |
| Language | C++17; CMake; GoogleTest or a very small assertion-based test runner if setup becomes costly. |
| GPU | Cloud NVIDIA GPU, usually Colab. Treat its GPU/driver/LLVM/CUDA version as runtime-discovered facts. |
| Experience | Strong compiler theory; no prior LLVM C++ API experience. Start with minimal LLVM modules and a hand-written CUDA-to-LLVM-IR reference. |
| Main proof points | Correctness, elementwise fusion, tiled matmul, executable PTX, reproducible measurements, and explanations. |
| Shape model | Static rank-1/rank-2 `f32` tensors only. No dynamic shapes. |
| Precision | Floating-point tolerance around `1e-3` for ordinary test sizes; document tolerance and dimensions per test. |

## Product specification

### DSL

```text
input A: f32[1024, 1024]
input B: f32[1024, 1024]
input bias: f32[1024]
T1 = matmul(A, B)
T2 = T1 + bias
C = relu(T2)
output C
```

Inputs are declared with `input NAME: f32[dims]`; assignments are single-definition; `output NAME` selects an output. Support scalar constants and these operations:

- Binary elementwise: `+`, `-`, `*`, `/`, `max(a,b)`
- Unary elementwise: `relu(x)`, `exp(x)`, `neg(x)`
- `matmul(A,B)` for rank-2 tensors
- Rank-1 `[N]` bias broadcasting across rank-2 `[M,N]`

The parser must issue useful diagnostics for malformed syntax, undefined names, redefinition, invalid matmul shapes, mismatched elementwise shapes, and invalid broadcasts. Freeze the language after Day 3.

### IR and invariants

Build a small custom Tensor IR, not a wrapper over LLVM IR. `Module` owns functions/programs; `Value` has a unique ID and type; `Instruction` has opcode, operands, result, source location, and use information. Types carry `f32` and static dimensions. An IR printer must make every transformation inspectable.

The verifier should check at least: one definition per named result, operands are defined before use in this straight-line IR, all operands/results have compatible types and shapes, use lists agree with operands, outputs exist, and no dangling values remain. Be precise in documentation: strict CFG dominance is unnecessary until the project introduces control flow; call this *SSA-style straight-line IR* if that is what is implemented.

Use a pass manager that can print IR before/after each pass and run the verifier after every pass in debug/test builds.

### Optimizations

Implement in this value order:

1. Dead-code elimination (never remove a selected output or side-effecting operation; this DSL has no side effects).
2. Elementwise fusion for a legal single-consumer chain, lowered as one kernel with intermediates in registers.
3. Matmul epilogue fusion for `matmul + bias + relu` when the matmul result is used only by that epilogue chain.
4. Constant folding and small algebraic rewrites (`x+0`, `x*1`, nested `relu`) if time permits.
5. CSE only after the above is solid; it must account for opcode, operands, types, and commutativity if claimed.

Fusion legality: fuse only when shapes/indexing are compatible and the producer has a single consumer, unless every consumer is included in the same group. Include a negative test where a shared producer must not be fused away. A simple explicit `FusionGroup` representation is clearer than a clever implicit scheme.

### GPU backend and runtime

Lower each fusion group to a kernel. For elementwise kernels, use a flat logical index, bounds check, and broadcast-index calculation. For matmul, provide:

- **Naïve kernel:** one thread computes one output element, reading global memory.
- **Tiled kernel:** start with a `16×16` block tile, shared-memory tiles for A/B, guarded edge loads/stores, barriers, and a register accumulator.
- **Fused epilogue:** apply column bias and ReLU before the matmul output store.

LLVM backend requirements: create an LLVM `Module`, use `IRBuilder`, declare/read NVVM special-register intrinsics for thread/block indices, mark kernels with `nvvm.annotations`, use address space 3 plus the NVVM barrier intrinsic for shared memory, create an NVPTX `TargetMachine`, and emit PTX. The runtime uses the CUDA **Driver API** (`cuInit`, context setup, module load, function lookup, memory allocation/copies, launch, events, cleanup). RAII wrappers are desirable but do not delay a working baseline.

Before authoring NVVM details from memory, compile a minimal CUDA kernel to device LLVM IR in the chosen environment and inspect the generated intrinsic/metadata patterns. Preserve that generated file as a documented reference or link to the command that reproduces it.

## Recommended repository layout

```text
tinytensor/
  CMakeLists.txt                 # dependency discovery and targets
  README.md                      # results, setup, architecture, limitations
  docs/design.md                 # IR, passes, lowering, fusion legality
  examples/                      # .tt input programs
  include/tinytensor/
    frontend/ ir/ passes/ backend/ runtime/ interpreter/ support/
  src/
    frontend/ ir/ passes/ backend/ runtime/ interpreter/ cli/
  tests/
    unit/ integration/ negative/ golden/
  bench/                         # benchmark executable + CSV/JSON raw results
  artifacts/
    ir/                          # checked-in before/after representative dumps
    ptx/                         # one representative generated PTX
```

Keep `tinytensorc` (compile/dump IR/emit PTX) separate from `tinytensor-run` (allocate, launch, compare) if practical. Do not check in huge binaries or environment-specific build directories.

## 14-day execution plan

Each day ends with a commit, a short note of what is implemented, exact commands run, known breakage, and the next smallest task. Do not label a feature complete because code exists; label it complete only when its listed gate passes.

| Day | Build focus | Deliverable and gate |
| --- | --- | --- |
| 1 | Bootstrap CMake, compiler warnings, test runner; pin/discover LLVM/CUDA versions. | `ctest` runs one smoke test; README records environment discovery and reproducible setup. |
| 2 | Lexer and recursive-descent parser for declarations, assignments, calls, precedence. | Parse valid simple programs and reject three malformed cases with line/column diagnostics. |
| 3 | Typed Tensor IR, printer, verifier, CPU interpreter for elementwise expressions. | Example prints deterministic IR; verifier rejects invalid use/type; interpreter tests pass. Freeze DSL. |
| 4 | DCE and baseline per-op elementwise execution/lowering plan. | DCE removes an unreachable computation and preserves outputs; optimized vs unoptimized interpreter agrees. |
| 5 | Elementwise fusion representation, legality check, fused CPU execution. | Legal chain fuses; shared-producer negative case does not; numerical differential tests pass. |
| 6 | First LLVM lesson: emit PTX for a trivial vector kernel, no custom DSL integration yet. | PTX is emitted and inspected; document target triple/CPU/features actually used. |
| 7 | CUDA Driver API smoke test: load PTX and run a vector add. | GPU result matches CPU for several non-block-size lengths. This is the critical backend checkpoint. |
| 8 | Connect fused elementwise Tensor IR lowering to LLVM/PTX/runtime. | `relu(a*b+c)` runs end-to-end and matches CPU; retain IR and PTX artifact. |
| 9 | Naïve matmul lowering plus CPU matmul. | Small rectangular and edge shapes match CPU; bounds behavior tested. |
| 10 | Tiled `16×16` shared-memory matmul. | Sizes not divisible by 16 pass correctness; no out-of-bounds behavior in guarded code review/tests. |
| 11 | Bias + ReLU epilogue fusion after matmul. | Fused and separate reference variants agree; IR makes fusion choice visible. |
| 12 | Benchmark harness with warmups, repeated timed runs, median and raw CSV/JSON. | Elementwise fused/unfused and naïve/tiled tables generated on the actual GPU. |
| 13 | Documentation, tests, artifacts, clean-environment build rehearsal. | `ctest` plus at least one GPU E2E command pass from documented steps. |
| 14 | Demo rehearsal and interview prep. | Five-minute demo works, project claims are audited, résumé bullet and talk track finalized. |

## Test gates and benchmark discipline

### Required test layers

1. **Frontend unit tests:** tokenization, precedence, calls, declarations, source diagnostics.
2. **IR/verifier tests:** types, uses, output existence, invalid shapes, invalid broadcast, redefinition.
3. **Pass tests:** expected IR snapshots for DCE and fusion; legal and illegal fusion cases.
4. **Differential tests:** CPU unoptimized vs CPU optimized; then GPU vs CPU over fixed seeded random inputs.
5. **GPU E2E tests:** elementwise lengths including non-multiples of block size; matmul shapes including non-multiples of 16 and rectangular cases.
6. **Negative tests:** invalid DSL should fail clearly, never crash.

Use maximum absolute error (and optionally relative error) with a stated threshold. Fix the seed and retain failing shapes/seed in the test output. GPU tests may be conditionally skipped if no GPU exists, but CI/readme must say that clearly.

For every benchmark: warm up, time only the operation intended for comparison, use CUDA events, run enough repetitions to report a median, record shape, block/tile size, GPU model, CUDA/driver/LLVM versions, and raw samples. Avoid claiming a speedup when timings are within noise. Explain results in terms of global-memory traffic, coalescing, reuse, occupancy/register pressure, and launch overhead—not vague “GPU optimization.”

## Scope control: what may be cut

The must-have vertical slice is: DSL (or, as last resort, a programmatic builder) → Tensor IR + verifier → CPU oracle → one fused elementwise GPU program → naïve/tiled matmul GPU programs → measured, reproducible demo.

Cut in this order if behind schedule:

1. Replace the text parser with a small C++ program builder **only if** parser work blocks end-to-end GPU execution; disclose it prominently.
2. Drop `exp`, division, `max`, and algebraic simplification.
3. Drop CSE and constant folding.
4. Keep only one clean matmul tile size (`16×16`) and fixed f32 static shapes.
5. Drop optional ORC CPU JIT, LLVM optimization comparisons, fuzzing, Nsight Compute, cuBLAS context comparison, and broad CI.

Never sacrifice CPU-vs-GPU correctness, explicit fusion behavior, tiled-vs-naïve comparison, or transparent documentation. If the LLVM C++ NVPTX route itself becomes blocked after a time-boxed investigation, a CUDA C++ prototype can validate kernel logic—but it is not a substitute for the stated backend and must be described as such, not passed off as LLVM code generation.

## Risks and mitigations

| Risk | Practical response |
| --- | --- |
| LLVM/CUDA version friction | Use prebuilt LLVM packages when available; write a tiny PTX emitter before integrating the compiler. Record exact versions. |
| Colab GPU unavailable/changes | Develop parser/IR/passes/CPU tests locally; keep GPU commands notebook-friendly and rerun benchmark only when allocated a GPU. |
| NVVM metadata/intrinsics unclear | Derive patterns from a tiny CUDA kernel compiled to LLVM IR; reduce to the smallest working module. |
| Driver API errors | Add a `CUDA_CHECK` helper that includes error name/string and call site; test vector add before the full compiler. |
| Numerical mismatch | First compare tiny deterministic tensors; print first differing index/value; separately test indexing, broadcast, and edge tiles. |
| Performance disappointment | Report it honestly; verify timing scope and median; show memory-traffic reasoning and limitations rather than tune blindly. |
| Scope creep | No new syntax/ops after Day 3; daily ship gate decides whether optional work is allowed. |

## Interview preparation

Be able to draw this pipeline without notes:

```text
DSL source → AST → typed Tensor IR → verifier/pass manager → fusion groups
          → kernel lowering → LLVM IR/NVPTX → PTX → CUDA Driver API launch
          → GPU output ↔ CPU interpreter oracle → benchmark report
```

Practice answers for these questions:

- Why are chains of simple ML operations often memory-bound, and how does fusion help?
- What makes your representation SSA-like? What exactly does your verifier prove—and what does it not prove?
- State your fusion legality rule and give an illegal shared-producer counterexample.
- Map a thread/block to output elements in naïve and tiled matmul. Why do shared-memory tiles reduce repeated global loads?
- Why are barriers required? How are edge tiles guarded?
- How does LLVM IR reach PTX and then actual GPU machine code? What does the CUDA Driver API load?
- How did you validate correctness and avoid benchmark traps?
- What would you add with more time: richer shape system, schedule/tile IR, vectorized loads, autotuning, occupancy analysis, WMMA/tensor cores, better alias/effect analysis?

Avoid claiming “full SSA dominance” without a CFG, “faster than cuBLAS,” general-purpose compilation, or production-grade optimization. Being exact about limitations signals stronger systems judgment than inflated scope.

### Résumé bullet template (use only after verified)

> Built TinyTensor, a C++17 tensor-DSL compiler targeting NVIDIA PTX via LLVM NVPTX and the CUDA Driver API; implemented typed SSA-style IR verification, legal elementwise/epilogue fusion, and shared-memory tiled matmul, validated against a CPU interpreter and benchmarked on [GPU model].

Replace bracketed facts only with measured facts. Add a quantified speedup only with a reproducible benchmark record.

## Definition of done

- A fresh reader can build and run the documented CPU tests and, on NVIDIA hardware, a GPU smoke/E2E demo.
- `README.md` includes architecture, setup, example DSL, known limitations, and an honest benchmark table.
- `docs/design.md` explains IR invariants, passes, fusion legality, indexing, and tiled matmul mapping.
- Repository contains representative IR before/after dumps and generated PTX for an example.
- Results distinguish **implemented and measured** from **implemented but not benchmarked** and **planned**.
- You can demo the primary program in five minutes and answer why every layer exists.

## Status ledger — keep this current

Use this exact vocabulary everywhere in the README, design doc, and interview notes:

| Feature | Status | Evidence |
| --- | --- | --- |
| Parser | Planned / Implemented / Tested | test target, example, or commit link |
| Tensor IR + verifier | Planned / Implemented / Tested | invariant tests / dump |
| Fusion | Planned / Implemented / Tested / Benchmarked | before/after IR, differential test, table |
| LLVM→PTX | Planned / Implemented / Tested | PTX artifact and target details |
| CUDA launch | Planned / Implemented / Tested | vector/add or E2E command |
| Naïve matmul | Planned / Implemented / Tested / Benchmarked | correctness and timing row |
| Tiled matmul | Planned / Implemented / Tested / Benchmarked | correctness and timing row |

Do not write “implemented” for pseudocode, unrun branches, or code that compiles but has not been compared to the CPU oracle. Do not write “benchmarked” without the raw environment and timing data. If only a CUDA prototype works, say that it is a prototype and explain the gap to LLVM lowering.

## Mentor-LLM starter prompt

Copy/paste the following with this file attached or included:

```text
You are my day-to-day technical mentor for TinyTensor, a 14-day C++17 mini compiler project for an NVIDIA system software/compiler intern interview. Treat the attached TinyTensor handover as the source of truth.

My background: I know compiler theory but am new to the LLVM C++ API. I have intermittent access to an NVIDIA cloud GPU through Google Colab. Optimize for a small, working, honestly documented end-to-end implementation—not breadth or impressive-sounding features.

For each response:
1. First ask for or infer my current day, current repository state, exact environment versions, most recent command/test output, and the smallest blocker if unknown.
2. Give one concrete next milestone with acceptance criteria, then a short ordered implementation plan. Do not propose a large rewrite unless evidence makes it necessary.
3. When giving code, keep it minimal and compilable, name the file paths, explain LLVM/NVVM concepts briefly, and include how to test it.
4. Prefer a vertical slice: parser/IR/CPU oracle → trivial PTX → Driver API vector add → fused elementwise → naïve matmul → tiled matmul → benchmarks.
5. Time-box unfamiliar LLVM/NVPTX issues. Recommend inspecting a minimal CUDA-generated device LLVM IR instead of guessing intrinsic or metadata spellings.
6. Require correctness tests before performance claims. For GPU issues, help isolate with tiny deterministic inputs and readable error checks.
7. Maintain a status ledger using Planned, Implemented, Tested, and Benchmarked. Explicitly flag any discrepancy between the project specification and what truly works.
8. Keep the scope-cut order in the handover. Never encourage fabricated results or claims.

Start by asking me for my current Day 1 repository state, CMake/LLVM/CUDA versions, and whether I have a GPU session today. Then give me the smallest useful first task.
```

## First prompt to run now

If starting immediately, send the mentor prompt above and add: “Today is Day 1. I have an empty repository. Help me create the smallest CMake C++17 skeleton, discover the installed LLVM/CUDA/NVIDIA GPU versions without assuming them, and add a smoke test. Do not begin the parser yet.”
