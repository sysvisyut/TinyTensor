# TinyTensor — agent context pack

This folder is the **source of truth for implementation decisions** for the 14-day TinyTensor project: a C++17 mini compiler for a static tensor DSL, custom SSA Tensor IR, optimization passes, LLVM NVPTX/PTX emission, and CUDA Driver API execution.

## Agent operating rules

1. Read `specs/api_contracts.md`, then `specs/ir_schema.md`, then the relevant task in `specs/task_matrix.md` before editing code.
2. Implement **one numbered task at a time**. Do not invent a second architecture while implementing a task.
3. Preserve public contracts and the canonical IR printer exactly. If a contract must change, first update this pack and its tests in the same change with an explicit rationale.
4. Run the task's acceptance tests. A build succeeding is not proof that behavior is correct.
5. Before writing a nontrivial component, explain to the user: what problem it solves, its input/output invariants, and one concrete example. Keep the explanation short enough for a learner to follow.
6. Never claim a backend, PTX, GPU result, performance number, or test is working unless it was actually built/run in the current environment. Say `not verified here` when appropriate.
7. For LLVM/NVPTX, pin and report the LLVM version, target triple, and GPU architecture. Follow `specs/nvvm_recipes.md`; API names can differ between LLVM releases.

## Product boundary

- **Inputs:** rank-1/rank-2 statically shaped `f32` tensors; scalar `f32` literals.
- **Ops:** `+ - * / max`, `relu`, `exp`, `neg`, `matmul` and scalar/rank-1 broadcast permitted by the type rules.
- **Outputs:** numerical agreement with a CPU reference, tolerance `1e-3` for GPU tests.
- **Required differentiators:** correct DCE and legal fusion; naive and tiled matmul; benchmark comparison.
- **Non-goals:** autodiff, dynamic shapes, full framework coverage, or cuBLAS-level performance.

## Suggested first prompt

> Read `README.md` and all files in `specs/`. Teach me the idea behind Task 1 in plain language, then implement only Task 1 from `specs/task_matrix.md`. Preserve the API and printer contracts exactly. Run its listed tests and report commands/results honestly; do not claim anything unverified.

## 14-day execution shape

| Days | Outcome |
|---|---|
| 1–3 | CMake/toolchain, frontend, Tensor IR, printer, verifier |
| 4–7 | folding, DCE, CSE, fusion with negative legality tests |
| 8–11 | LLVM module/PTX smoke test, runtime launch, CPU-reference agreement |
| 12–14 | tiled matmul, benchmark table, documentation, interview rehearsal |

If the deadline tightens: replace the parser with a builder API first, then cut algebraic simplification, then CSE. Do **not** cut fusion, tiled matmul, or honest benchmarks.
