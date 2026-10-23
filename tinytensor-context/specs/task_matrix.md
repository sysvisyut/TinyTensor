# Sequential implementation task matrix

Complete tasks in order. Each task ends with a checkpoint; do not silently advance when it fails. Suggested layout: `include/tinytensor/`, `src/{frontend,ir,passes,backend,runtime}/`, `tests/`, `examples/`, `bench/`.

| # | Files to create/change | Deliverable and acceptance tests |
|---|---|---|
| 0 | `CMakeLists.txt`, `README.md`, `tests/CMakeLists.txt` | C++17, warnings, GoogleTest discovery; `ctest --output-on-failure` runs a placeholder test. Record LLVM/CUDA availability; CPU-only mode must configure. |
| 1 | `include/tinytensor/ir.hpp`, `src/ir.cpp`, `tests/ir_test.cpp` | Implement exactly `api_contracts.md`. Build `matmul+add+relu`; assert use counts, generated types, and byte-for-byte canonical printer output. |
| 2 | `include/tinytensor/verifier.hpp`, `src/ir/verifier.cpp`, `tests/verifier_test.cpp` | Verify valid IR and reject dangling/mismatched links, output foreign values, invalid matmul shapes, and invalid broadcast. Error identifies the violated invariant. |
| 3 | `include/tinytensor/pass.hpp`, `src/pass.cpp`, `tests/pass_manager_test.cpp` | Ordered manager returns aggregate changed flag; verification runs after each pass when enabled. A deliberately corrupt test pass is caught. |
| 4 | `include/tinytensor/frontend/{lexer,parser}.hpp`, `src/frontend/{lexer,parser}.cpp`, `tests/parser_test.cpp` | Parse the sample DSL into canonical IR. Test precedence, duplicate names, undefined names, line-number diagnostics, rank/shape errors. If schedule slips, replace this task with a small builder-based example—not the optimization/backend tasks. |
| 5 | `include/tinytensor/passes/{fold,simplify,dce}.hpp`, `src/passes/{fold,simplify,dce}.cpp`, tests | Constant folding and safe identities; DCE removes dead backwards slice only. Test output roots and that live values retain IDs. |
| 6 | `include/tinytensor/passes/cse.hpp`, `src/passes/cse.cpp`, `tests/cse_test.cpp` | Reuse only same op, ordered operands, and equal type; never CSE non-commutative operand reversals. Optimized CPU interpretation agrees with original. |
| 7 | `include/tinytensor/passes/fusion.hpp`, `src/passes/fusion.cpp`, `tests/fusion_test.cpp` | Elementwise and `matmul+bias+relu` grouping. Required negative test: a shared producer with an outside consumer remains unfused. Test output and CPU-reference equivalence. |
| 8 | `include/tinytensor/runtime/cpu_reference.hpp`, `src/runtime/cpu_reference.cpp`, tests | Evaluate every v1 op with broadcast; random small DAG comparison becomes the semantic oracle for every pass. |
| 9 | `include/tinytensor/backend/nvptx_target.hpp`, `src/backend/nvptx_target.cpp`, `tests/nvptx_smoke_test.cpp` | Implement only the index-write smoke test in `nvvm_recipes.md`; verify module and emit PTX. GPU execution is optional/marked skipped without CUDA. Save generated IR/PTX on failure. |
| 10 | `include/tinytensor/runtime/cuda_driver.hpp`, `src/runtime/cuda_driver.cpp`, GPU tests | RAII Driver API wrappers; load PTX, allocate/copy/launch/copy-back. Vector add matches CPU within `1e-3`; every CUDA result is checked. |
| 11 | `src/backend/lowering.cpp`, end-to-end test | Lower unfused elementwise and naive matmul. Test non-square matrices and non-multiple block sizes against CPU. |
| 12 | `src/backend/tiled_matmul.cpp`, GPU tests | Fixed 16×16 AS3 tile algorithm; all threads execute barriers; zero-fill bounds. Test `17×19 × 19×23`, then 256/512. |
| 13 | `bench/benchmark.cpp`, `examples/`, README tables | Median of repeated timed launches, GPU/driver/build metadata, unfused-vs-fused and naive-vs-tiled. Never fill tables with invented numbers. |
| 14 | `docs/design.md`, `docs/interview_notes.md` | Explain SSA/use-def, fusion legality, memory/coalescing/tile mapping, LLVM→PTX→driver JIT, measured results, and limitations. Rehearse explaining any line of the main example. |

## Per-task completion report

Return: (1) files changed, (2) learner explanation, (3) exact tests run and result, (4) known limits/skips, and (5) the next task number. “Implemented” requires tests; “GPU verified” requires an actual GPU launch and comparison.

## 14-day calendar checkpoint

Days 1–3: Tasks 0–4. Days 4–7: Tasks 5–8. Days 8–11: Tasks 9–11. Days 12–14: Tasks 12–14. A correct, explainable subset is stronger interview evidence than an untested broad pipeline.
