# LLVM/NVPTX recipe book — pinned, cautious, smoke-test first

## Version policy

Pin one LLVM major version (recommended project baseline: LLVM 17 or 18) and record `llvm-config --version`, CUDA driver version, `nvptx64-nvidia-cuda` target triple, and `sm_XX` CPU. The **concepts/names** below are stable NVPTX conventions; the exact C++ overloads, opaque-pointer behavior, headers, target initialization, and available `Intrinsic::ID` enum values are version-dependent. Compile the smoke test before building real lowering. If it disagrees with a local LLVM tree, inspect that tree's `include/llvm/IR/IntrinsicsNVVM.td` and generated `IntrinsicEnums.inc`, then adapt only the small compatibility wrapper.

Authoritative background: LLVM's [NVPTX backend guide](https://llvm.org/docs/NVPTXUsage.html) maps shared memory to address space 3, `tid.x`/`ctaid.x`/`ntid.x` special registers, and `llvm.nvvm.barrier0` to `__syncthreads()`.

## Minimal smoke test target

First emit a module containing a void `tiny_smoke(ptr addrspace(1) %out)` kernel that writes `threadIdx.x + blockIdx.x * blockDim.x` to `%out`, verifies with `llvm::verifyModule`, emits PTX, loads it via `cuModuleLoadData`, launches one block, and checks results on host. Do not start tiled matmul until this works.

Required module settings:

```cpp
llvm::LLVMContext C;
llvm::Module M("tinytensor", C);
M.setTargetTriple("nvptx64-nvidia-cuda");
// Set M data layout from the TargetMachine you actually constructed.
```

Target initialization/TargetMachine construction changes enough across releases that it belongs in `src/backend/nvptx_target.cpp`, not copied through the project. Check every `TargetRegistry::lookupTarget` error and every `TargetMachine*` for null.

## Kernel and `nvvm.annotations`

The emitted LLVM IR contract is `!nvvm.annotations = !{!0}` and `!0 = !{ptr @kernel, !"kernel", i32 1}`. In typed-pointer LLVM, the first field prints as the function's pointer type. This C++ metadata construction is broadly usable with LLVM 17/18:

```cpp
auto* voidTy = llvm::Type::getVoidTy(C);
auto* i8PtrGlobal = llvm::PointerType::get(C, /*AddressSpace=*/1);
auto* fnTy = llvm::FunctionType::get(voidTy, {i8PtrGlobal}, false);
auto* kernel = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage,
                                      "tiny_smoke", M);
auto* one = llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 1);
auto* md = llvm::MDNode::get(C, {llvm::ValueAsMetadata::get(kernel),
                                 llvm::MDString::get(C, "kernel"),
                                 llvm::ConstantAsMetadata::get(one)});
M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
```

This marks a PTX kernel for normal NVPTX lowering. It does not validate ABI argument layout or make a host launch correct; launch with CUDA Driver API parameters matching the generated function signature.

## Special-register reads

Prefer `llvm::Intrinsic::getDeclaration(&M, ID)` when the local LLVM exposes the desired IDs. The *IR names and result type* are the reliable cross-check:

```text
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
```

In an LLVM release where these enum identifiers exist, the call pattern is:

```cpp
auto* tidFn = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
llvm::Value* tid = B.CreateCall(tidFn, {}, "tid.x");
```

Use the analogous `nvvm_read_ptx_sreg_ctaid_x` and `nvvm_read_ptx_sreg_ntid_x` IDs only after compiling against the selected headers. Index formula: `global = ctaid_x * ntid_x + tid_x`; use `CreateMul` and `CreateAdd`. If the enum spelling/availability differs, use `M.getOrInsertFunction` with the exact IR name and `i32 ()` `FunctionType`, then verify the resulting module; do not guess intrinsic declarations.

## Shared memory and CTA barrier

NVPTX shared memory is address space **3**; global device pointers are normally AS **1**. A fixed shared tile is a module-scope global, not an entry-block `alloca`:

```cpp
auto* f32 = llvm::Type::getFloatTy(C);
auto* tileTy = llvm::ArrayType::get(f32, 16 * 16);
auto* shared = new llvm::GlobalVariable(
    M, tileTy, /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
    llvm::UndefValue::get(tileTy), "tileA", nullptr,
    llvm::GlobalVariable::NotThreadLocal, /*AddressSpace=*/3);
```

Dynamic shared memory is a separate later milestone: it needs an external AS3 declaration and a matching dynamic shared-memory byte count in `cuLaunchKernel`. Do not conflate it with fixed-size tiles.

The v1 full-block synchronization intrinsic has IR name `llvm.nvvm.barrier0()` and semantics equivalent to `__syncthreads()`. When the local enum `Intrinsic::nvvm_barrier0` is available:

```cpp
auto* barrier = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::nvvm_barrier0);
B.CreateCall(barrier);
```

All non-exited threads in a block must execute the barrier consistently. Never put it only on one side of a per-thread bounds branch. For edge tiles, guard global loads/stores but let all threads reach barriers; load zero for out-of-range elements.

## Smoke-test gates and common traps

1. `llvm::verifyModule(M, &llvm::errs())` must pass before PTX emission.
2. Save `.ll` and `.ptx`; inspect that the kernel is visible and special registers/barrier appear as expected.
3. Run a one-block index-write test, then a vector add, then naive matmul, then fixed 16×16 tiled matmul.
4. Test dimensions not divisible by 16 before reporting tiled correctness/performance.

`exp` must normally be lowered through CUDA's compatible libdevice for the selected CUDA/LLVM setup, not assumed to be a native LLVM instruction. PTX emission alone is not proof that driver JIT or symbol resolution will work. LLVM's current guide documents newer `barrier.cta.*` and `mbarrier` families too; they are out of v1 scope—use `barrier0` for the simple tiled kernel.
