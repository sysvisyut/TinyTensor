#pragma once

#include <memory>
#include <string_view>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
} // namespace llvm

namespace tinytensor::backend {

// Builds a 16x16 tiled matrix multiplication kernel module:
// Kernel signature: void tiled_matmul(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %m, i32 %k, i32 %n)
// Computes out[row * n + col] = sum_{p=0}^{k-1} a[row * k + p] * b[p * n + col]
// Using 16x16 shared memory tiles in address space 3, full-block barriers, and edge zero-fill.
std::unique_ptr<llvm::Module> createTiledMatMulModule(
    llvm::LLVMContext& ctx,
    llvm::TargetMachine* targetMachine = nullptr,
    std::string_view moduleName = "tinytensor_tiled_matmul",
    std::string_view kernelName = "tiled_matmul");

} // namespace tinytensor::backend
