#pragma once

#include <memory>
#include <string_view>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
} // namespace llvm

namespace tinytensor::backend {

// Builds a naive matrix multiplication kernel module:
// Kernel signature: void naive_matmul(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %m, i32 %k, i32 %n)
// Computes out[row * n + col] = sum_{p=0}^{k-1} a[row * k + p] * b[p * n + col]
// Directly reading from global memory without shared memory tiling.
std::unique_ptr<llvm::Module> createNaiveMatMulModule(
    llvm::LLVMContext& ctx,
    llvm::TargetMachine* targetMachine = nullptr,
    std::string_view moduleName = "tinytensor_naive_matmul",
    std::string_view kernelName = "naive_matmul");

} // namespace tinytensor::backend
