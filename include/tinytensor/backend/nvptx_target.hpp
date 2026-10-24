#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <optional>

namespace llvm {
class LLVMContext;
class Module;
class TargetMachine;
class Function;
} // namespace llvm

namespace tinytensor::backend {

struct NVPTXTargetConfig {
  std::string triple = "nvptx64-nvidia-cuda";
  std::string sm = "sm_70";
  std::string features = "";
};

// Initializes the NVPTX target backend in LLVM. Safe to call multiple times.
// Returns true if the NVPTX target is available in this LLVM build.
bool initializeNVPTXTarget();

// Returns true if the NVPTX target is supported by the linked LLVM build.
bool isNVPTXTargetSupported();

// Creates an NVPTX TargetMachine using the specified configuration.
// Returns nullptr if target creation fails.
std::unique_ptr<llvm::TargetMachine> createNVPTXTargetMachine(
    const NVPTXTargetConfig& config = {});

// Emits an LLVM module as a human-readable IR string.
std::string emitLLVMIR(const llvm::Module& module);

// Verifies an LLVM module using llvm::verifyModule.
// Returns true if the module satisfies all LLVM invariants;
// returns false and populates errorMsg (if provided) if verification fails.
bool verifyLLVMModule(const llvm::Module& module, std::string* errorMsg = nullptr);

// Compiles an LLVM Module to PTX assembly text using the given TargetMachine.
// Returns std::nullopt and populates errorMsg (if provided) on failure.
std::optional<std::string> emitPTX(llvm::Module& module,
                                  llvm::TargetMachine& targetMachine,
                                  std::string* errorMsg = nullptr);

// Marks an LLVM Function as an NVPTX kernel by attaching !nvvm.annotations
// and setting the CallingConv::PTX_Kernel calling convention.
void markAsNVPTXKernel(llvm::Module& module, llvm::Function* kernelFn);

// Builds the canonical smoke test module as specified in specs/nvvm_recipes.md:
// Kernel signature: void tiny_smoke(ptr addrspace(1) %out)
// Body: writes global_idx = (blockIdx.x * blockDim.x + threadIdx.x) into %out[global_idx].
std::unique_ptr<llvm::Module> createSmokeTestModule(
    llvm::LLVMContext& ctx,
    llvm::TargetMachine* targetMachine = nullptr,
    std::string_view moduleName = "tinytensor_smoke",
    std::string_view kernelName = "tiny_smoke");

// Helper to save LLVM IR and PTX artifacts to disk on failure or for inspection.
void saveArtifact(std::string_view filepath, std::string_view content);

} // namespace tinytensor::backend
