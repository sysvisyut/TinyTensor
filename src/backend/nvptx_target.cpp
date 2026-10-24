#include "tinytensor/backend/nvptx_target.hpp"

#include <fstream>
#include <mutex>
#include <iostream>

#if defined(TINYTENSOR_HAS_LLVM)

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

namespace tinytensor::backend {

namespace {

std::once_flag g_init_nvptx_flag;

void doInitNVPTX() {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

} // namespace

bool initializeNVPTXTarget() {
  std::call_once(g_init_nvptx_flag, doInitNVPTX);
  return isNVPTXTargetSupported();
}

bool isNVPTXTargetSupported() {
  std::call_once(g_init_nvptx_flag, doInitNVPTX);
  std::string err;
  llvm::Triple triple("nvptx64-nvidia-cuda");
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, err);
  return target != nullptr;
}

std::unique_ptr<llvm::TargetMachine> createNVPTXTargetMachine(
    const NVPTXTargetConfig& config) {
  if (!initializeNVPTXTarget()) {
    return nullptr;
  }

  std::string err;
  llvm::Triple triple(config.triple);
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target) {
    return nullptr;
  }

  llvm::TargetOptions opt;
  // Create TargetMachine for NVPTX (e.g. sm_70 by default)
  std::unique_ptr<llvm::TargetMachine> tm(
      target->createTargetMachine(triple, config.sm, config.features, opt, std::nullopt));
  return tm;
}

std::string emitLLVMIR(const llvm::Module& module) {
  std::string irStr;
  llvm::raw_string_ostream os(irStr);
  module.print(os, nullptr);
  return irStr;
}

bool verifyLLVMModule(const llvm::Module& module, std::string* errorMsg) {
  std::string diag;
  llvm::raw_string_ostream os(diag);
  // llvm::verifyModule returns true if broken, false if valid
  bool broken = llvm::verifyModule(module, &os);
  if (broken) {
    if (errorMsg) {
      *errorMsg = diag;
    }
    return false;
  }
  return true;
}

std::optional<std::string> emitPTX(llvm::Module& module,
                                  llvm::TargetMachine& targetMachine,
                                  std::string* errorMsg) {
  // First ensure module is valid
  if (!verifyLLVMModule(module, errorMsg)) {
    return std::nullopt;
  }

  // Ensure module matches TargetMachine data layout
  module.setDataLayout(targetMachine.createDataLayout());

  llvm::legacy::PassManager pm;
  llvm::SmallVector<char, 0> ptxBuffer;
  llvm::raw_svector_ostream ptxStream(ptxBuffer);

  if (targetMachine.addPassesToEmitFile(
          pm, ptxStream, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
    if (errorMsg) {
      *errorMsg = "TargetMachine does not support AssemblyFile emission";
    }
    return std::nullopt;
  }

  pm.run(module);
  return std::string(ptxBuffer.data(), ptxBuffer.size());
}

void markAsNVPTXKernel(llvm::Module& module, llvm::Function* kernelFn) {
  if (!kernelFn) {
    return;
  }
  // Set calling convention to PTX_Kernel so LLVM emits .visible .entry
  kernelFn->setCallingConv(llvm::CallingConv::PTX_Kernel);

  // Set !nvvm.annotations = !{!0} where !0 = !{ptr @kernelFn, !"kernel", i32 1}
  llvm::LLVMContext& ctx = module.getContext();
  auto* one = llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1);
  auto* md = llvm::MDNode::get(ctx, {
      llvm::ValueAsMetadata::get(kernelFn),
      llvm::MDString::get(ctx, "kernel"),
      llvm::ConstantAsMetadata::get(one)
  });
  module.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md);
}

std::unique_ptr<llvm::Module> createSmokeTestModule(
    llvm::LLVMContext& ctx,
    llvm::TargetMachine* targetMachine,
    std::string_view moduleName,
    std::string_view kernelName) {
  auto mod = std::make_unique<llvm::Module>(moduleName, ctx);
  llvm::Triple triple("nvptx64-nvidia-cuda");
  mod->setTargetTriple(triple);

  if (targetMachine) {
    mod->setDataLayout(targetMachine->createDataLayout());
  }

  auto* voidTy = llvm::Type::getVoidTy(ctx);
  auto* i32Ty = llvm::Type::getInt32Ty(ctx);
  // Device pointer in global memory (AddressSpace = 1)
  auto* globalPtrTy = llvm::PointerType::get(ctx, /*AddressSpace=*/1);
  auto* fnTy = llvm::FunctionType::get(voidTy, {globalPtrTy}, false);

  auto* kernel = llvm::Function::Create(
      fnTy, llvm::Function::ExternalLinkage, kernelName, *mod);
  markAsNVPTXKernel(*mod, kernel);

  auto* entryBB = llvm::BasicBlock::Create(ctx, "entry", kernel);
  llvm::IRBuilder<> builder(entryBB);

  // Read thread special registers:
  //   tid.x:   llvm.nvvm.read.ptx.sreg.tid.x
  //   ctaid.x: llvm.nvvm.read.ptx.sreg.ctaid.x
  //   ntid.x:  llvm.nvvm.read.ptx.sreg.ntid.x
  auto* tidFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  auto* ctaidFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  auto* ntidFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  llvm::Value* tid = builder.CreateCall(tidFn, {}, "tid.x");
  llvm::Value* ctaid = builder.CreateCall(ctaidFn, {}, "ctaid.x");
  llvm::Value* ntid = builder.CreateCall(ntidFn, {}, "ntid.x");

  // global_idx = ctaid.x * ntid.x + tid.x
  llvm::Value* blkOffset = builder.CreateMul(ctaid, ntid, "blk_offset");
  llvm::Value* globalIdx = builder.CreateAdd(blkOffset, tid, "global_idx");

  // Store global_idx into %out[global_idx]
  llvm::Value* outPtr = kernel->getArg(0);
  llvm::Value* elemPtr = builder.CreateGEP(i32Ty, outPtr, globalIdx, "elem_ptr");
  builder.CreateStore(globalIdx, elemPtr);
  builder.CreateRetVoid();

  return mod;
}

void saveArtifact(std::string_view filepath, std::string_view content) {
  std::ofstream outFile{std::string(filepath)};
  if (outFile.is_open()) {
    outFile.write(content.data(), static_cast<std::streamsize>(content.size()));
  }
}

} // namespace tinytensor::backend

#else // !defined(TINYTENSOR_HAS_LLVM)

namespace tinytensor::backend {

bool initializeNVPTXTarget() { return false; }
bool isNVPTXTargetSupported() { return false; }

std::unique_ptr<llvm::TargetMachine> createNVPTXTargetMachine(
    const NVPTXTargetConfig&) {
  return nullptr;
}

std::string emitLLVMIR(const llvm::Module&) { return ""; }

bool verifyLLVMModule(const llvm::Module&, std::string* errorMsg) {
  if (errorMsg) *errorMsg = "LLVM support not compiled in";
  return false;
}

std::optional<std::string> emitPTX(llvm::Module&,
                                  llvm::TargetMachine&,
                                  std::string* errorMsg) {
  if (errorMsg) *errorMsg = "LLVM support not compiled in";
  return std::nullopt;
}

void markAsNVPTXKernel(llvm::Module&, llvm::Function*) {}

std::unique_ptr<llvm::Module> createSmokeTestModule(
    llvm::LLVMContext&,
    llvm::TargetMachine*,
    std::string_view,
    std::string_view) {
  return nullptr;
}

void saveArtifact(std::string_view filepath, std::string_view content) {
  std::ofstream outFile{std::string(filepath)};
  if (outFile.is_open()) {
    outFile.write(content.data(), static_cast<std::streamsize>(content.size()));
  }
}

} // namespace tinytensor::backend

#endif // TINYTENSOR_HAS_LLVM
