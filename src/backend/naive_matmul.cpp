#include "tinytensor/backend/naive_matmul.hpp"
#include "tinytensor/backend/nvptx_target.hpp"

#if defined(TINYTENSOR_HAS_LLVM)

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>

namespace tinytensor::backend {

std::unique_ptr<llvm::Module> createNaiveMatMulModule(
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
  auto* f32Ty = llvm::Type::getFloatTy(ctx);
  auto* i32Ty = llvm::Type::getInt32Ty(ctx);
  auto* i64Ty = llvm::Type::getInt64Ty(ctx);
  auto* globalPtrTy = llvm::PointerType::get(ctx, /*AddressSpace=*/1);

  // Kernel signature:
  // void naive_matmul(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b,
  //                   i32 %m, i32 %k, i32 %n)
  auto* fnTy = llvm::FunctionType::get(
      voidTy, {globalPtrTy, globalPtrTy, globalPtrTy, i32Ty, i32Ty, i32Ty}, false);

  auto* kernel = llvm::Function::Create(
      fnTy, llvm::Function::ExternalLinkage, kernelName, *mod);
  markAsNVPTXKernel(*mod, kernel);

  // Intrinsics for special-register reads
  auto* tidXFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  auto* tidYFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y);
  auto* ctaidXFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  auto* ctaidYFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  auto* ntidXFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  auto* ntidYFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y);

  // Basic blocks
  auto* entryBB = llvm::BasicBlock::Create(ctx, "entry", kernel);
  auto* kInitBB = llvm::BasicBlock::Create(ctx, "k_init", kernel);
  auto* loopHeaderBB = llvm::BasicBlock::Create(ctx, "loop_header", kernel);
  auto* writeAccumBB = llvm::BasicBlock::Create(ctx, "write_accum", kernel);
  auto* writeZeroBB = llvm::BasicBlock::Create(ctx, "write_zero", kernel);
  auto* storeBB = llvm::BasicBlock::Create(ctx, "store_c", kernel);
  auto* exitBB = llvm::BasicBlock::Create(ctx, "exit", kernel);

  llvm::IRBuilder<> builder(entryBB);

  llvm::Value* outPtr = kernel->getArg(0);
  llvm::Value* aPtr   = kernel->getArg(1);
  llvm::Value* bPtr   = kernel->getArg(2);
  llvm::Value* mArg   = kernel->getArg(3);
  llvm::Value* kArg   = kernel->getArg(4);
  llvm::Value* nArg   = kernel->getArg(5);

  llvm::Value* tx = builder.CreateCall(tidXFn, {}, "tx");
  llvm::Value* ty = builder.CreateCall(tidYFn, {}, "ty");
  llvm::Value* bx = builder.CreateCall(ctaidXFn, {}, "bx");
  llvm::Value* by = builder.CreateCall(ctaidYFn, {}, "by");
  llvm::Value* bdimx = builder.CreateCall(ntidXFn, {}, "bdimx");
  llvm::Value* bdimy = builder.CreateCall(ntidYFn, {}, "bdimy");

  // row = by * bdimy + ty, col = bx * bdimx + tx
  llvm::Value* row = builder.CreateAdd(builder.CreateMul(by, bdimy), ty, "row");
  llvm::Value* col = builder.CreateAdd(builder.CreateMul(bx, bdimx), tx, "col");

  llvm::Value* rowInBounds = builder.CreateICmpSLT(row, mArg, "row_in_bounds");
  llvm::Value* colInBounds = builder.CreateICmpSLT(col, nArg, "col_in_bounds");
  llvm::Value* inBounds = builder.CreateAnd(rowInBounds, colInBounds, "in_bounds");
  builder.CreateCondBr(inBounds, kInitBB, exitBB);

  // k_init block
  builder.SetInsertPoint(kInitBB);
  llvm::Value* hasK = builder.CreateICmpSGT(kArg, builder.getInt32(0), "has_k");
  builder.CreateCondBr(hasK, loopHeaderBB, writeZeroBB);

  // loop_header block: straight-line reduction loop directly from global memory
  builder.SetInsertPoint(loopHeaderBB);
  auto* kPhi = builder.CreatePHI(i32Ty, 2, "k_idx");
  kPhi->addIncoming(builder.getInt32(0), kInitBB);
  auto* accumPhi = builder.CreatePHI(f32Ty, 2, "accum");
  accumPhi->addIncoming(llvm::ConstantFP::get(f32Ty, 0.0), kInitBB);

  llvm::Value* row64 = builder.CreateSExt(row, i64Ty);
  llvm::Value* k64 = builder.CreateSExt(kArg, i64Ty);
  llvm::Value* kIdx64 = builder.CreateSExt(kPhi, i64Ty);
  llvm::Value* col64 = builder.CreateSExt(col, i64Ty);
  llvm::Value* n64 = builder.CreateSExt(nArg, i64Ty);

  llvm::Value* aOffset = builder.CreateAdd(builder.CreateMul(row64, k64), kIdx64, "a_offset");
  llvm::Value* aElemPtr = builder.CreateGEP(f32Ty, aPtr, aOffset, "a_elem_ptr");
  llvm::Value* aVal = builder.CreateLoad(f32Ty, aElemPtr, "a_val");

  llvm::Value* bOffset = builder.CreateAdd(builder.CreateMul(kIdx64, n64), col64, "b_offset");
  llvm::Value* bElemPtr = builder.CreateGEP(f32Ty, bPtr, bOffset, "b_elem_ptr");
  llvm::Value* bVal = builder.CreateLoad(f32Ty, bElemPtr, "b_val");

  llvm::Value* prod = builder.CreateFMul(aVal, bVal, "prod");
  llvm::Value* nextAccum = builder.CreateFAdd(accumPhi, prod, "next_accum");

  llvm::Value* nextK = builder.CreateAdd(kPhi, builder.getInt32(1), "next_k");
  kPhi->addIncoming(nextK, loopHeaderBB);
  accumPhi->addIncoming(nextAccum, loopHeaderBB);

  llvm::Value* loopDone = builder.CreateICmpSGE(nextK, kArg, "loop_done");
  builder.CreateCondBr(loopDone, writeAccumBB, loopHeaderBB);

  // write_accum block
  builder.SetInsertPoint(writeAccumBB);
  builder.CreateBr(storeBB);

  // write_zero block
  builder.SetInsertPoint(writeZeroBB);
  builder.CreateBr(storeBB);

  // store_c block
  builder.SetInsertPoint(storeBB);
  auto* finalRes = builder.CreatePHI(f32Ty, 2, "final_res");
  finalRes->addIncoming(nextAccum, writeAccumBB);
  finalRes->addIncoming(llvm::ConstantFP::get(f32Ty, 0.0), writeZeroBB);

  llvm::Value* cOffset = builder.CreateAdd(builder.CreateMul(row64, n64), col64, "c_offset");
  llvm::Value* cElemPtr = builder.CreateGEP(f32Ty, outPtr, cOffset, "c_elem_ptr");
  builder.CreateStore(finalRes, cElemPtr);
  builder.CreateBr(exitBB);

  // exit block
  builder.SetInsertPoint(exitBB);
  builder.CreateRetVoid();

  return mod;
}

} // namespace tinytensor::backend

#else // !defined(TINYTENSOR_HAS_LLVM)

namespace tinytensor::backend {

std::unique_ptr<llvm::Module> createNaiveMatMulModule(
    llvm::LLVMContext&,
    llvm::TargetMachine*,
    std::string_view,
    std::string_view) {
  return nullptr;
}

} // namespace tinytensor::backend

#endif // TINYTENSOR_HAS_LLVM
