#include "tinytensor/backend/tiled_matmul.hpp"
#include "tinytensor/backend/nvptx_target.hpp"

#if defined(TINYTENSOR_HAS_LLVM)

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>

namespace tinytensor::backend {

namespace {

// Emits a full-block synchronization barrier (equivalent to __syncthreads() / bar.sync 0).
// In LLVM 20+, nvvm_barrier0 was migrated to nvvm_barrier_cta_sync_aligned_all(0).
void emitBlockBarrier(llvm::Module& mod, llvm::IRBuilder<>& builder) {
#if defined(LLVM_VERSION_MAJOR) && LLVM_VERSION_MAJOR >= 20
  auto* barrierFn = llvm::Intrinsic::getOrInsertDeclaration(
      &mod, llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all);
  builder.CreateCall(barrierFn, {builder.getInt32(0)});
#else
  auto* barrierFn = llvm::Intrinsic::getOrInsertDeclaration(
      &mod, llvm::Intrinsic::nvvm_barrier0);
  builder.CreateCall(barrierFn, {});
#endif
}

} // namespace

std::unique_ptr<llvm::Module> createTiledMatMulModule(
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
  // void tiled_matmul(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b,
  //                   i32 %m, i32 %k, i32 %n)
  auto* fnTy = llvm::FunctionType::get(
      voidTy, {globalPtrTy, globalPtrTy, globalPtrTy, i32Ty, i32Ty, i32Ty}, false);

  auto* kernel = llvm::Function::Create(
      fnTy, llvm::Function::ExternalLinkage, kernelName, *mod);
  markAsNVPTXKernel(*mod, kernel);

  // Module-scope shared memory tiles: 16x16 float elements in AddressSpace=3
  constexpr unsigned kTileDim = 16;
  constexpr unsigned kTileElements = kTileDim * kTileDim; // 256
  auto* tileTy = llvm::ArrayType::get(f32Ty, kTileElements);

  auto* sharedA = new llvm::GlobalVariable(
      *mod, tileTy, /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
      llvm::UndefValue::get(tileTy), "tileA", nullptr,
      llvm::GlobalVariable::NotThreadLocal, /*AddressSpace=*/3);
  sharedA->setAlignment(llvm::Align(16));

  auto* sharedB = new llvm::GlobalVariable(
      *mod, tileTy, /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
      llvm::UndefValue::get(tileTy), "tileB", nullptr,
      llvm::GlobalVariable::NotThreadLocal, /*AddressSpace=*/3);
  sharedB->setAlignment(llvm::Align(16));

  // Intrinsics for special-register reads
  auto* tidXFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
  auto* tidYFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y);
  auto* ctaidXFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  auto* ctaidYFn = llvm::Intrinsic::getOrInsertDeclaration(
      mod.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y);

  // Basic blocks
  auto* entryBB = llvm::BasicBlock::Create(ctx, "entry", kernel);
  auto* loopHeaderBB = llvm::BasicBlock::Create(ctx, "loop_header", kernel);
  auto* loadABB = llvm::BasicBlock::Create(ctx, "load_a", kernel);
  auto* zeroABB = llvm::BasicBlock::Create(ctx, "zero_a", kernel);
  auto* joinABB = llvm::BasicBlock::Create(ctx, "join_a", kernel);
  auto* loadBBB = llvm::BasicBlock::Create(ctx, "load_b", kernel);
  auto* zeroBBB = llvm::BasicBlock::Create(ctx, "zero_b", kernel);
  auto* joinBBB = llvm::BasicBlock::Create(ctx, "join_b", kernel);
  auto* afterLoopBB = llvm::BasicBlock::Create(ctx, "after_loop", kernel);
  auto* storeCBB = llvm::BasicBlock::Create(ctx, "store_c", kernel);
  auto* exitBB = llvm::BasicBlock::Create(ctx, "exit", kernel);

  llvm::IRBuilder<> builder(entryBB);

  llvm::Value* outPtr = kernel->getArg(0);
  llvm::Value* aPtr = kernel->getArg(1);
  llvm::Value* bPtr = kernel->getArg(2);
  llvm::Value* mArg = kernel->getArg(3);
  llvm::Value* kArg = kernel->getArg(4);
  llvm::Value* nArg = kernel->getArg(5);

  llvm::Value* tx = builder.CreateCall(tidXFn, {}, "tx");
  llvm::Value* ty = builder.CreateCall(tidYFn, {}, "ty");
  llvm::Value* bx = builder.CreateCall(ctaidXFn, {}, "bx");
  llvm::Value* by = builder.CreateCall(ctaidYFn, {}, "by");

  llvm::Value* cTileDim = builder.getInt32(kTileDim);
  llvm::Value* row = builder.CreateAdd(builder.CreateMul(by, cTileDim), ty, "row");
  llvm::Value* col = builder.CreateAdd(builder.CreateMul(bx, cTileDim), tx, "col");
  llvm::Value* localTileIdx = builder.CreateAdd(
      builder.CreateMul(ty, cTileDim), tx, "local_tile_idx");

  // num_phases = (K + 15) / 16
  llvm::Value* kPlus15 = builder.CreateAdd(kArg, builder.getInt32(15), "k_plus_15");
  llvm::Value* numPhases = builder.CreateSDiv(kPlus15, cTileDim, "num_phases");

  llvm::Value* hasPhases = builder.CreateICmpSGT(numPhases, builder.getInt32(0), "has_phases");
  builder.CreateCondBr(hasPhases, loopHeaderBB, afterLoopBB);

  // --- Loop Header ---
  builder.SetInsertPoint(loopHeaderBB);
  auto* phasePhi = builder.CreatePHI(i32Ty, 2, "phase_idx");
  phasePhi->addIncoming(builder.getInt32(0), entryBB);
  auto* accumPhi = builder.CreatePHI(f32Ty, 2, "accum");
  accumPhi->addIncoming(llvm::ConstantFP::get(f32Ty, 0.0), entryBB);

  // Load A element for this phase:
  // Global coordinate of A element: row = by * 16 + ty, a_col = phase * 16 + tx
  llvm::Value* aCol = builder.CreateAdd(builder.CreateMul(phasePhi, cTileDim), tx, "a_col");
  llvm::Value* rowValid = builder.CreateICmpSLT(row, mArg, "row_valid");
  llvm::Value* aColValid = builder.CreateICmpSLT(aCol, kArg, "acol_valid");
  llvm::Value* aValid = builder.CreateAnd(rowValid, aColValid, "a_valid");
  builder.CreateCondBr(aValid, loadABB, zeroABB);

  // --- Load A (in bounds) ---
  builder.SetInsertPoint(loadABB);
  llvm::Value* row64 = builder.CreateSExt(row, i64Ty);
  llvm::Value* k64 = builder.CreateSExt(kArg, i64Ty);
  llvm::Value* aCol64 = builder.CreateSExt(aCol, i64Ty);
  llvm::Value* aOffset = builder.CreateAdd(builder.CreateMul(row64, k64), aCol64, "a_offset");
  llvm::Value* aElemPtr = builder.CreateGEP(f32Ty, aPtr, aOffset, "a_elem_ptr");
  llvm::Value* loadedA = builder.CreateLoad(f32Ty, aElemPtr, "loaded_a");
  builder.CreateBr(joinABB);

  // --- Zero A (out of bounds) ---
  builder.SetInsertPoint(zeroABB);
  builder.CreateBr(joinABB);

  // --- Join A and Store to Shared Memory ---
  builder.SetInsertPoint(joinABB);
  auto* valAPhi = builder.CreatePHI(f32Ty, 2, "val_a");
  valAPhi->addIncoming(loadedA, loadABB);
  valAPhi->addIncoming(llvm::ConstantFP::get(f32Ty, 0.0), zeroABB);

  llvm::Value* tileAPtr = builder.CreateGEP(
      tileTy, sharedA, {builder.getInt32(0), localTileIdx}, "tileA_elem_ptr");
  builder.CreateStore(valAPhi, tileAPtr);

  // Load B element for this phase:
  // Global coordinate of B element: b_row = phase * 16 + ty, col = bx * 16 + tx
  llvm::Value* bRow = builder.CreateAdd(builder.CreateMul(phasePhi, cTileDim), ty, "b_row");
  llvm::Value* bRowValid = builder.CreateICmpSLT(bRow, kArg, "brow_valid");
  llvm::Value* colValid = builder.CreateICmpSLT(col, nArg, "col_valid");
  llvm::Value* bValid = builder.CreateAnd(bRowValid, colValid, "b_valid");
  builder.CreateCondBr(bValid, loadBBB, zeroBBB);

  // --- Load B (in bounds) ---
  builder.SetInsertPoint(loadBBB);
  llvm::Value* bRow64 = builder.CreateSExt(bRow, i64Ty);
  llvm::Value* n64 = builder.CreateSExt(nArg, i64Ty);
  llvm::Value* col64 = builder.CreateSExt(col, i64Ty);
  llvm::Value* bOffset = builder.CreateAdd(builder.CreateMul(bRow64, n64), col64, "b_offset");
  llvm::Value* bElemPtr = builder.CreateGEP(f32Ty, bPtr, bOffset, "b_elem_ptr");
  llvm::Value* loadedB = builder.CreateLoad(f32Ty, bElemPtr, "loaded_b");
  builder.CreateBr(joinBBB);

  // --- Zero B (out of bounds) ---
  builder.SetInsertPoint(zeroBBB);
  builder.CreateBr(joinBBB);

  // --- Join B, Store to Shared Memory, Synchronize, Accumulate ---
  builder.SetInsertPoint(joinBBB);
  auto* valBPhi = builder.CreatePHI(f32Ty, 2, "val_b");
  valBPhi->addIncoming(loadedB, loadBBB);
  valBPhi->addIncoming(llvm::ConstantFP::get(f32Ty, 0.0), zeroBBB);

  llvm::Value* tileBPtr = builder.CreateGEP(
      tileTy, sharedB, {builder.getInt32(0), localTileIdx}, "tileB_elem_ptr");
  builder.CreateStore(valBPhi, tileBPtr);

  // Synchronization Point 1: Wait until all threads in CTA populate shared memory tiles
  emitBlockBarrier(*mod, builder);

  // Unrolled accumulation for the 16 elements in the current tile
  llvm::Value* curAccum = accumPhi;
  llvm::Value* tyTimes16 = builder.CreateMul(ty, cTileDim, "ty_x_16");
  for (unsigned k = 0; k < kTileDim; ++k) {
    llvm::Value* aIdx = builder.CreateAdd(tyTimes16, builder.getInt32(k));
    llvm::Value* aSharedPtr = builder.CreateGEP(
        tileTy, sharedA, {builder.getInt32(0), aIdx});
    llvm::Value* aSharedVal = builder.CreateLoad(f32Ty, aSharedPtr, "a_tile_val");

    llvm::Value* bIdx = builder.CreateAdd(builder.getInt32(k * kTileDim), tx);
    llvm::Value* bSharedPtr = builder.CreateGEP(
        tileTy, sharedB, {builder.getInt32(0), bIdx});
    llvm::Value* bSharedVal = builder.CreateLoad(f32Ty, bSharedPtr, "b_tile_val");

    llvm::Value* prod = builder.CreateFMul(aSharedVal, bSharedVal, "prod");
    curAccum = builder.CreateFAdd(curAccum, prod, "accum_next");
  }

  // Synchronization Point 2: Wait until all threads in CTA finish reading tiles
  // before the next phase overwrites them
  emitBlockBarrier(*mod, builder);

  // Advance phase
  llvm::Value* nextPhase = builder.CreateAdd(phasePhi, builder.getInt32(1), "next_phase");
  phasePhi->addIncoming(nextPhase, joinBBB);
  accumPhi->addIncoming(curAccum, joinBBB);

  llvm::Value* loopDone = builder.CreateICmpSGE(nextPhase, numPhases, "loop_done");
  builder.CreateCondBr(loopDone, afterLoopBB, loopHeaderBB);

  // --- After Loop ---
  builder.SetInsertPoint(afterLoopBB);
  auto* finalAccum = builder.CreatePHI(f32Ty, 2, "final_accum");
  finalAccum->addIncoming(llvm::ConstantFP::get(f32Ty, 0.0), entryBB);
  finalAccum->addIncoming(curAccum, joinBBB);

  // Guard output store: only threads strictly inside matrix bounds C[M, N] write
  llvm::Value* rowInBounds = builder.CreateICmpSLT(row, mArg, "row_in_bounds");
  llvm::Value* colInBounds = builder.CreateICmpSLT(col, nArg, "col_in_bounds");
  llvm::Value* cInBounds = builder.CreateAnd(rowInBounds, colInBounds, "c_in_bounds");
  builder.CreateCondBr(cInBounds, storeCBB, exitBB);

  // --- Store C ---
  builder.SetInsertPoint(storeCBB);
  llvm::Value* outRow64 = builder.CreateSExt(row, i64Ty);
  llvm::Value* outN64 = builder.CreateSExt(nArg, i64Ty);
  llvm::Value* outCol64 = builder.CreateSExt(col, i64Ty);
  llvm::Value* cOffset = builder.CreateAdd(
      builder.CreateMul(outRow64, outN64), outCol64, "c_offset");
  llvm::Value* cElemPtr = builder.CreateGEP(f32Ty, outPtr, cOffset, "c_elem_ptr");
  builder.CreateStore(finalAccum, cElemPtr);
  builder.CreateBr(exitBB);

  // --- Exit ---
  builder.SetInsertPoint(exitBB);
  builder.CreateRetVoid();

  return mod;
}

} // namespace tinytensor::backend

#else // !defined(TINYTENSOR_HAS_LLVM)

namespace tinytensor::backend {

std::unique_ptr<llvm::Module> createTiledMatMulModule(
    llvm::LLVMContext&,
    llvm::TargetMachine*,
    std::string_view,
    std::string_view) {
  return nullptr;
}

} // namespace tinytensor::backend

#endif // TINYTENSOR_HAS_LLVM
