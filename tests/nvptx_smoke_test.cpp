#include <gtest/gtest.h>

#include "tinytensor/backend/nvptx_target.hpp"

#if defined(TINYTENSOR_HAS_LLVM)
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Target/TargetMachine.h>
#endif

#include <filesystem>
#include <fstream>
#include <string>

using namespace tinytensor::backend;

#if defined(TINYTENSOR_HAS_LLVM)

class NVPTXSmokeTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(initializeNVPTXTarget()) << "NVPTX target must be supported in this LLVM build";
  }
};

TEST_F(NVPTXSmokeTest, TargetInitializationAndSupport) {
  EXPECT_TRUE(isNVPTXTargetSupported());
  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", "sm_70", ""});
  ASSERT_NE(tm, nullptr);
  EXPECT_STREQ(tm->getTargetTriple().str().c_str(), "nvptx64-nvidia-cuda");
}

TEST_F(NVPTXSmokeTest, TargetMachineCreationRejectsInvalidTriple) {
  auto tm = createNVPTXTargetMachine({"invalid_triple_arch_unknown", "sm_70", ""});
  EXPECT_EQ(tm, nullptr);
}

TEST_F(NVPTXSmokeTest, EmitSmokeTestModuleVerification) {
  llvm::LLVMContext ctx;
  auto tm = createNVPTXTargetMachine();
  ASSERT_NE(tm, nullptr);

  auto mod = createSmokeTestModule(ctx, tm.get(), "smoke_mod", "tiny_smoke");
  ASSERT_NE(mod, nullptr);

  // Check target triple and data layout
  EXPECT_EQ(mod->getTargetTriple().str(), "nvptx64-nvidia-cuda");
  EXPECT_FALSE(mod->getDataLayoutStr().empty());

  // Check module verification passes
  std::string verifyError;
  bool verified = verifyLLVMModule(*mod, &verifyError);
  EXPECT_TRUE(verified) << "Verification failed: " << verifyError;

  // Verify function properties
  llvm::Function* kernel = mod->getFunction("tiny_smoke");
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->arg_size(), 1u);
  EXPECT_TRUE(kernel->getReturnType()->isVoidTy());

  // Check kernel calling convention for PTX entry
  EXPECT_EQ(kernel->getCallingConv(), llvm::CallingConv::PTX_Kernel);

  // Check kernel argument address space 1 (global memory)
  llvm::Type* argTy = kernel->getArg(0)->getType();
  ASSERT_TRUE(argTy->isPointerTy());
  EXPECT_EQ(argTy->getPointerAddressSpace(), 1u);

  // Check nvvm.annotations metadata
  llvm::NamedMDNode* annotations = mod->getNamedMetadata("nvvm.annotations");
  ASSERT_NE(annotations, nullptr);
  EXPECT_GE(annotations->getNumOperands(), 1u);

  // Inspect generated LLVM IR text without brittle full-string comparison
  std::string irText = emitLLVMIR(*mod);
  EXPECT_NE(irText.find("void @tiny_smoke(ptr addrspace(1)"), std::string::npos);
  EXPECT_NE(irText.find("llvm.nvvm.read.ptx.sreg.tid.x"), std::string::npos);
  EXPECT_NE(irText.find("llvm.nvvm.read.ptx.sreg.ctaid.x"), std::string::npos);
  EXPECT_NE(irText.find("llvm.nvvm.read.ptx.sreg.ntid.x"), std::string::npos);
  EXPECT_NE(irText.find("!nvvm.annotations"), std::string::npos);
  EXPECT_NE(irText.find("!\"kernel\", i32 1"), std::string::npos);
}

TEST_F(NVPTXSmokeTest, EmitPTXAssemblyVerification) {
  llvm::LLVMContext ctx;
  auto tm = createNVPTXTargetMachine({"nvptx64-nvidia-cuda", "sm_70", ""});
  ASSERT_NE(tm, nullptr);

  auto mod = createSmokeTestModule(ctx, tm.get(), "smoke_mod", "tiny_smoke");
  ASSERT_NE(mod, nullptr);

  std::string ptxError;
  auto ptxOpt = emitPTX(*mod, *tm, &ptxError);
  if (!ptxOpt.has_value()) {
    // If emission failed, save IR artifact for debugging
    std::string ir = emitLLVMIR(*mod);
    saveArtifact("/tmp/failed_smoke_test.ll", ir);
    FAIL() << "PTX emission failed: " << ptxError;
  }

  const std::string& ptx = *ptxOpt;
  EXPECT_FALSE(ptx.empty());

  // Verify critical PTX properties
  EXPECT_NE(ptx.find(".version"), std::string::npos);
  EXPECT_NE(ptx.find(".target sm_70"), std::string::npos);
  EXPECT_NE(ptx.find(".address_size 64"), std::string::npos);
  // Must be emitted as an entry point (.visible .entry), not a plain device function (.func)
  EXPECT_NE(ptx.find(".visible .entry tiny_smoke"), std::string::npos);
  EXPECT_NE(ptx.find("%tid.x"), std::string::npos);
  EXPECT_NE(ptx.find("%ctaid.x"), std::string::npos);
  EXPECT_NE(ptx.find("%ntid.x"), std::string::npos);
  EXPECT_NE(ptx.find("st.global"), std::string::npos);

  // Save generated artifacts to disk for user inspection
  saveArtifact("/tmp/tiny_smoke.ll", emitLLVMIR(*mod));
  saveArtifact("/tmp/tiny_smoke.ptx", ptx);
}

TEST_F(NVPTXSmokeTest, VerifyModuleCatchesDeliberatelyMalformedModule) {
  llvm::LLVMContext ctx;
  llvm::Module malformedMod("bad_mod", ctx);
  malformedMod.setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));

  auto* voidTy = llvm::Type::getVoidTy(ctx);
  auto* fnTy = llvm::FunctionType::get(voidTy, {}, false);
  auto* fn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, "bad_fn", malformedMod);

  // Create a basic block with NO terminator instruction (violates LLVM IR invariants)
  auto* bb = llvm::BasicBlock::Create(ctx, "entry", fn);
  llvm::IRBuilder<> builder(bb);
  auto* i32Ty = llvm::Type::getInt32Ty(ctx);
  auto* one = llvm::ConstantInt::get(i32Ty, 1);
  builder.CreateAdd(one, one);
  // Deliberately omit CreateRetVoid()

  std::string errorMsg;
  bool valid = verifyLLVMModule(malformedMod, &errorMsg);
  EXPECT_FALSE(valid);
  EXPECT_FALSE(errorMsg.empty());
  EXPECT_NE(errorMsg.find("Basic Block in function 'bad_fn' does not have terminator"), std::string::npos);
}

TEST_F(NVPTXSmokeTest, SaveArtifactHelper) {
  std::string testPath = "/tmp/tinytensor_test_artifact.txt";
  std::string content = "Smoke test artifact content 12345";
  saveArtifact(testPath, content);

  std::ifstream inFile(testPath);
  ASSERT_TRUE(inFile.is_open());
  std::string readContent((std::istreambuf_iterator<char>(inFile)),
                           std::istreambuf_iterator<char>());
  EXPECT_EQ(readContent, content);
  std::filesystem::remove(testPath);
}

#else

TEST(NVPTXSmokeTest, LLVMDisabledGracefulFallback) {
  EXPECT_FALSE(initializeNVPTXTarget());
  EXPECT_FALSE(isNVPTXTargetSupported());
}

#endif // TINYTENSOR_HAS_LLVM
