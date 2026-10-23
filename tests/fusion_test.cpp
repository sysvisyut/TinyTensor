#include <gtest/gtest.h>
#include "tinytensor/ir.hpp"
#include "tinytensor/passes/fusion.hpp"
#include "tinytensor/runtime/cpu_reference.hpp"
#include "tinytensor/verifier.hpp"

using namespace tinytensor;

TEST(FusionTest, FusibleMatMulBiasReluCanonicalAndEquivalence) {
  Module module;
  IRBuilder b(module);

  // DSL:
  // input A: f32[1024,1024]
  // input B: f32[1024,1024]
  // input bias: f32[1024]
  // T1 = matmul(A, B)
  // T2 = T1 + bias
  // C = relu(T2)
  // output C

  Value& A = b.input("A", TensorType{DType::F32, {1024, 1024}});
  Value& B = b.input("B", TensorType{DType::F32, {1024, 1024}});
  Value& bias = b.input("bias", TensorType{DType::F32, {1024}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  b.output(C);

  // 1. Initial verification
  EXPECT_TRUE(verifyModule(module).ok);

  // 2. Run Fusion Pass
  FusionPass pass;
  bool changed = pass.runOnModule(module);
  EXPECT_TRUE(changed);

  // 3. Post-fusion verification
  VerificationResult res = verifyModule(module);
  EXPECT_TRUE(res.ok) << "Verifier failed after fusion: " << res.message;

  // 4. Verify canonical text schema matching specs/ir_schema.md lines 82-89 exactly
  const std::string expected_fused_ir =
      "module {\n"
      "  %0 = input \"A\" : f32[1024,1024]\n"
      "  %1 = input \"B\" : f32[1024,1024]\n"
      "  %2 = input \"bias\" : f32[1024]\n"
      "  %6 = fused_matmul_epilogue %0, %1, %2 : f32[1024,1024]\n"
      "  output %6\n"
      "}\n";

  EXPECT_EQ(module.print(), expected_fused_ir);
}

TEST(FusionTest, DownstreamUseOrderingPreservesDominance) {
  Module module;
  IRBuilder b(module);

  // Program with a downstream consumer after relu:
  // T1 = matmul(A, B)
  // T2 = T1 + bias
  // C = relu(T2)
  // downstream = exp(C)
  // output downstream
  Value& A = b.input("A", TensorType{DType::F32, {2, 2}});
  Value& B = b.input("B", TensorType{DType::F32, {2, 2}});
  Value& bias = b.input("bias", TensorType{DType::F32, {2}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  Value& downstream = b.createUnary(Op::Exp, C);
  b.output(downstream);

  EXPECT_TRUE(verifyModule(module).ok);

  TensorBuffer A_buf(TensorType{DType::F32, {2, 2}}, {1.0f, 2.0f, 3.0f, 4.0f});
  TensorBuffer B_buf(TensorType{DType::F32, {2, 2}}, {0.5f, 0.0f, 0.0f, 0.5f});
  TensorBuffer bias_buf(TensorType{DType::F32, {2}}, {-1.0f, 2.0f});

  CPUReferenceExecutor executor;
  auto pre_outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});

  FusionPass pass;
  bool changed = pass.runOnModule(module);
  EXPECT_TRUE(changed);

  // Must verify cleanly without any dominance violation
  VerificationResult res = verifyModule(module);
  EXPECT_TRUE(res.ok) << "Verifier dominance failure after downstream rewrite: " << res.message;

  // Instruction list must have fused_matmul_epilogue before exp
  ASSERT_EQ(module.instructions().size(), 2u);
  EXPECT_EQ(module.instructions()[0]->op(), Op::FusedMatMulEpilogue);
  EXPECT_EQ(module.instructions()[1]->op(), Op::Exp);

  // Downstream exp instruction's operand must be the fused instruction's result
  EXPECT_EQ(module.instructions()[1]->operands()[0], &module.instructions()[0]->result());

  // CPU reference must produce identical results
  auto post_outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});
  ASSERT_EQ(post_outputs.size(), 1u);
  EXPECT_NEAR(pre_outputs[0].data[0], post_outputs[0].data[0], 1e-5f);
  EXPECT_NEAR(pre_outputs[0].data[1], post_outputs[0].data[1], 1e-5f);
  EXPECT_NEAR(pre_outputs[0].data[2], post_outputs[0].data[2], 1e-5f);
  EXPECT_NEAR(pre_outputs[0].data[3], post_outputs[0].data[3], 1e-5f);
}

TEST(FusionTest, RejectSharedProducerWithOutsideConsumer) {
  Module module;
  IRBuilder b(module);

  // MatMul feeds both Add and an independent Exp consumer
  Value& A = b.input("A", TensorType{DType::F32, {2, 2}});
  Value& B = b.input("B", TensorType{DType::F32, {2, 2}});
  Value& bias = b.input("bias", TensorType{DType::F32, {2}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C1 = b.createUnary(Op::Relu, T2);
  Value& C2 = b.createUnary(Op::Exp, T1); // outside consumer of T1

  b.output(C1);
  b.output(C2);

  EXPECT_EQ(T1.useCount(), 2u);
  EXPECT_TRUE(verifyModule(module).ok);

  std::string pre_print = module.print();

  FusionPass pass;
  bool changed = pass.runOnModule(module);

  // Shared producer must NOT be fused
  EXPECT_FALSE(changed);
  EXPECT_EQ(module.print(), pre_print);
  EXPECT_TRUE(verifyModule(module).ok);
}

TEST(FusionTest, RejectIntermediateOutput) {
  Module module;
  IRBuilder b(module);

  Value& A = b.input("A", TensorType{DType::F32, {2, 2}});
  Value& B = b.input("B", TensorType{DType::F32, {2, 2}});
  Value& bias = b.input("bias", TensorType{DType::F32, {2}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);

  // T2 is marked as an external output
  b.output(T2);
  b.output(C);

  EXPECT_TRUE(T2.isOutput());
  EXPECT_TRUE(verifyModule(module).ok);

  std::string pre_print = module.print();

  FusionPass pass;
  bool changed = pass.runOnModule(module);

  // Cannot fuse because T2 is an observable output
  EXPECT_FALSE(changed);
  EXPECT_EQ(module.print(), pre_print);
  EXPECT_TRUE(verifyModule(module).ok);
}

TEST(FusionTest, RejectIncompatibleBiasShape) {
  Module module;
  // B is [4, 8], but bias is [6] (mismatched column dimension)
  Value& A = module.addInput("A", TensorType{DType::F32, {2, 4}});
  Value& B = module.addInput("B", TensorType{DType::F32, {4, 8}});
  Value& bias_bad = module.addInput("bias", TensorType{DType::F32, {6}});

  Instruction& t1 = module.addInstruction(Op::MatMul, {&A, &B}, TensorType{DType::F32, {2, 8}});
  Instruction& t2 = module.addInstruction(Op::Add, {&t1.result(), &bias_bad}, TensorType{DType::F32, {2, 8}});
  Instruction& c = module.addInstruction(Op::Relu, {&t2.result()}, TensorType{DType::F32, {2, 8}});
  module.addOutput(c.result());

  FusionPass pass;
  bool changed = pass.runOnModule(module);

  // Incompatible shape: rejected
  EXPECT_FALSE(changed);
}

TEST(FusionTest, CPUEquivalenceNumericalAccuracy) {
  Module module;
  IRBuilder b(module);

  // Non-square test case: A is [2, 3], B is [3, 2], bias is [2]
  Value& A = b.input("A", TensorType{DType::F32, {2, 3}});
  Value& B = b.input("B", TensorType{DType::F32, {3, 2}});
  Value& bias = b.input("bias", TensorType{DType::F32, {2}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  b.output(C);

  // A = [[ 1, -2,  3],
  //      [ 4,  0, -1]]
  // B = [[ 2, -1],
  //      [ 1,  3],
  //      [-1,  2]]
  // bias = [-5, 10]
  TensorBuffer A_buf(TensorType{DType::F32, {2, 3}}, {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f});
  TensorBuffer B_buf(TensorType{DType::F32, {3, 2}}, {2.0f, -1.0f, 1.0f, 3.0f, -1.0f, 2.0f});
  TensorBuffer bias_buf(TensorType{DType::F32, {2}}, {-5.0f, 10.0f});

  CPUReferenceExecutor executor;
  auto unfused_out = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});

  // Execute fusion pass
  FusionPass pass;
  EXPECT_TRUE(pass.runOnModule(module));
  EXPECT_TRUE(verifyModule(module).ok);

  auto fused_out = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});

  ASSERT_EQ(unfused_out.size(), 1u);
  ASSERT_EQ(fused_out.size(), 1u);
  EXPECT_EQ(unfused_out[0].type, fused_out[0].type);

  for (std::size_t i = 0; i < unfused_out[0].data.size(); ++i) {
    EXPECT_NEAR(unfused_out[0].data[i], fused_out[0].data[i], 1e-5f);
  }
}
