#include <gtest/gtest.h>
#include "tinytensor/ir.hpp"
#include "tinytensor/passes/dce.hpp"
#include "tinytensor/runtime/cpu_reference.hpp"
#include "tinytensor/verifier.hpp"

using namespace tinytensor;

TEST(DCETest, CanonicalBaselineDCEAndSemanticEquivalence) {
  Module module;
  IRBuilder b(module);

  // DSL:
  // input A: f32[1024,1024]
  // input B: f32[1024,1024]
  // input bias: f32[1024]
  // T1 = matmul(A, B)
  // T2 = T1 + bias
  // C = relu(T2)
  // unused = exp(C)
  // output C

  Value& A = b.input("A", TensorType{DType::F32, {2, 2}});
  Value& B = b.input("B", TensorType{DType::F32, {2, 2}});
  Value& bias = b.input("bias", TensorType{DType::F32, {2}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  Value& unused = b.createUnary(Op::Exp, C);
  b.output(C);

  EXPECT_EQ(unused.id(), 6u);

  // 1. Pre-DCE Verifier Check
  EXPECT_TRUE(verifyModule(module).ok);

  // 2. Pre-DCE CPU Reference Run
  TensorBuffer A_buf(TensorType{DType::F32, {2, 2}}, {1.0f, 2.0f, 3.0f, 4.0f});
  TensorBuffer B_buf(TensorType{DType::F32, {2, 2}}, {0.5f, 0.0f, 0.0f, 0.5f});
  TensorBuffer bias_buf(TensorType{DType::F32, {2}}, {-1.0f, 2.0f});

  CPUReferenceExecutor executor;
  auto pre_outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});
  ASSERT_EQ(pre_outputs.size(), 1u);

  // 3. Run DCE Pass
  DCEPass dce;
  bool changed = dce.runOnModule(module);
  EXPECT_TRUE(changed);

  // 4. Post-DCE Verifier Check
  EXPECT_TRUE(verifyModule(module).ok);

  // 5. Post-DCE CPU Reference Run & Semantic Equivalence Proof
  auto post_outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});
  ASSERT_EQ(post_outputs.size(), 1u);
  EXPECT_EQ(pre_outputs[0].type, post_outputs[0].type);
  EXPECT_EQ(pre_outputs[0].data, post_outputs[0].data);

  // 6. Verify surviving live IDs: %0 through %5 must be preserved
  EXPECT_EQ(A.id(), 0u);
  EXPECT_EQ(B.id(), 1u);
  EXPECT_EQ(bias.id(), 2u);
  EXPECT_EQ(T1.id(), 3u);
  EXPECT_EQ(T2.id(), 4u);
  EXPECT_EQ(C.id(), 5u);

  // 7. Verify Canonical Post-DCE Text Schema
  const std::string expected_post_dce =
      "module {\n"
      "  %0 = input \"A\" : f32[2,2]\n"
      "  %1 = input \"B\" : f32[2,2]\n"
      "  %2 = input \"bias\" : f32[2]\n"
      "  %3 = matmul %0, %1 : f32[2,2]\n"
      "  %4 = add %3, %2 : f32[2,2]\n"
      "  %5 = relu %4 : f32[2,2]\n"
      "  output %5\n"
      "}\n";

  EXPECT_EQ(module.print(), expected_post_dce);
}

TEST(DCETest, DeadChainRemoval) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("x", TensorType{DType::F32, {4}});

  // Live branch: in -> neg -> output
  Value& live_neg = b.createUnary(Op::Neg, in);
  b.output(live_neg);

  // Dead chain: in -> add -> relu -> exp (none output)
  Value& dead1 = b.createBinary(Op::Add, in, in);
  Value& dead2 = b.createUnary(Op::Relu, dead1);
  Value& dead3 = b.createUnary(Op::Exp, dead2);
  EXPECT_EQ(dead3.id(), 4u);

  // 4 instructions total
  EXPECT_EQ(module.instructions().size(), 4u);
  EXPECT_TRUE(verifyModule(module).ok);

  TensorBuffer in_buf(TensorType{DType::F32, {4}}, {1.0f, -2.0f, 3.0f, -4.0f});
  CPUReferenceExecutor executor;
  auto pre_outputs = executor.execute(module, {{"x", in_buf}});

  DCEPass dce;
  bool changed = dce.runOnModule(module);
  EXPECT_TRUE(changed);

  // Only the 1 live instruction must remain
  EXPECT_EQ(module.instructions().size(), 1u);
  EXPECT_EQ(module.instructions()[0]->result().id(), live_neg.id());
  EXPECT_TRUE(verifyModule(module).ok);

  auto post_outputs = executor.execute(module, {{"x", in_buf}});
  EXPECT_EQ(pre_outputs[0].data, post_outputs[0].data);
}

TEST(DCETest, MultipleIndependentLiveOutputs) {
  Module module;
  IRBuilder b(module);

  Value& x = b.input("x", TensorType{DType::F32, {4}});
  Value& y = b.input("y", TensorType{DType::F32, {4}});

  Value& out1 = b.createUnary(Op::Relu, x);
  Value& out2 = b.createUnary(Op::Neg, y);
  Value& dead = b.createBinary(Op::Mul, x, y);
  EXPECT_EQ(dead.id(), 4u);

  b.output(out1);
  b.output(out2);

  EXPECT_EQ(module.instructions().size(), 3u);
  EXPECT_TRUE(verifyModule(module).ok);

  TensorBuffer x_buf(TensorType{DType::F32, {4}}, {-1.0f, 2.0f, -3.0f, 4.0f});
  TensorBuffer y_buf(TensorType{DType::F32, {4}}, {5.0f, -6.0f, 7.0f, -8.0f});

  CPUReferenceExecutor executor;
  auto pre_outputs = executor.execute(module, {{"x", x_buf}, {"y", y_buf}});
  ASSERT_EQ(pre_outputs.size(), 2u);

  DCEPass dce;
  bool changed = dce.runOnModule(module);
  EXPECT_TRUE(changed);

  // Exactly 2 live output instructions remain
  EXPECT_EQ(module.instructions().size(), 2u);
  EXPECT_TRUE(verifyModule(module).ok);

  auto post_outputs = executor.execute(module, {{"x", x_buf}, {"y", y_buf}});
  ASSERT_EQ(post_outputs.size(), 2u);
  EXPECT_EQ(pre_outputs[0].data, post_outputs[0].data);
  EXPECT_EQ(pre_outputs[1].data, post_outputs[1].data);
}

TEST(DCETest, RetainSharedProducerWithLiveConsumer) {
  Module module;
  IRBuilder b(module);

  Value& x = b.input("x", TensorType{DType::F32, {4}});
  Value& y = b.input("y", TensorType{DType::F32, {4}});

  // Shared producer: p = x + y
  Value& p = b.createBinary(Op::Add, x, y);
  // Consumer 1 is live (output)
  Value& live_c = b.createUnary(Op::Relu, p);
  b.output(live_c);
  // Consumer 2 is dead
  Value& dead_c = b.createUnary(Op::Exp, p);
  EXPECT_EQ(dead_c.id(), 4u);

  EXPECT_EQ(p.useCount(), 2u);
  EXPECT_EQ(module.instructions().size(), 3u);

  DCEPass dce;
  bool changed = dce.runOnModule(module);
  EXPECT_TRUE(changed);

  // Producer 'p' and 'live_c' must be retained; only 'dead_c' is removed
  EXPECT_EQ(module.instructions().size(), 2u);
  EXPECT_EQ(p.useCount(), 1u);
  EXPECT_EQ(p.users()[0], live_c.definingInst());
  EXPECT_TRUE(verifyModule(module).ok);
}

TEST(DCETest, NoOpIdempotencyAndConvergence) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("x", TensorType{DType::F32, {4}});
  Value& out = b.createUnary(Op::Relu, in);
  b.output(out);

  DCEPass dce;
  // Module is already clean: DCE is a no-op
  bool changed_first = dce.runOnModule(module);
  EXPECT_FALSE(changed_first);

  // Running again must remain false (stable fixpoint)
  bool changed_second = dce.runOnModule(module);
  EXPECT_FALSE(changed_second);
}

TEST(DCETest, PassManagerRunsDCEAndCatchesCorruptPass) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("x", TensorType{DType::F32, {4}});
  Value& out = b.createUnary(Op::Relu, in);
  Value& dead = b.createUnary(Op::Exp, in);
  b.output(out);
  EXPECT_EQ(dead.id(), 2u);

  PassManager pm(/*verify_after_each_pass=*/true);
  pm.addPass(std::make_unique<DCEPass>());

  bool changed = pm.run(module);
  EXPECT_TRUE(changed);
  EXPECT_EQ(module.instructions().size(), 1u);
  EXPECT_TRUE(verifyModule(module).ok);

  // Create a deliberately corrupt pass that introduces an invariant violation
  class CorruptPass final : public Pass {
   public:
    std::string name() const override { return "CorruptPass"; }
    bool runOnModule(Module& m) override {
      if (!m.instructions().empty()) {
        // Clear operand to null (violating non-null operand invariant)
        m.instructions()[0]->setOperand(0, &m.instructions()[0]->result());
        return true;
      }
      return false;
    }
  };

  PassManager corrupt_pm(/*verify_after_each_pass=*/true);
  corrupt_pm.addPass(std::make_unique<CorruptPass>());

  // PassManager must intercept the corruption and throw runtime_error
  EXPECT_THROW(corrupt_pm.run(module), std::runtime_error);
}
