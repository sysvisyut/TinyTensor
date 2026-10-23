#include <gtest/gtest.h>
#include "tinytensor/ir.hpp"

using namespace tinytensor;

TEST(IRTest, BaselineMatMulAddReluAndCanonicalPrint) {
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

  Value& A = b.input("A", TensorType{DType::F32, {1024, 1024}});
  Value& B = b.input("B", TensorType{DType::F32, {1024, 1024}});
  Value& bias = b.input("bias", TensorType{DType::F32, {1024}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  Value& unused = b.createUnary(Op::Exp, C);

  b.output(C);

  // Verify IDs
  EXPECT_EQ(A.id(), 0u);
  EXPECT_EQ(B.id(), 1u);
  EXPECT_EQ(bias.id(), 2u);
  EXPECT_EQ(T1.id(), 3u);
  EXPECT_EQ(T2.id(), 4u);
  EXPECT_EQ(C.id(), 5u);
  EXPECT_EQ(unused.id(), 6u);

  // Verify Types
  EXPECT_EQ(A.type(), (TensorType{DType::F32, {1024, 1024}}));
  EXPECT_EQ(B.type(), (TensorType{DType::F32, {1024, 1024}}));
  EXPECT_EQ(bias.type(), (TensorType{DType::F32, {1024}}));
  EXPECT_EQ(T1.type(), (TensorType{DType::F32, {1024, 1024}}));
  EXPECT_EQ(T2.type(), (TensorType{DType::F32, {1024, 1024}}));
  EXPECT_EQ(C.type(), (TensorType{DType::F32, {1024, 1024}}));
  EXPECT_EQ(unused.type(), (TensorType{DType::F32, {1024, 1024}}));

  // Verify Defining Instructions
  EXPECT_EQ(A.definingInst(), nullptr);
  EXPECT_EQ(B.definingInst(), nullptr);
  EXPECT_EQ(bias.definingInst(), nullptr);
  ASSERT_NE(T1.definingInst(), nullptr);
  EXPECT_EQ(T1.definingInst()->op(), Op::MatMul);
  ASSERT_NE(T2.definingInst(), nullptr);
  EXPECT_EQ(T2.definingInst()->op(), Op::Add);
  ASSERT_NE(C.definingInst(), nullptr);
  EXPECT_EQ(C.definingInst()->op(), Op::Relu);
  ASSERT_NE(unused.definingInst(), nullptr);
  EXPECT_EQ(unused.definingInst()->op(), Op::Exp);

  // Verify Use Counts
  EXPECT_EQ(A.useCount(), 1u);
  EXPECT_EQ(B.useCount(), 1u);
  EXPECT_EQ(bias.useCount(), 1u);
  EXPECT_EQ(T1.useCount(), 1u);
  EXPECT_EQ(T2.useCount(), 1u);
  EXPECT_EQ(C.useCount(), 1u);      // used by unused (exp)
  EXPECT_EQ(unused.useCount(), 0u); // dead

  // Verify Output Status
  EXPECT_FALSE(A.isOutput());
  EXPECT_FALSE(T1.isOutput());
  EXPECT_FALSE(T2.isOutput());
  EXPECT_TRUE(C.isOutput());
  EXPECT_FALSE(unused.isOutput());

  // Canonical printed output - byte for byte check against ir_schema.md lines 53-63
  const std::string expected_print =
      "module {\n"
      "  %0 = input \"A\" : f32[1024,1024]\n"
      "  %1 = input \"B\" : f32[1024,1024]\n"
      "  %2 = input \"bias\" : f32[1024]\n"
      "  %3 = matmul %0, %1 : f32[1024,1024]\n"
      "  %4 = add %3, %2 : f32[1024,1024]\n"
      "  %5 = relu %4 : f32[1024,1024]\n"
      "  %6 = exp %5 : f32[1024,1024]\n"
      "  output %5\n"
      "}\n";

  EXPECT_EQ(module.print(), expected_print);
}

TEST(IRTest, EraseInstructionPreservesLiveIDsAndReciprocalLinks) {
  Module module;
  IRBuilder b(module);

  Value& A = b.input("A", TensorType{DType::F32, {1024, 1024}});
  Value& B = b.input("B", TensorType{DType::F32, {1024, 1024}});
  Value& bias = b.input("bias", TensorType{DType::F32, {1024}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  Value& unused = b.createUnary(Op::Exp, C);
  b.output(C);

  Instruction* dead_inst = unused.definingInst();
  ASSERT_NE(dead_inst, nullptr);

  // Erase the dead exp instruction
  bool erased = module.eraseInstruction(*dead_inst);
  EXPECT_TRUE(erased);

  // C's use count must drop to 0
  EXPECT_EQ(C.useCount(), 0u);

  // Live IDs must remain unchanged (%0 through %5)
  EXPECT_EQ(A.id(), 0u);
  EXPECT_EQ(B.id(), 1u);
  EXPECT_EQ(bias.id(), 2u);
  EXPECT_EQ(T1.id(), 3u);
  EXPECT_EQ(T2.id(), 4u);
  EXPECT_EQ(C.id(), 5u);

  // Canonical printed output after DCE - byte for byte check against ir_schema.md lines 68-77
  const std::string expected_post_dce =
      "module {\n"
      "  %0 = input \"A\" : f32[1024,1024]\n"
      "  %1 = input \"B\" : f32[1024,1024]\n"
      "  %2 = input \"bias\" : f32[1024]\n"
      "  %3 = matmul %0, %1 : f32[1024,1024]\n"
      "  %4 = add %3, %2 : f32[1024,1024]\n"
      "  %5 = relu %4 : f32[1024,1024]\n"
      "  output %5\n"
      "}\n";

  EXPECT_EQ(module.print(), expected_post_dce);
}

TEST(IRTest, EraseSafetyRejectsLiveOrOutputInstructions) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("X", TensorType{DType::F32, {16}});
  Value& v1 = b.createUnary(Op::Relu, in);
  Value& v2 = b.createUnary(Op::Exp, v1);
  b.output(v2);

  // v1 has a user (v2) -> cannot erase
  EXPECT_FALSE(module.eraseInstruction(*v1.definingInst()));
  EXPECT_EQ(v1.useCount(), 1u);

  // v2 is an output -> cannot erase
  EXPECT_FALSE(module.eraseInstruction(*v2.definingInst()));
  EXPECT_TRUE(v2.isOutput());
}

TEST(IRTest, ReplaceAllUsesWithUpdatesOperandsAndOutputs) {
  Module module;
  IRBuilder b(module);

  Value& in1 = b.input("x", TensorType{DType::F32, {8}});
  Value& in2 = b.input("y", TensorType{DType::F32, {8}});

  Value& r1 = b.createUnary(Op::Relu, in1);
  Value& r2 = b.createUnary(Op::Relu, in2);
  Value& sum = b.createBinary(Op::Add, r1, r2);
  b.output(sum);

  EXPECT_EQ(r1.useCount(), 1u);
  EXPECT_EQ(sum.definingInst()->operands()[0], &r1);

  // Replace all uses of r1 with r2
  module.replaceAllUsesWith(r1, r2);

  EXPECT_EQ(r1.useCount(), 0u);
  EXPECT_EQ(r2.useCount(), 2u);
  EXPECT_EQ(sum.definingInst()->operands()[0], &r2);
  EXPECT_EQ(sum.definingInst()->operands()[1], &r2);

  // Test output replacement
  module.replaceAllUsesWith(sum, r2);
  EXPECT_TRUE(r2.isOutput());
  EXPECT_FALSE(sum.isOutput());
  EXPECT_EQ(module.outputs().size(), 1u);
  EXPECT_EQ(module.outputs()[0], &r2);
}

TEST(IRTest, OperandMultiplicityAndSetOperand) {
  Module module;
  IRBuilder b(module);

  Value& x = b.input("x", TensorType{DType::F32, {4}});
  Value& double_x = b.createBinary(Op::Add, x, x);

  // x is used twice in the same instruction
  EXPECT_EQ(x.useCount(), 2u);
  EXPECT_EQ(x.users().size(), 2u);
  EXPECT_EQ(x.users()[0], double_x.definingInst());
  EXPECT_EQ(x.users()[1], double_x.definingInst());

  Value& y = b.input("y", TensorType{DType::F32, {4}});
  double_x.definingInst()->setOperand(0, &y);

  EXPECT_EQ(x.useCount(), 1u);
  EXPECT_EQ(y.useCount(), 1u);
  EXPECT_EQ(double_x.definingInst()->operands()[0], &y);
  EXPECT_EQ(double_x.definingInst()->operands()[1], &x);

  // Bounds and null checks
  EXPECT_THROW(double_x.definingInst()->setOperand(5, &y), std::out_of_range);
  EXPECT_THROW(double_x.definingInst()->setOperand(0, nullptr), std::invalid_argument);
}

TEST(IRTest, InvalidShapeAndMatMulChecks) {
  Module module;
  IRBuilder b(module);

  Value& rank1 = b.input("v", TensorType{DType::F32, {16}});
  Value& m1 = b.input("A", TensorType{DType::F32, {16, 32}});
  Value& m2_bad = b.input("B", TensorType{DType::F32, {64, 16}});
  Value& m3_incompatible_binary = b.input("C", TensorType{DType::F32, {16, 64}});

  // MatMul requires rank 2
  EXPECT_THROW(b.createMatMul(rank1, m1), std::invalid_argument);

  // MatMul inner dimension mismatch [16, 32] x [64, 16]
  EXPECT_THROW(b.createMatMul(m1, m2_bad), std::invalid_argument);

  // Binary incompatible shapes [16, 32] and [16, 64]
  EXPECT_THROW(b.createBinary(Op::Add, m1, m3_incompatible_binary), std::invalid_argument);
}

TEST(IRTest, ConstantsAndScalarBroadcasting) {
  Module module;
  IRBuilder b(module);

  Value& v = b.input("x", TensorType{DType::F32, {32}});
  Value& c = b.constant(2.5f);

  EXPECT_TRUE(c.type().isScalar());
  EXPECT_EQ(c.definingInst(), nullptr);

  // Scalar broadcasts to rank-1 tensor
  Value& scaled = b.createBinary(Op::Mul, v, c);
  EXPECT_EQ(scaled.type(), (TensorType{DType::F32, {32}}));

  std::string printed = module.print();
  EXPECT_NE(printed.find("  %1 = constant 2.5 : f32[]\n"), std::string::npos);
  EXPECT_NE(printed.find("  %2 = mul %0, %1 : f32[32]\n"), std::string::npos);
}
