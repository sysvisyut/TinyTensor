#include <gtest/gtest.h>
#include "tinytensor/ir.hpp"
#include "tinytensor/verifier.hpp"

using namespace tinytensor;

TEST(VerifierTest, ValidBaselineMatMulAddReluModule) {
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

  EXPECT_EQ(unused.id(), 6u);

  // The module must pass verification even with dead value `unused`
  VerificationResult res = verifyModule(module);
  EXPECT_TRUE(res.ok) << "Verification failed: " << res.message;
  EXPECT_TRUE(res.message.empty());

  // Confirm deterministic canonical output
  const std::string golden =
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

  EXPECT_EQ(module.print(), golden);
  // Re-verify determinism across consecutive print calls
  EXPECT_EQ(module.print(), golden);
}

TEST(VerifierTest, RejectOperandDefinedAfterUseDominanceViolation) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("x", TensorType{DType::F32, {8}});
  // Create inst0: relu(in) -> %1
  Value& v1 = b.createUnary(Op::Relu, in);
  // Create inst1: exp(v1) -> %2
  Value& v2 = b.createUnary(Op::Exp, v1);
  b.output(v2);

  // Deliberately corrupt operand of inst0 to point to %2 (defined at inst1, after inst0)
  Instruction* inst0 = v1.definingInst();
  ASSERT_NE(inst0, nullptr);
  inst0->setOperand(0, &v2);

  VerificationResult res = verifyModule(module);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("dominance violation"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectForeignOrUndefinedOperand) {
  Module m1;
  Module m2;
  IRBuilder b1(m1);
  IRBuilder b2(m2);

  Value& in1 = b1.input("a", TensorType{DType::F32, {16}});
  Value& foreign_in = b2.input("foreign", TensorType{DType::F32, {16}});

  Value& v1 = b1.createUnary(Op::Relu, in1);
  b1.output(v1);

  // Set operand of m1's instruction to a foreign value from m2
  v1.definingInst()->setOperand(0, &foreign_in);

  VerificationResult res = verifyModule(m1);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("does not belong to this module"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectForeignOutput) {
  Module m1;
  Module m2;
  IRBuilder b1(m1);
  IRBuilder b2(m2);

  Value& in1 = b1.input("a", TensorType{DType::F32, {16}});
  Value& foreign_val = b2.input("foreign", TensorType{DType::F32, {16}});

  b1.createUnary(Op::Relu, in1);
  b1.output(foreign_val); // Adding foreign value to m1's outputs

  VerificationResult res = verifyModule(m1);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("Output value does not belong to this module"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectInvalidOperationArity) {
  Module module;
  // Construct a binary op with 1 operand directly using module.addInstruction
  Value& in = module.addInput("x", TensorType{DType::F32, {32}});
  module.addInstruction(Op::Add, {&in}, TensorType{DType::F32, {32}});

  VerificationResult res = verifyModule(module);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("requires 2 operands"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectIncompatibleMatMulShapes) {
  Module module;
  Value& m1 = module.addInput("A", TensorType{DType::F32, {1024, 512}});
  Value& m2 = module.addInput("B", TensorType{DType::F32, {256, 1024}});

  // Inner dimensions 512 != 256
  module.addInstruction(Op::MatMul, {&m1, &m2}, TensorType{DType::F32, {1024, 1024}});

  VerificationResult res = verifyModule(module);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("inner dimension mismatch"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectIncompatibleBinaryBroadcastShapes) {
  Module module;
  Value& a = module.addInput("A", TensorType{DType::F32, {16, 32}});
  Value& b = module.addInput("B", TensorType{DType::F32, {16, 64}});

  // Incompatible trailing dimension 32 != 64
  module.addInstruction(Op::Add, {&a, &b}, TensorType{DType::F32, {16, 32}});

  VerificationResult res = verifyModule(module);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("incompatible operand shapes"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectReciprocalUseDefMismatch) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("x", TensorType{DType::F32, {8}});
  Value& v = b.createUnary(Op::Relu, in);
  b.output(v);

  // Tamper with in's user list (simulate corrupted user list)
  // Calling removeUser removes the backlink while the instruction operand remains
  // Using const_cast to simulate corruption for test validation
  const_cast<std::vector<Instruction*>&>(in.users()).clear();

  VerificationResult res = verifyModule(module);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("does not match actual instruction operand references"), std::string::npos)
      << "Message: " << res.message;
}

TEST(VerifierTest, RejectIllegalOpcodeInstruction) {
  Module module;
  Value& in = module.addInput("x", TensorType{DType::F32, {8}});
  module.addInstruction(Op::Input, {&in}, TensorType{DType::F32, {8}});

  VerificationResult res = verifyModule(module);
  EXPECT_FALSE(res.ok);
  EXPECT_NE(res.message.find("illegal opcode"), std::string::npos)
      << "Message: " << res.message;
}
