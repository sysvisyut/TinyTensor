#include <gtest/gtest.h>
#include "tinytensor/ir.hpp"
#include "tinytensor/runtime/cpu_reference.hpp"
#include <cmath>
#include <vector>

using namespace tinytensor;

// Tolerance rationale:
// For single-precision IEEE 754 floats, machine epsilon is ~1.19e-7.
// In matrix multiplication [M, K] x [K, N], each element is an accumulation of K products.
// For K <= 1024, numerical error can accumulate up to K * eps ~= 1.2e-4.
// For CPU reference tests, we test exact elementwise matches with 1e-6f tolerance,
// and matrix products with 1e-5f tolerance.
constexpr float kElementwiseTolerance = 1e-6f;
constexpr float kMatMulTolerance = 1e-5f;

TEST(CPUReferenceTest, ElementwiseUnaryAndBinary) {
  Module module;
  IRBuilder b(module);

  Value& x = b.input("x", TensorType{DType::F32, {4}});
  Value& y = b.input("y", TensorType{DType::F32, {4}});

  Value& add_val = b.createBinary(Op::Add, x, y);
  Value& sub_val = b.createBinary(Op::Sub, x, y);
  Value& mul_val = b.createBinary(Op::Mul, x, y);
  Value& div_val = b.createBinary(Op::Div, x, y);
  Value& max_val = b.createBinary(Op::Max, x, y);
  Value& neg_val = b.createUnary(Op::Neg, x);

  b.output(add_val);
  b.output(sub_val);
  b.output(mul_val);
  b.output(div_val);
  b.output(max_val);
  b.output(neg_val);

  TensorBuffer x_buf(TensorType{DType::F32, {4}}, {10.0f, -4.0f, 6.0f, 0.0f});
  TensorBuffer y_buf(TensorType{DType::F32, {4}}, {2.0f, 2.0f, -3.0f, 5.0f});

  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"x", x_buf}, {"y", y_buf}});
  ASSERT_EQ(outputs.size(), 6u);

  // Add: 10+2=12, -4+2=-2, 6+(-3)=3, 0+5=5
  EXPECT_NEAR(outputs[0].data[0], 12.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[1], -2.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[2], 3.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[3], 5.0f, kElementwiseTolerance);

  // Sub: 10-2=8, -4-2=-6, 6-(-3)=9, 0-5=-5
  EXPECT_NEAR(outputs[1].data[0], 8.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[1].data[1], -6.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[1].data[2], 9.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[1].data[3], -5.0f, kElementwiseTolerance);

  // Mul: 10*2=20, -4*2=-8, 6*-3=-18, 0*5=0
  EXPECT_NEAR(outputs[2].data[0], 20.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[2].data[1], -8.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[2].data[2], -18.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[2].data[3], 0.0f, kElementwiseTolerance);

  // Div: 10/2=5, -4/2=-2, 6/(-3)=-2, 0/5=0
  EXPECT_NEAR(outputs[3].data[0], 5.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[3].data[1], -2.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[3].data[2], -2.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[3].data[3], 0.0f, kElementwiseTolerance);

  // Max: max(10,2)=10, max(-4,2)=2, max(6,-3)=6, max(0,5)=5
  EXPECT_NEAR(outputs[4].data[0], 10.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[4].data[1], 2.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[4].data[2], 6.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[4].data[3], 5.0f, kElementwiseTolerance);

  // Neg: -10, 4, -6, -0
  EXPECT_NEAR(outputs[5].data[0], -10.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[5].data[1], 4.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[5].data[2], -6.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[5].data[3], 0.0f, kElementwiseTolerance);
}

TEST(CPUReferenceTest, ActivationReluAndExp) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("in", TensorType{DType::F32, {4}});
  Value& r = b.createUnary(Op::Relu, in);
  Value& e = b.createUnary(Op::Exp, in);
  b.output(r);
  b.output(e);

  TensorBuffer in_buf(TensorType{DType::F32, {4}}, {-3.0f, 0.0f, 2.5f, -0.01f});
  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"in", in_buf}});

  // ReLU: max(0, x)
  EXPECT_NEAR(outputs[0].data[0], 0.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[1], 0.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[2], 2.5f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[3], 0.0f, kElementwiseTolerance);

  // Exp: exp(x)
  EXPECT_NEAR(outputs[1].data[0], std::exp(-3.0f), kElementwiseTolerance);
  EXPECT_NEAR(outputs[1].data[1], 1.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[1].data[2], std::exp(2.5f), 1e-4f);
  EXPECT_NEAR(outputs[1].data[3], std::exp(-0.01f), kElementwiseTolerance);
}

TEST(CPUReferenceTest, ScalarAndRank1TrailingBroadcast) {
  Module module;
  IRBuilder b(module);

  // Matrix [2, 3] + rank-1 bias [3] + constant scalar
  Value& mat = b.input("mat", TensorType{DType::F32, {2, 3}});
  Value& bias = b.input("bias", TensorType{DType::F32, {3}});
  Value& c = b.constant(1.5f);

  Value& biased = b.createBinary(Op::Add, mat, bias);
  Value& scaled = b.createBinary(Op::Mul, biased, c);
  b.output(scaled);

  TensorBuffer mat_buf(TensorType{DType::F32, {2, 3}}, {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
  });
  TensorBuffer bias_buf(TensorType{DType::F32, {3}}, {10.0f, 20.0f, 30.0f});

  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"mat", mat_buf}, {"bias", bias_buf}});

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].type, (TensorType{DType::F32, {2, 3}}));

  // Row 0: (1+10)*1.5 = 16.5, (2+20)*1.5 = 33.0, (3+30)*1.5 = 49.5
  // Row 1: (4+10)*1.5 = 21.0, (5+20)*1.5 = 37.5, (6+30)*1.5 = 54.0
  EXPECT_NEAR(outputs[0].data[0], 16.5f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[1], 33.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[2], 49.5f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[3], 21.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[4], 37.5f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[5], 54.0f, kElementwiseTolerance);
}

TEST(CPUReferenceTest, NonSquareMatMulAndBoundaryShapes) {
  Module module;
  IRBuilder b(module);

  // Non-square matrix multiplication: [2, 3] x [3, 2] -> [2, 2]
  Value& A = b.input("A", TensorType{DType::F32, {2, 3}});
  Value& B = b.input("B", TensorType{DType::F32, {3, 2}});
  Value& C = b.createMatMul(A, B);
  b.output(C);

  // A:
  // [ 1, 2, 3 ]
  // [ 4, 5, 6 ]
  TensorBuffer A_buf(TensorType{DType::F32, {2, 3}}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // B:
  // [ 7,  8 ]
  // [ 9,  1 ]
  // [ 2,  3 ]
  TensorBuffer B_buf(TensorType{DType::F32, {3, 2}}, {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f});

  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}});

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].type, (TensorType{DType::F32, {2, 2}}));

  // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
  // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9  = 19
  // C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
  // C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
  EXPECT_NEAR(outputs[0].data[0], 31.0f, kMatMulTolerance);
  EXPECT_NEAR(outputs[0].data[1], 19.0f, kMatMulTolerance);
  EXPECT_NEAR(outputs[0].data[2], 85.0f, kMatMulTolerance);
  EXPECT_NEAR(outputs[0].data[3], 55.0f, kMatMulTolerance);
}

TEST(CPUReferenceTest, MatMulBoundary1x1AndVectorShapes) {
  Module module;
  IRBuilder b(module);

  // 1x1 matmul boundary case
  Value& A = b.input("A", TensorType{DType::F32, {1, 1}});
  Value& B = b.input("B", TensorType{DType::F32, {1, 1}});
  Value& C = b.createMatMul(A, B);
  b.output(C);

  TensorBuffer A_buf(TensorType{DType::F32, {1, 1}}, {4.5f});
  TensorBuffer B_buf(TensorType{DType::F32, {1, 1}}, {2.0f});

  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}});
  EXPECT_NEAR(outputs[0].data[0], 9.0f, kMatMulTolerance);
}

TEST(CPUReferenceTest, NonSquareMatMul17x19x23) {
  Module module;
  IRBuilder b(module);

  // Test dimensions [17, 19] x [19, 23] -> [17, 23]
  Value& A = b.input("A", TensorType{DType::F32, {17, 19}});
  Value& B = b.input("B", TensorType{DType::F32, {19, 23}});
  Value& C = b.createMatMul(A, B);
  b.output(C);

  std::vector<float> a_data(17 * 19);
  for (std::size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
  }

  std::vector<float> b_data(19 * 23);
  for (std::size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  }

  TensorBuffer A_buf(TensorType{DType::F32, {17, 19}}, a_data);
  TensorBuffer B_buf(TensorType{DType::F32, {19, 23}}, b_data);

  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}});

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].type, (TensorType{DType::F32, {17, 23}}));

  // Compute reference dot product for a few sample entries
  // Check C[0, 0]
  float expected_0_0 = 0.0f;
  for (int k = 0; k < 19; ++k) {
    expected_0_0 += a_data[static_cast<std::size_t>(0 * 19 + k)] *
                    b_data[static_cast<std::size_t>(k * 23 + 0)];
  }
  EXPECT_NEAR(outputs[0].data[0], expected_0_0, kMatMulTolerance);

  // Check C[16, 22] (last element)
  float expected_last = 0.0f;
  for (int k = 0; k < 19; ++k) {
    expected_last += a_data[static_cast<std::size_t>(16 * 19 + k)] *
                     b_data[static_cast<std::size_t>(k * 23 + 22)];
  }
  EXPECT_NEAR(outputs[0].data[17 * 23 - 1], expected_last, kMatMulTolerance);
}

TEST(CPUReferenceTest, BaselineEndToEndMatMulAddRelu) {
  Module module;
  IRBuilder b(module);

  // Baseline DAG: T1 = matmul(A, B); T2 = T1 + bias; C = relu(T2); output C
  Value& A = b.input("A", TensorType{DType::F32, {2, 2}});
  Value& B = b.input("B", TensorType{DType::F32, {2, 2}});
  Value& bias = b.input("bias", TensorType{DType::F32, {2}});

  Value& T1 = b.createMatMul(A, B);
  Value& T2 = b.createBinary(Op::Add, T1, bias);
  Value& C = b.createUnary(Op::Relu, T2);
  b.output(C);

  // A = [[1, 2], [3, 4]]
  // B = [[1, 0], [0, 1]] (identity) -> T1 = A
  // bias = [-2, 10] -> T2 = [[1-2, 2+10], [3-2, 4+10]] = [[-1, 12], [1, 14]]
  // C = relu(T2) = [[0, 12], [1, 14]]
  TensorBuffer A_buf(TensorType{DType::F32, {2, 2}}, {1.0f, 2.0f, 3.0f, 4.0f});
  TensorBuffer B_buf(TensorType{DType::F32, {2, 2}}, {1.0f, 0.0f, 0.0f, 1.0f});
  TensorBuffer bias_buf(TensorType{DType::F32, {2}}, {-2.0f, 10.0f});

  CPUReferenceExecutor executor;
  auto outputs = executor.execute(module, {{"A", A_buf}, {"B", B_buf}, {"bias", bias_buf}});

  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_NEAR(outputs[0].data[0], 0.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[1], 12.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[2], 1.0f, kElementwiseTolerance);
  EXPECT_NEAR(outputs[0].data[3], 14.0f, kElementwiseTolerance);
}

TEST(CPUReferenceTest, FailureCasesMissingOrMismatchedInputs) {
  Module module;
  IRBuilder b(module);

  Value& in = b.input("expected_name", TensorType{DType::F32, {4}});
  b.output(in);

  CPUReferenceExecutor executor;

  // Missing input
  EXPECT_THROW(executor.execute(module, std::unordered_map<std::string, TensorBuffer>{}),
               std::invalid_argument);

  // Mismatched shape in input buffer: declared {4}, provided {8}
  TensorBuffer wrong_shape(TensorType{DType::F32, {8}}, std::vector<float>(8, 1.0f));
  EXPECT_THROW(executor.execute(module, {{"expected_name", wrong_shape}}), std::invalid_argument);

  // Invalid buffer construction with inconsistent data length
  EXPECT_THROW(TensorBuffer(TensorType{DType::F32, {4}}, {1.0f, 2.0f}), std::invalid_argument);
}
