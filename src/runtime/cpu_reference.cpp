#include "tinytensor/runtime/cpu_reference.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

namespace tinytensor {

// -----------------------------------------------------------------------------
// TensorBuffer
// -----------------------------------------------------------------------------
TensorBuffer::TensorBuffer(TensorType t, std::vector<float> d)
    : type(std::move(t)), data(std::move(d)) {
  if (data.size() != numElements()) {
    throw std::invalid_argument(
        "TensorBuffer: data size (" + std::to_string(data.size()) +
        ") does not match expected element count (" + std::to_string(numElements()) + ")");
  }
}

std::size_t TensorBuffer::numElements() const noexcept {
  if (type.isScalar()) {
    return 1;
  }
  std::size_t count = 1;
  for (std::int64_t dim : type.shape) {
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

TensorBuffer TensorBuffer::zeros(TensorType t) {
  TensorBuffer tb;
  tb.type = std::move(t);
  tb.data.assign(tb.numElements(), 0.0f);
  return tb;
}

TensorBuffer TensorBuffer::scalar(float val) {
  TensorBuffer tb;
  tb.type = TensorType{DType::F32, {}};
  tb.data = {val};
  return tb;
}

bool TensorBuffer::operator==(const TensorBuffer& other) const noexcept {
  return type == other.type && data == other.data;
}

// -----------------------------------------------------------------------------
// CPUReferenceExecutor Evaluation Helpers
// -----------------------------------------------------------------------------
static TensorBuffer evalUnary(Op op, const TensorBuffer& operand) {
  TensorBuffer result;
  result.type = operand.type;
  result.data.resize(operand.data.size());

  switch (op) {
    case Op::Relu:
      for (std::size_t i = 0; i < operand.data.size(); ++i) {
        result.data[i] = std::max(0.0f, operand.data[i]);
      }
      break;
    case Op::Exp:
      for (std::size_t i = 0; i < operand.data.size(); ++i) {
        result.data[i] = std::exp(operand.data[i]);
      }
      break;
    case Op::Neg:
      for (std::size_t i = 0; i < operand.data.size(); ++i) {
        result.data[i] = -operand.data[i];
      }
      break;
    default:
      throw std::invalid_argument("CPUReferenceExecutor: unsupported unary operation");
  }
  return result;
}

static float applyBinaryScalarOp(Op op, float a, float b) {
  switch (op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a * b;
    case Op::Div:
      if (b == 0.0f) {
        // IEEE 754 division semantics
        return a / b;
      }
      return a / b;
    case Op::Max: return std::max(a, b);
    default:
      throw std::invalid_argument("CPUReferenceExecutor: unsupported binary operation");
  }
}

static TensorBuffer evalBinary(Op op, const TensorBuffer& lhs, const TensorBuffer& rhs) {
  // Case 1: Identical shapes
  if (lhs.type.shape == rhs.type.shape) {
    TensorBuffer result;
    result.type = lhs.type;
    result.data.resize(lhs.data.size());
    for (std::size_t i = 0; i < lhs.data.size(); ++i) {
      result.data[i] = applyBinaryScalarOp(op, lhs.data[i], rhs.data[i]);
    }
    return result;
  }

  // Case 2: LHS is scalar
  if (lhs.type.isScalar()) {
    TensorBuffer result;
    result.type = rhs.type;
    result.data.resize(rhs.data.size());
    float scalar_val = lhs.data[0];
    for (std::size_t i = 0; i < rhs.data.size(); ++i) {
      result.data[i] = applyBinaryScalarOp(op, scalar_val, rhs.data[i]);
    }
    return result;
  }

  // Case 3: RHS is scalar
  if (rhs.type.isScalar()) {
    TensorBuffer result;
    result.type = lhs.type;
    result.data.resize(lhs.data.size());
    float scalar_val = rhs.data[0];
    for (std::size_t i = 0; i < lhs.data.size(); ++i) {
      result.data[i] = applyBinaryScalarOp(op, lhs.data[i], scalar_val);
    }
    return result;
  }

  // Case 4: LHS is [M, N] and RHS is [N] (rank-1 trailing broadcast)
  if (lhs.type.rank() == 2 && rhs.type.rank() == 1) {
    std::int64_t M = lhs.type.shape[0];
    std::int64_t N = lhs.type.shape[1];
    if (rhs.type.shape[0] != N) {
      throw std::invalid_argument("CPUReferenceExecutor: broadcast dimension mismatch");
    }
    TensorBuffer result;
    result.type = lhs.type;
    result.data.resize(static_cast<std::size_t>(M * N));
    for (std::int64_t r = 0; r < M; ++r) {
      for (std::int64_t c = 0; c < N; ++c) {
        std::size_t idx = static_cast<std::size_t>(r * N + c);
        result.data[idx] = applyBinaryScalarOp(op, lhs.data[idx], rhs.data[static_cast<std::size_t>(c)]);
      }
    }
    return result;
  }

  // Case 5: LHS is [N] and RHS is [M, N] (rank-1 trailing broadcast)
  if (lhs.type.rank() == 1 && rhs.type.rank() == 2) {
    std::int64_t M = rhs.type.shape[0];
    std::int64_t N = rhs.type.shape[1];
    if (lhs.type.shape[0] != N) {
      throw std::invalid_argument("CPUReferenceExecutor: broadcast dimension mismatch");
    }
    TensorBuffer result;
    result.type = rhs.type;
    result.data.resize(static_cast<std::size_t>(M * N));
    for (std::int64_t r = 0; r < M; ++r) {
      for (std::int64_t c = 0; c < N; ++c) {
        std::size_t idx = static_cast<std::size_t>(r * N + c);
        result.data[idx] = applyBinaryScalarOp(op, lhs.data[static_cast<std::size_t>(c)], rhs.data[idx]);
      }
    }
    return result;
  }

  throw std::invalid_argument("CPUReferenceExecutor: incompatible broadcast shapes for binary operation");
}

static TensorBuffer evalMatMul(const TensorBuffer& lhs, const TensorBuffer& rhs) {
  if (lhs.type.rank() != 2 || rhs.type.rank() != 2) {
    throw std::invalid_argument("CPUReferenceExecutor: MatMul requires rank-2 inputs");
  }
  std::int64_t M = lhs.type.shape[0];
  std::int64_t K = lhs.type.shape[1];
  std::int64_t K2 = rhs.type.shape[0];
  std::int64_t N = rhs.type.shape[1];

  if (K != K2) {
    throw std::invalid_argument("CPUReferenceExecutor: MatMul inner dimension mismatch: " +
                                std::to_string(K) + " vs " + std::to_string(K2));
  }

  TensorBuffer result;
  result.type = TensorType{DType::F32, {M, N}};
  result.data.assign(static_cast<std::size_t>(M * N), 0.0f);

  for (std::int64_t r = 0; r < M; ++r) {
    for (std::int64_t c = 0; c < N; ++c) {
      float acc = 0.0f;
      for (std::int64_t k = 0; k < K; ++k) {
        acc += lhs.data[static_cast<std::size_t>(r * K + k)] *
               rhs.data[static_cast<std::size_t>(k * N + c)];
      }
      result.data[static_cast<std::size_t>(r * N + c)] = acc;
    }
  }
  return result;
}

static TensorBuffer evalFusedMatMulEpilogue(
    const TensorBuffer& A, const TensorBuffer& B, const TensorBuffer& bias) {
  if (A.type.rank() != 2 || B.type.rank() != 2 || bias.type.rank() != 1) {
    throw std::invalid_argument("CPUReferenceExecutor: FusedMatMulEpilogue rank mismatch");
  }
  std::int64_t M = A.type.shape[0];
  std::int64_t K = A.type.shape[1];
  std::int64_t N = B.type.shape[1];
  if (B.type.shape[0] != K || bias.type.shape[0] != N) {
    throw std::invalid_argument("CPUReferenceExecutor: FusedMatMulEpilogue dimension mismatch");
  }

  TensorBuffer result;
  result.type = TensorType{DType::F32, {M, N}};
  result.data.assign(static_cast<std::size_t>(M * N), 0.0f);

  for (std::int64_t r = 0; r < M; ++r) {
    for (std::int64_t c = 0; c < N; ++c) {
      float acc = 0.0f;
      for (std::int64_t k = 0; k < K; ++k) {
        acc += A.data[static_cast<std::size_t>(r * K + k)] *
               B.data[static_cast<std::size_t>(k * N + c)];
      }
      // Apply column bias and ReLU
      result.data[static_cast<std::size_t>(r * N + c)] = std::max(0.0f, acc + bias.data[static_cast<std::size_t>(c)]);
    }
  }
  return result;
}

// -----------------------------------------------------------------------------
// CPUReferenceExecutor
// -----------------------------------------------------------------------------
std::vector<TensorBuffer> CPUReferenceExecutor::execute(
    const Module& module,
    const std::unordered_map<std::string, TensorBuffer>& inputs) const {
  std::unordered_map<Value::Id, TensorBuffer> id_inputs;

  for (const Value* in : module.inputs()) {
    auto it = inputs.find(in->name());
    if (it == inputs.end()) {
      throw std::invalid_argument("CPUReferenceExecutor: missing input buffer for '" + in->name() +
                                  "' (% " + std::to_string(in->id()) + ")");
    }
    id_inputs[in->id()] = it->second;
  }

  return execute(module, id_inputs);
}

std::vector<TensorBuffer> CPUReferenceExecutor::execute(
    const Module& module,
    const std::unordered_map<Value::Id, TensorBuffer>& inputs) const {
  std::unordered_map<const Value*, TensorBuffer> env;

  // 1. Bind inputs
  for (const Value* in : module.inputs()) {
    auto it = inputs.find(in->id());
    if (it == inputs.end()) {
      throw std::invalid_argument("CPUReferenceExecutor: missing input buffer for Value %" +
                                  std::to_string(in->id()));
    }
    const TensorBuffer& buf = it->second;
    if (buf.type != in->type()) {
      throw std::invalid_argument("CPUReferenceExecutor: input buffer type mismatch for Value %" +
                                  std::to_string(in->id()));
    }
    if (buf.data.size() != buf.numElements()) {
      throw std::invalid_argument("CPUReferenceExecutor: input buffer data size mismatch for Value %" +
                                  std::to_string(in->id()));
    }
    env[in] = buf;
  }

  // 2. Bind constants
  for (const auto& [cval, f] : module.constants()) {
    env[cval] = TensorBuffer::scalar(f);
  }

  // 3. Sequentially evaluate straight-line instructions
  for (const auto& inst : module.instructions()) {
    const Value* res = &inst->result();
    Op op = inst->op();

    switch (op) {
      case Op::Relu:
      case Op::Exp:
      case Op::Neg: {
        const Value* opnd = inst->operands()[0];
        auto it = env.find(opnd);
        if (it == env.end()) {
          throw std::runtime_error("CPUReferenceExecutor: operand value %" + std::to_string(opnd->id()) +
                                   " not found in environment");
        }
        env[res] = evalUnary(op, it->second);
        break;
      }
      case Op::Add:
      case Op::Sub:
      case Op::Mul:
      case Op::Div:
      case Op::Max: {
        const Value* lhs_val = inst->operands()[0];
        const Value* rhs_val = inst->operands()[1];
        auto lhs_it = env.find(lhs_val);
        auto rhs_it = env.find(rhs_val);
        if (lhs_it == env.end() || rhs_it == env.end()) {
          throw std::runtime_error("CPUReferenceExecutor: binary operands not found in environment");
        }
        env[res] = evalBinary(op, lhs_it->second, rhs_it->second);
        break;
      }
      case Op::MatMul: {
        const Value* lhs_val = inst->operands()[0];
        const Value* rhs_val = inst->operands()[1];
        auto lhs_it = env.find(lhs_val);
        auto rhs_it = env.find(rhs_val);
        if (lhs_it == env.end() || rhs_it == env.end()) {
          throw std::runtime_error("CPUReferenceExecutor: MatMul operands not found in environment");
        }
        env[res] = evalMatMul(lhs_it->second, rhs_it->second);
        break;
      }
      case Op::FusedMatMulEpilogue: {
        const Value* a_val = inst->operands()[0];
        const Value* b_val = inst->operands()[1];
        const Value* bias_val = inst->operands()[2];
        auto a_it = env.find(a_val);
        auto b_it = env.find(b_val);
        auto bias_it = env.find(bias_val);
        if (a_it == env.end() || b_it == env.end() || bias_it == env.end()) {
          throw std::runtime_error("CPUReferenceExecutor: FusedMatMulEpilogue operands missing");
        }
        env[res] = evalFusedMatMulEpilogue(a_it->second, b_it->second, bias_it->second);
        break;
      }
      default:
        throw std::invalid_argument("CPUReferenceExecutor: unsupported opcode in evaluation");
    }
  }

  // 4. Collect output buffers
  std::vector<TensorBuffer> outputs;
  outputs.reserve(module.outputs().size());
  for (const Value* out : module.outputs()) {
    auto it = env.find(out);
    if (it == env.end()) {
      throw std::runtime_error("CPUReferenceExecutor: output value %" + std::to_string(out->id()) +
                               " not found in evaluated environment");
    }
    outputs.push_back(it->second);
  }

  return outputs;
}

} // namespace tinytensor
