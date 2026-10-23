#include "tinytensor/verifier.hpp"
#include "tinytensor/ir.hpp"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tinytensor {

static std::string formatShape(const std::vector<std::int64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      s += ",";
    }
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

static bool checkBinaryBroadcast(const TensorType& lhs, const TensorType& rhs, TensorType& out_type) {
  if (lhs.dtype != rhs.dtype) {
    return false;
  }
  if (lhs.shape == rhs.shape) {
    out_type = lhs;
    return true;
  }
  if (lhs.isScalar()) {
    out_type = rhs;
    return true;
  }
  if (rhs.isScalar()) {
    out_type = lhs;
    return true;
  }
  // Rank-1 [N] broadcasting to trailing dimension of rank-2 [M, N]
  if (lhs.rank() == 2 && rhs.rank() == 1 && lhs.shape[1] == rhs.shape[0]) {
    out_type = lhs;
    return true;
  }
  if (lhs.rank() == 1 && rhs.rank() == 2 && lhs.shape[0] == rhs.shape[1]) {
    out_type = rhs;
    return true;
  }
  return false;
}

VerificationResult verifyModule(const Module& module) {
  std::unordered_set<const Value*> val_set;
  std::unordered_map<Value::Id, const Value*> id_to_val;

  // 1. Verify Values and ID uniqueness
  for (const auto& v : module.values()) {
    if (!v) {
      return {false, "Module contains a null Value pointer"};
    }
    if (id_to_val.find(v->id()) != id_to_val.end()) {
      return {false, "Duplicate Value ID %" + std::to_string(v->id())};
    }
    id_to_val[v->id()] = v.get();
    val_set.insert(v.get());

    if (v->type().dtype != DType::F32) {
      return {false, "Value %" + std::to_string(v->id()) + " has unsupported dtype"};
    }
    if (v->type().rank() > 2) {
      return {false, "Value %" + std::to_string(v->id()) + " has invalid rank " +
                         std::to_string(v->type().rank()) + " (must be 0, 1, or 2)"};
    }
    for (std::int64_t dim : v->type().shape) {
      if (dim <= 0) {
        return {false, "Value %" + std::to_string(v->id()) + " has non-positive dimension " +
                           std::to_string(dim)};
      }
    }
  }

  // 2. Verify Inputs
  std::unordered_set<const Value*> input_set;
  for (const Value* in : module.inputs()) {
    if (!in || val_set.find(in) == val_set.end()) {
      return {false, "Module input does not belong to module values"};
    }
    if (in->definingInst() != nullptr) {
      return {false, "Input %" + std::to_string(in->id()) + " must have null defining instruction"};
    }
    input_set.insert(in);
  }

  // 3. Map instructions and check instruction nullness
  std::unordered_map<const Instruction*, std::size_t> inst_index;
  std::unordered_set<const Instruction*> inst_set;
  for (std::size_t i = 0; i < module.instructions().size(); ++i) {
    const auto& inst = module.instructions()[i];
    if (!inst) {
      return {false, "Module contains null instruction at index " + std::to_string(i)};
    }
    inst_index[inst.get()] = i;
    inst_set.insert(inst.get());
  }

  // 4. Verify Instructions, SSA dominance, typing, and arity
  std::unordered_map<const Value*, const Instruction*> defined_by;
  for (std::size_t i = 0; i < module.instructions().size(); ++i) {
    const Instruction* inst = module.instructions()[i].get();
    const Value* result = &inst->result();

    if (val_set.find(result) == val_set.end()) {
      return {false, "Instruction at index " + std::to_string(i) + " result does not belong to module"};
    }
    if (result->definingInst() != inst) {
      return {false, "Instruction %" + std::to_string(result->id()) +
                         " result does not point back to its defining instruction"};
    }
    if (defined_by.find(result) != defined_by.end()) {
      return {false, "Value %" + std::to_string(result->id()) + " has duplicate defining instructions"};
    }
    defined_by[result] = inst;

    if (inst->op() == Op::Input || inst->op() == Op::Constant) {
      return {false, "Instruction %" + std::to_string(result->id()) +
                         " has illegal opcode (Input/Constant are not instructions)"};
    }

    // Check operands
    for (std::size_t op_idx = 0; op_idx < inst->operands().size(); ++op_idx) {
      const Value* opnd = inst->operands()[op_idx];
      if (!opnd) {
        return {false, "Instruction %" + std::to_string(result->id()) + " has null operand at index " +
                           std::to_string(op_idx)};
      }
      if (val_set.find(opnd) == val_set.end()) {
        return {false, "Instruction %" + std::to_string(result->id()) + " operand at index " +
                           std::to_string(op_idx) + " does not belong to this module"};
      }

      // Straight-line dominance: operand defining instruction must precede this instruction
      if (opnd->definingInst() != nullptr) {
        auto def_it = inst_index.find(opnd->definingInst());
        if (def_it == inst_index.end()) {
          return {false, "Instruction %" + std::to_string(result->id()) + " operand %" +
                             std::to_string(opnd->id()) + " has foreign defining instruction"};
        }
        if (def_it->second >= i) {
          return {false, "Operand %" + std::to_string(opnd->id()) + " used at instruction %" +
                             std::to_string(result->id()) +
                             " is defined after its use (dominance violation)"};
        }
      }
    }

    // Arity and shape checks per Op
    switch (inst->op()) {
      case Op::Relu:
      case Op::Exp:
      case Op::Neg: {
        if (inst->operands().size() != 1) {
          return {false, "Unary instruction %" + std::to_string(result->id()) +
                             " requires 1 operand, got " + std::to_string(inst->operands().size())};
        }
        if (inst->operands()[0]->type() != result->type()) {
          return {false, "Unary instruction %" + std::to_string(result->id()) + " result type " +
                             formatShape(result->type().shape) + " does not match operand type " +
                             formatShape(inst->operands()[0]->type().shape)};
        }
        break;
      }
      case Op::Add:
      case Op::Sub:
      case Op::Mul:
      case Op::Div:
      case Op::Max: {
        if (inst->operands().size() != 2) {
          return {false, "Binary instruction %" + std::to_string(result->id()) +
                             " requires 2 operands, got " + std::to_string(inst->operands().size())};
        }
        TensorType expected;
        if (!checkBinaryBroadcast(inst->operands()[0]->type(), inst->operands()[1]->type(), expected)) {
          return {false, "Binary instruction %" + std::to_string(result->id()) +
                             " has incompatible operand shapes " +
                             formatShape(inst->operands()[0]->type().shape) + " and " +
                             formatShape(inst->operands()[1]->type().shape)};
        }
        if (result->type() != expected) {
          return {false, "Binary instruction %" + std::to_string(result->id()) + " result shape " +
                             formatShape(result->type().shape) + " does not match broadcast shape " +
                             formatShape(expected.shape)};
        }
        break;
      }
      case Op::MatMul: {
        if (inst->operands().size() != 2) {
          return {false, "MatMul instruction %" + std::to_string(result->id()) +
                             " requires 2 operands, got " + std::to_string(inst->operands().size())};
        }
        const auto& tA = inst->operands()[0]->type();
        const auto& tB = inst->operands()[1]->type();
        if (tA.rank() != 2 || tB.rank() != 2) {
          return {false, "MatMul instruction %" + std::to_string(result->id()) +
                             " operands must be rank-2 tensors, got ranks " +
                             std::to_string(tA.rank()) + " and " + std::to_string(tB.rank())};
        }
        if (tA.shape[1] != tB.shape[0]) {
          return {false, "MatMul instruction %" + std::to_string(result->id()) +
                             " inner dimension mismatch: " + formatShape(tA.shape) + " and " +
                             formatShape(tB.shape)};
        }
        TensorType expected{DType::F32, {tA.shape[0], tB.shape[1]}};
        if (result->type() != expected) {
          return {false, "MatMul instruction %" + std::to_string(result->id()) + " result shape " +
                             formatShape(result->type().shape) + " does not match expected shape " +
                             formatShape(expected.shape)};
        }
        break;
      }
      case Op::FusedMatMulEpilogue: {
        if (inst->operands().size() != 3) {
          return {false, "FusedMatMulEpilogue instruction %" + std::to_string(result->id()) +
                             " requires 3 operands (A, B, bias), got " +
                             std::to_string(inst->operands().size())};
        }
        const auto& tA = inst->operands()[0]->type();
        const auto& tB = inst->operands()[1]->type();
        const auto& tBias = inst->operands()[2]->type();
        if (tA.rank() != 2 || tB.rank() != 2) {
          return {false, "FusedMatMulEpilogue instruction %" + std::to_string(result->id()) +
                             " A and B must be rank-2"};
        }
        if (tA.shape[1] != tB.shape[0]) {
          return {false, "FusedMatMulEpilogue instruction %" + std::to_string(result->id()) +
                             " inner dimension mismatch"};
        }
        if (tBias.rank() != 1 || tBias.shape[0] != tB.shape[1]) {
          return {false, "FusedMatMulEpilogue instruction %" + std::to_string(result->id()) +
                             " bias shape " + formatShape(tBias.shape) +
                             " does not match column dimension " + std::to_string(tB.shape[1])};
        }
        TensorType expected{DType::F32, {tA.shape[0], tB.shape[1]}};
        if (result->type() != expected) {
          return {false, "FusedMatMulEpilogue instruction %" + std::to_string(result->id()) +
                             " result shape does not match expected shape"};
        }
        break;
      }
      case Op::FusedElementwise: {
        if (inst->operands().empty()) {
          return {false, "FusedElementwise instruction %" + std::to_string(result->id()) +
                             " requires at least 1 operand"};
        }
        break;
      }
      default:
        break;
    }
  }

  // 5. Verify that non-leaf values have a defining instruction
  for (const auto& v : module.values()) {
    if (input_set.find(v.get()) == input_set.end() && v->definingInst() == nullptr) {
      if (!v->type().isScalar()) {
        return {false, "Non-leaf Value %" + std::to_string(v->id()) + " has null defining instruction"};
      }
    }
    if (v->definingInst() != nullptr) {
      if (inst_set.find(v->definingInst()) == inst_set.end()) {
        return {false, "Value %" + std::to_string(v->id()) + " has foreign defining instruction"};
      }
    }
  }

  // 6. Verify reciprocal use-def lists (multiplicity agreement)
  std::unordered_map<const Value*, std::size_t> actual_uses;
  for (const auto& inst : module.instructions()) {
    for (const Value* opnd : inst->operands()) {
      actual_uses[opnd]++;
    }
  }

  for (const auto& v : module.values()) {
    std::size_t expected_count = actual_uses[v.get()];
    if (v->useCount() != expected_count) {
      return {false, "Value %" + std::to_string(v->id()) + " useCount() (" +
                         std::to_string(v->useCount()) +
                         ") does not match actual instruction operand references (" +
                         std::to_string(expected_count) + ")"};
    }
    for (const Instruction* user : v->users()) {
      if (inst_set.find(user) == inst_set.end()) {
        return {false, "Value %" + std::to_string(v->id()) + " has foreign user instruction"};
      }
    }
  }

  // 7. Verify Outputs
  std::unordered_set<const Value*> output_set;
  for (const Value* out : module.outputs()) {
    if (!out || val_set.find(out) == val_set.end()) {
      return {false, "Output value does not belong to this module"};
    }
    if (!out->isOutput()) {
      return {false, "Output %" + std::to_string(out->id()) + " has isOutput() == false"};
    }
    output_set.insert(out);
  }

  for (const auto& v : module.values()) {
    if (v->isOutput() && output_set.find(v.get()) == output_set.end()) {
      return {false, "Value %" + std::to_string(v->id()) +
                         " has isOutput() == true but is not in module outputs"};
    }
  }

  return {true, ""};
}

} // namespace tinytensor
