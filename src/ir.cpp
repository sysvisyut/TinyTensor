#include "tinytensor/ir.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace tinytensor {

// -----------------------------------------------------------------------------
// TensorType
// -----------------------------------------------------------------------------
bool TensorType::operator==(const TensorType& other) const noexcept {
  return dtype == other.dtype && shape == other.shape;
}

// -----------------------------------------------------------------------------
// Value
// -----------------------------------------------------------------------------
Value::Value(Id id, TensorType type, std::string name, Instruction* defining_inst) noexcept
    : id_(id),
      type_(std::move(type)),
      name_(std::move(name)),
      defining_inst_(defining_inst),
      users_(),
      is_output_(false) {}

Value::Id Value::id() const noexcept { return id_; }
const TensorType& Value::type() const noexcept { return type_; }
const std::string& Value::name() const noexcept { return name_; }
Instruction* Value::definingInst() const noexcept { return defining_inst_; }
const std::vector<Instruction*>& Value::users() const noexcept { return users_; }
std::size_t Value::useCount() const noexcept { return users_.size(); }
bool Value::isOutput() const noexcept { return is_output_; }

void Value::addUser(Instruction* user) {
  users_.push_back(user);
}

void Value::removeUser(Instruction* user) {
  auto it = std::find(users_.begin(), users_.end(), user);
  if (it != users_.end()) {
    users_.erase(it);
  }
}

// -----------------------------------------------------------------------------
// Instruction
// -----------------------------------------------------------------------------
Instruction::Instruction(Op op, std::vector<Value*> operands, Value* result) noexcept
    : op_(op), operands_(std::move(operands)), result_(result) {}

Op Instruction::op() const noexcept { return op_; }
const std::vector<Value*>& Instruction::operands() const noexcept { return operands_; }
Value& Instruction::result() const noexcept { return *result_; }

void Instruction::setOperand(std::size_t index, Value* replacement) {
  if (index >= operands_.size()) {
    throw std::out_of_range("Instruction::setOperand: operand index out of range");
  }
  if (!replacement) {
    throw std::invalid_argument("Instruction::setOperand: replacement cannot be null");
  }
  Value* old = operands_[index];
  if (old == replacement) {
    return;
  }
  old->removeUser(this);
  operands_[index] = replacement;
  replacement->addUser(this);
}

// -----------------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------------
Value& Module::addInput(std::string name, TensorType type) {
  auto val = std::make_unique<Value>(next_value_id_++, std::move(type), std::move(name), nullptr);
  Value* ptr = val.get();
  values_.push_back(std::move(val));
  inputs_.push_back(ptr);
  return *ptr;
}

Value& Module::addConstant(float value) {
  Value::Id id = next_value_id_++;
  std::string name = "c" + std::to_string(id);
  auto val = std::make_unique<Value>(id, TensorType{DType::F32, {}}, std::move(name), nullptr);
  Value* ptr = val.get();
  values_.push_back(std::move(val));
  constants_.emplace_back(ptr, value);
  return *ptr;
}

Instruction& Module::addInstruction(Op op, std::vector<Value*> operands,
                                    TensorType result_type, std::string result_name) {
  Value::Id id = next_value_id_++;
  auto result_val = std::make_unique<Value>(id, std::move(result_type), std::move(result_name), nullptr);
  Value* result_ptr = result_val.get();
  values_.push_back(std::move(result_val));

  auto inst = std::make_unique<Instruction>(op, std::move(operands), result_ptr);
  Instruction* inst_ptr = inst.get();
  result_ptr->defining_inst_ = inst_ptr;

  for (Value* opnd : inst_ptr->operands()) {
    if (opnd) {
      opnd->addUser(inst_ptr);
    }
  }

  instructions_.push_back(std::move(inst));
  return *inst_ptr;
}

void Module::addOutput(Value& value) {
  value.is_output_ = true;
  outputs_.push_back(&value);
}

bool Module::eraseInstruction(Instruction& instruction) {
  if (instruction.result().useCount() > 0 || instruction.result().isOutput()) {
    return false;
  }

  auto inst_it = std::find_if(instructions_.begin(), instructions_.end(),
                              [&](const std::unique_ptr<Instruction>& item) {
                                return item.get() == &instruction;
                              });
  if (inst_it == instructions_.end()) {
    return false;
  }

  for (Value* opnd : instruction.operands()) {
    if (opnd) {
      opnd->removeUser(&instruction);
    }
  }

  Value* result_ptr = &instruction.result();
  instructions_.erase(inst_it);

  auto val_it = std::find_if(values_.begin(), values_.end(),
                            [&](const std::unique_ptr<Value>& item) {
                              return item.get() == result_ptr;
                            });
  if (val_it != values_.end()) {
    values_.erase(val_it);
  }

  return true;
}

void Module::replaceAllUsesWith(Value& from, Value& to) {
  if (&from == &to) {
    return;
  }

  std::vector<Instruction*> users_copy = from.users_;
  for (Instruction* user : users_copy) {
    for (std::size_t i = 0; i < user->operands().size(); ++i) {
      if (user->operands()[i] == &from) {
        user->setOperand(i, &to);
      }
    }
  }

  for (Value*& out : outputs_) {
    if (out == &from) {
      out = &to;
      to.is_output_ = true;
    }
  }
  from.is_output_ = false;
}

const std::vector<std::unique_ptr<Value>>& Module::values() const noexcept {
  return values_;
}

const std::vector<std::unique_ptr<Instruction>>& Module::instructions() const noexcept {
  return instructions_;
}

const std::vector<Value*>& Module::inputs() const noexcept {
  return inputs_;
}

const std::vector<Value*>& Module::outputs() const noexcept {
  return outputs_;
}

const std::vector<std::pair<Value*, float>>& Module::constants() const noexcept {
  return constants_;
}

static const char* opToString(Op op) {
  switch (op) {
    case Op::Input: return "input";
    case Op::Constant: return "constant";
    case Op::Add: return "add";
    case Op::Sub: return "sub";
    case Op::Mul: return "mul";
    case Op::Div: return "div";
    case Op::Max: return "max";
    case Op::Relu: return "relu";
    case Op::Exp: return "exp";
    case Op::Neg: return "neg";
    case Op::MatMul: return "matmul";
    case Op::FusedElementwise: return "fused_elementwise";
    case Op::FusedMatMulEpilogue: return "fused_matmul_epilogue";
  }
  return "unknown";
}

static std::string typeToString(const TensorType& type) {
  std::string s = "f32[";
  for (std::size_t i = 0; i < type.shape.size(); ++i) {
    if (i > 0) {
      s += ",";
    }
    s += std::to_string(type.shape[i]);
  }
  s += "]";
  return s;
}

static std::string formatFloat(float val) {
  std::ostringstream ss;
  ss << val;
  std::string s = ss.str();
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
    s += ".0";
  }
  return s;
}

std::string Module::print() const {
  std::ostringstream ss;
  ss << "module {\n";

  // 1. Inputs in declaration order
  for (const auto* in : inputs_) {
    ss << "  %" << in->id() << " = input \"" << in->name() << "\" : " << typeToString(in->type()) << "\n";
  }

  // 2. Constants in creation order
  for (const auto& [cval, f] : constants_) {
    ss << "  %" << cval->id() << " = constant " << formatFloat(f) << " : f32[]\n";
  }

  // 3. Instructions in construction order
  for (const auto& inst : instructions_) {
    ss << "  %" << inst->result().id() << " = " << opToString(inst->op());
    for (std::size_t i = 0; i < inst->operands().size(); ++i) {
      if (i == 0) {
        ss << " ";
      } else {
        ss << ", ";
      }
      ss << "%" << inst->operands()[i]->id();
    }
    ss << " : " << typeToString(inst->result().type()) << "\n";
  }

  // 4. Outputs in declaration order
  for (const auto* out : outputs_) {
    ss << "  output %" << out->id() << "\n";
  }

  ss << "}\n";
  return ss.str();
}

// -----------------------------------------------------------------------------
// IRBuilder
// -----------------------------------------------------------------------------
IRBuilder::IRBuilder(Module& module) noexcept : module_(module) {}

Value& IRBuilder::input(std::string name, TensorType type) {
  return module_.addInput(std::move(name), std::move(type));
}

Value& IRBuilder::constant(float value) {
  return module_.addConstant(value);
}

Value& IRBuilder::createUnary(Op op, Value& operand, std::string name) {
  return module_.addInstruction(op, {&operand}, operand.type(), std::move(name)).result();
}

static TensorType inferBinaryType(const TensorType& lhs, const TensorType& rhs) {
  if (lhs.dtype != rhs.dtype) {
    throw std::invalid_argument("Binary operands have mismatched data types");
  }
  if (lhs.shape == rhs.shape) {
    return lhs;
  }
  if (lhs.isScalar()) {
    return rhs;
  }
  if (rhs.isScalar()) {
    return lhs;
  }
  // Rank-1 [N] broadcasting to trailing dimension of rank-2 [M, N]
  if (lhs.rank() == 2 && rhs.rank() == 1) {
    if (lhs.shape[1] == rhs.shape[0]) {
      return lhs;
    }
  } else if (lhs.rank() == 1 && rhs.rank() == 2) {
    if (lhs.shape[0] == rhs.shape[1]) {
      return rhs;
    }
  }
  throw std::invalid_argument("Incompatible shapes for binary operation");
}

Value& IRBuilder::createBinary(Op op, Value& lhs, Value& rhs, std::string name) {
  TensorType res_type = inferBinaryType(lhs.type(), rhs.type());
  return module_.addInstruction(op, {&lhs, &rhs}, res_type, std::move(name)).result();
}

Value& IRBuilder::createMatMul(Value& lhs, Value& rhs, std::string name) {
  if (lhs.type().rank() != 2 || rhs.type().rank() != 2) {
    throw std::invalid_argument("MatMul operands must be rank-2");
  }
  if (lhs.type().shape[1] != rhs.type().shape[0]) {
    throw std::invalid_argument("MatMul inner dimension mismatch");
  }
  TensorType res_type{DType::F32, {lhs.type().shape[0], rhs.type().shape[1]}};
  return module_.addInstruction(Op::MatMul, {&lhs, &rhs}, res_type, std::move(name)).result();
}

void IRBuilder::output(Value& value) {
  module_.addOutput(value);
}

} // namespace tinytensor
