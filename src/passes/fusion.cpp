#include "tinytensor/passes/fusion.hpp"
#include "tinytensor/ir.hpp"
#include <algorithm>
#include <vector>

namespace tinytensor {

static bool isPureElementwiseOp(Op op) {
  switch (op) {
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::Div:
    case Op::Max:
    case Op::Relu:
    case Op::Exp:
    case Op::Neg:
    case Op::FusedElementwise:
      return true;
    default:
      return false;
  }
}

static bool isMatMulEpilogueAdd(const Instruction* inst) {
  if (inst->op() != Op::Add) {
    return false;
  }
  for (const Value* opnd : inst->operands()) {
    if (opnd && opnd->definingInst() && opnd->definingInst()->op() == Op::MatMul) {
      return true;
    }
  }
  return false;
}

static bool tryFuseMatMulEpilogue(Module& module) {
  for (const auto& inst : module.instructions()) {
    if (inst->op() != Op::Relu) {
      continue;
    }
    Instruction* relu_inst = inst.get();
    Value* sum_val = relu_inst->operands()[0];
    Instruction* add_inst = sum_val->definingInst();

    // Intermediate sum must be an Add, have exactly one user, and not be an output
    if (!add_inst || add_inst->op() != Op::Add) {
      continue;
    }
    if (sum_val->useCount() != 1 || sum_val->isOutput()) {
      continue;
    }

    // Check operands of Add for MatMul and Bias
    Value* matmul_val = nullptr;
    Value* bias_val = nullptr;

    Value* op0 = add_inst->operands()[0];
    Value* op1 = add_inst->operands()[1];

    if (op0->definingInst() && op0->definingInst()->op() == Op::MatMul) {
      matmul_val = op0;
      bias_val = op1;
    } else if (op1->definingInst() && op1->definingInst()->op() == Op::MatMul) {
      matmul_val = op1;
      bias_val = op0;
    } else {
      continue;
    }

    Instruction* matmul_inst = matmul_val->definingInst();

    // Intermediate MatMul must have exactly one user and not be an output
    if (matmul_val->useCount() != 1 || matmul_val->isOutput()) {
      continue;
    }

    // Verify tensor ranks and dimension compatibility
    Value* A = matmul_inst->operands()[0];
    Value* B = matmul_inst->operands()[1];

    if (A->type().rank() != 2 || B->type().rank() != 2) {
      continue;
    }
    if (A->type().shape[1] != B->type().shape[0]) {
      continue;
    }
    if (bias_val->type().rank() != 1 || bias_val->type().shape[0] != B->type().shape[1]) {
      continue;
    }

    // Legality confirmed: Insert fused instruction directly at the MatMul location
    // to preserve strict straight-line definition-before-use dominance.
    TensorType res_type = relu_inst->result().type();
    Instruction& fused_inst = module.insertInstructionBefore(
        *matmul_inst, Op::FusedMatMulEpilogue, {A, B, bias_val}, res_type);

    // Rewire all consumers and outputs of relu_inst to the new fused instruction
    module.replaceAllUsesWith(relu_inst->result(), fused_inst.result());

    // Erase dead unfused instructions in reverse dependency order
    module.eraseInstruction(*relu_inst);
    module.eraseInstruction(*add_inst);
    module.eraseInstruction(*matmul_inst);

    return true;
  }
  return false;
}

static bool tryFuseElementwise(Module& module) {
  for (const auto& inst : module.instructions()) {
    Instruction* C = inst.get();
    if (!isPureElementwiseOp(C->op()) || isMatMulEpilogueAdd(C)) {
      continue;
    }

    for (Value* opnd : C->operands()) {
      if (!opnd) {
        continue;
      }
      Instruction* P = opnd->definingInst();
      if (!P || !isPureElementwiseOp(P->op()) || isMatMulEpilogueAdd(P)) {
        continue;
      }

      // Legality: P must not be an observable output
      if (P->result().isOutput()) {
        continue;
      }

      // Legality: all uses of P must be inside C
      std::size_t uses_by_c = 0;
      for (const Value* c_opnd : C->operands()) {
        if (c_opnd == &P->result()) {
          uses_by_c++;
        }
      }
      if (P->result().useCount() != uses_by_c) {
        continue;
      }

      // Shape check: P and C must have identical shape
      if (P->result().type().shape != C->result().type().shape) {
        continue;
      }

      // Build FusionGroup for P
      FusionGroup fg_P;
      if (P->op() == Op::FusedElementwise) {
        const FusionGroup* existing = module.getFusionGroup(P);
        if (!existing) {
          continue;
        }
        fg_P = *existing;
      } else {
        fg_P.external_inputs = P->operands();
        FusedElementwiseOp iop;
        iop.op = P->op();
        for (std::size_t i = 0; i < P->operands().size(); ++i) {
          iop.operand_indices.push_back(i);
        }
        fg_P.internal_ops.push_back(iop);
      }

      // Build FusionGroup for C
      FusionGroup fg_C;
      if (C->op() == Op::FusedElementwise) {
        const FusionGroup* existing = module.getFusionGroup(C);
        if (!existing) {
          continue;
        }
        fg_C = *existing;
      } else {
        fg_C.external_inputs = C->operands();
        FusedElementwiseOp iop;
        iop.op = C->op();
        for (std::size_t i = 0; i < C->operands().size(); ++i) {
          iop.operand_indices.push_back(i);
        }
        fg_C.internal_ops.push_back(iop);
      }

      // Merge fg_P and fg_C into merged FusionGroup
      FusionGroup merged;
      merged.external_inputs = fg_P.external_inputs;
      for (Value* v : fg_C.external_inputs) {
        if (v == &P->result()) {
          continue;
        }
        auto it = std::find(merged.external_inputs.begin(), merged.external_inputs.end(), v);
        if (it == merged.external_inputs.end()) {
          merged.external_inputs.push_back(v);
        }
      }

      std::size_t num_ext = merged.external_inputs.size();

      // Remap nodes of fg_P
      std::vector<std::size_t> map_P(fg_P.external_inputs.size() + fg_P.internal_ops.size());
      for (std::size_t i = 0; i < fg_P.external_inputs.size(); ++i) {
        auto it = std::find(merged.external_inputs.begin(), merged.external_inputs.end(), fg_P.external_inputs[i]);
        map_P[i] = static_cast<std::size_t>(std::distance(merged.external_inputs.begin(), it));
      }
      for (std::size_t j = 0; j < fg_P.internal_ops.size(); ++j) {
        std::size_t new_node = num_ext + j;
        map_P[fg_P.external_inputs.size() + j] = new_node;
        FusedElementwiseOp new_iop;
        new_iop.op = fg_P.internal_ops[j].op;
        for (std::size_t opnd_idx : fg_P.internal_ops[j].operand_indices) {
          new_iop.operand_indices.push_back(map_P[opnd_idx]);
        }
        merged.internal_ops.push_back(new_iop);
      }
      std::size_t p_result_node = map_P[fg_P.external_inputs.size() + fg_P.internal_ops.size() - 1];

      // Remap nodes of fg_C
      std::vector<std::size_t> map_C(fg_C.external_inputs.size() + fg_C.internal_ops.size());
      for (std::size_t i = 0; i < fg_C.external_inputs.size(); ++i) {
        if (fg_C.external_inputs[i] == &P->result()) {
          map_C[i] = p_result_node;
        } else {
          auto it = std::find(merged.external_inputs.begin(), merged.external_inputs.end(), fg_C.external_inputs[i]);
          map_C[i] = static_cast<std::size_t>(std::distance(merged.external_inputs.begin(), it));
        }
      }
      for (std::size_t j = 0; j < fg_C.internal_ops.size(); ++j) {
        std::size_t new_node = num_ext + fg_P.internal_ops.size() + j;
        map_C[fg_C.external_inputs.size() + j] = new_node;
        FusedElementwiseOp new_iop;
        new_iop.op = fg_C.internal_ops[j].op;
        for (std::size_t opnd_idx : fg_C.internal_ops[j].operand_indices) {
          new_iop.operand_indices.push_back(map_C[opnd_idx]);
        }
        merged.internal_ops.push_back(new_iop);
      }

      // Insert fused instruction before C to preserve straight-line dominance
      Instruction& fused_inst = module.insertInstructionBefore(
          *C, Op::FusedElementwise, merged.external_inputs, C->result().type());
      module.setFusionGroup(&fused_inst, std::move(merged));

      module.replaceAllUsesWith(C->result(), fused_inst.result());
      module.eraseInstruction(*C);
      module.eraseInstruction(*P);

      return true;
    }
  }
  return false;
}

bool FusionPass::runOnModule(Module& module) {
  bool changed = false;
  bool local_changed = true;

  while (local_changed) {
    local_changed = false;

    if (tryFuseMatMulEpilogue(module)) {
      local_changed = true;
      changed = true;
      continue;
    }

    if (tryFuseElementwise(module)) {
      local_changed = true;
      changed = true;
      continue;
    }
  }

  return changed;
}

} // namespace tinytensor
