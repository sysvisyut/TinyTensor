#include "tinytensor/passes/fusion.hpp"
#include "tinytensor/ir.hpp"

namespace tinytensor {

bool FusionPass::runOnModule(Module& module) {
  bool changed = false;
  bool local_changed = true;

  while (local_changed) {
    local_changed = false;

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

      local_changed = true;
      changed = true;
      break; // Restart scan over updated instructions
    }
  }

  return changed;
}

} // namespace tinytensor
