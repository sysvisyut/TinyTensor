#include "tinytensor/passes/dce.hpp"
#include "tinytensor/ir.hpp"

namespace tinytensor {

bool DCEPass::runOnModule(Module& module) {
  bool changed = false;
  bool local_changed = true;

  // Sweeping backwards guarantees that consumers are visited before their producers,
  // naturally collapsing cascades of dead instructions. Loop until a fixpoint is reached.
  while (local_changed) {
    local_changed = false;
    for (std::size_t i = module.instructions().size(); i > 0; --i) {
      Instruction* inst = module.instructions()[i - 1].get();
      if (inst->result().useCount() == 0 && !inst->result().isOutput()) {
        if (module.eraseInstruction(*inst)) {
          local_changed = true;
          changed = true;
        }
      }
    }
  }

  return changed;
}

} // namespace tinytensor
