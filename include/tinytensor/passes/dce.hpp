#pragma once
#include "tinytensor/pass.hpp"

namespace tinytensor {

/// Dead-Code Elimination (DCE) pass.
/// Traverses straight-line SSA IR and recursively prunes unused instructions
/// that do not contribute to any observable output.
class DCEPass final : public Pass {
 public:
  DCEPass() = default;
  std::string name() const override { return "DCEPass"; }
  bool runOnModule(Module& module) override;
};

} // namespace tinytensor
