#pragma once
#include "tinytensor/pass.hpp"

namespace tinytensor {

/// FusionPass implements pattern-based kernel fusion for TinyTensor IR.
/// In v1, it fuses legal matmul + column bias + relu epilogue chains into
/// a single FusedMatMulEpilogue instruction.
class FusionPass final : public Pass {
 public:
  FusionPass() = default;
  std::string name() const override { return "FusionPass"; }
  bool runOnModule(Module& module) override;
};

} // namespace tinytensor
