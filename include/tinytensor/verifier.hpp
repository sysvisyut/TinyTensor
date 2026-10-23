#pragma once
#include <string>

namespace tinytensor {

class Module;

struct VerificationResult final {
  bool ok;
  std::string message;
};

/// Verifies that the given Module satisfies all SSA, typing, shape,
/// dominance, and reciprocal use-def invariants.
VerificationResult verifyModule(const Module& module);

} // namespace tinytensor
