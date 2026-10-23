#include "tinytensor/pass.hpp"
#include "tinytensor/verifier.hpp"
#include <stdexcept>

namespace tinytensor {

PassManager::PassManager(bool verify_after_each_pass) noexcept
    : verify_after_each_pass_(verify_after_each_pass) {}

void PassManager::addPass(std::unique_ptr<Pass> pass) {
  if (pass) {
    passes_.push_back(std::move(pass));
  }
}

bool PassManager::run(Module& module) {
  bool any_changed = false;

  // Run initial verification if enabled
  if (verify_after_each_pass_) {
    VerificationResult init_res = verifyModule(module);
    if (!init_res.ok) {
      throw std::runtime_error("PassManager: initial module verification failed: " + init_res.message);
    }
  }

  for (const auto& pass : passes_) {
    bool changed = pass->runOnModule(module);
    if (changed) {
      any_changed = true;
    }
    if (verify_after_each_pass_) {
      VerificationResult res = verifyModule(module);
      if (!res.ok) {
        throw std::runtime_error("PassManager: verification failed after pass '" + pass->name() +
                                 "': " + res.message);
      }
    }
  }

  return any_changed;
}

} // namespace tinytensor
