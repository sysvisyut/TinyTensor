#pragma once
#include <memory>
#include <string>
#include <vector>

namespace tinytensor {

class Module;

class Pass {
 public:
  virtual ~Pass() = default;
  virtual std::string name() const = 0;
  virtual bool runOnModule(Module& module) = 0; // true iff module changed
};

class PassManager final {
 public:
  explicit PassManager(bool verify_after_each_pass = true) noexcept;
  void addPass(std::unique_ptr<Pass> pass);
  bool run(Module& module); // throws std::runtime_error if enabled verification fails

 private:
  bool verify_after_each_pass_;
  std::vector<std::unique_ptr<Pass>> passes_;
};

} // namespace tinytensor
