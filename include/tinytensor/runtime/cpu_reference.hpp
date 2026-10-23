#pragma once
#include "tinytensor/ir.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinytensor {

/// Host-side dense buffer holding row-major float data for a specific TensorType.
struct TensorBuffer final {
  TensorType type;
  std::vector<float> data;

  TensorBuffer() = default;
  TensorBuffer(TensorType t, std::vector<float> d);

  static TensorBuffer zeros(TensorType t);
  static TensorBuffer scalar(float val);

  std::size_t numElements() const noexcept;
  bool operator==(const TensorBuffer& other) const noexcept;
  bool operator!=(const TensorBuffer& other) const noexcept { return !(*this == other); }
};

/// Ground-truth CPU reference interpreter for TinyTensor IR.
/// Evaluates all v1 operations with scalar and rank-1 trailing broadcasting.
class CPUReferenceExecutor final {
 public:
  CPUReferenceExecutor() = default;

  /// Executes the module using inputs identified by their declaration names.
  /// Returns output buffers in the exact order of module.outputs().
  std::vector<TensorBuffer> execute(
      const Module& module,
      const std::unordered_map<std::string, TensorBuffer>& inputs) const;

  /// Executes the module using inputs identified by their Value IDs.
  /// Returns output buffers in the exact order of module.outputs().
  std::vector<TensorBuffer> execute(
      const Module& module,
      const std::unordered_map<Value::Id, TensorBuffer>& inputs) const;
};

} // namespace tinytensor
