#include <gtest/gtest.h>
#include "tinytensor/version.hpp"
#include <string_view>
#include <type_traits>

TEST(TinyTensorSanityTest, VersionString) {
  EXPECT_STREQ(tinytensor::getVersion(), "0.1.0");
}

TEST(TinyTensorSanityTest, Cpp17FeaturesAvailable) {
  constexpr std::string_view sv = "tinytensor";
  static_assert(sv.length() == 10, "Compile-time string_view failed");
  EXPECT_EQ(sv.length(), 10u);

  // Structured bindings and type traits check
  auto pair = std::make_pair(42, 3.14f);
  auto [first, second] = pair;
  EXPECT_EQ(first, 42);
  EXPECT_FLOAT_EQ(second, 3.14f);
}
