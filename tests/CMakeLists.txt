include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Suppress installation and GMock to keep build fast
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(tinytensor_placeholder_test
  placeholder_test.cpp
)

target_link_libraries(tinytensor_placeholder_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_placeholder_test)

add_executable(tinytensor_ir_test
  ir_test.cpp
)

target_link_libraries(tinytensor_ir_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_ir_test)

add_executable(tinytensor_verifier_test
  verifier_test.cpp
)

target_link_libraries(tinytensor_verifier_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_verifier_test)

add_executable(tinytensor_cpu_reference_test
  cpu_reference_test.cpp
)

target_link_libraries(tinytensor_cpu_reference_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_cpu_reference_test)

add_executable(tinytensor_dce_test
  dce_test.cpp
)

target_link_libraries(tinytensor_dce_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_dce_test)

add_executable(tinytensor_fusion_test
  fusion_test.cpp
)

target_link_libraries(tinytensor_fusion_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_fusion_test)

add_executable(tinytensor_nvptx_smoke_test
  nvptx_smoke_test.cpp
)

target_link_libraries(tinytensor_nvptx_smoke_test
  PRIVATE
    tinytensor
    tinytensor_warnings
    GTest::gtest_main
)

gtest_discover_tests(tinytensor_nvptx_smoke_test)

