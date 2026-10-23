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
