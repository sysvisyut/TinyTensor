add_test([=[CUDADriverFallbackTest.UnavailableInCpuOnlyBuild]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cuda_integration_test [==[--gtest_filter=CUDADriverFallbackTest.UnavailableInCpuOnlyBuild]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CUDADriverFallbackTest.UnavailableInCpuOnlyBuild]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cuda_integration_test.cpp:24]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CUDAIntegrationTest.VectorAddF32EndToEnd]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cuda_integration_test [==[--gtest_filter=CUDAIntegrationTest.VectorAddF32EndToEnd]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CUDAIntegrationTest.VectorAddF32EndToEnd]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cuda_integration_test.cpp:37]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CUDAIntegrationTest.MalformedPTXThrowsWithJITDiagnostics]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cuda_integration_test [==[--gtest_filter=CUDAIntegrationTest.MalformedPTXThrowsWithJITDiagnostics]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CUDAIntegrationTest.MalformedPTXThrowsWithJITDiagnostics]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cuda_integration_test.cpp:134]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CUDAIntegrationTest.NonexistentKernelNameThrows]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cuda_integration_test [==[--gtest_filter=CUDAIntegrationTest.NonexistentKernelNameThrows]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CUDAIntegrationTest.NonexistentKernelNameThrows]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cuda_integration_test.cpp:159]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(tinytensor_cuda_integration_test_TESTS [==[CUDADriverFallbackTest.UnavailableInCpuOnlyBuild]==] [==[CUDAIntegrationTest.VectorAddF32EndToEnd]==] [==[CUDAIntegrationTest.MalformedPTXThrowsWithJITDiagnostics]==] [==[CUDAIntegrationTest.NonexistentKernelNameThrows]==])
