add_test([=[CPUReferenceTest.ElementwiseUnaryAndBinary]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.ElementwiseUnaryAndBinary]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.ElementwiseUnaryAndBinary]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:18]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.ActivationReluAndExp]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.ActivationReluAndExp]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.ActivationReluAndExp]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:83]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.ScalarAndRank1TrailingBroadcast]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.ScalarAndRank1TrailingBroadcast]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.ScalarAndRank1TrailingBroadcast]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:110]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.NonSquareMatMulAndBoundaryShapes]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.NonSquareMatMulAndBoundaryShapes]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.NonSquareMatMulAndBoundaryShapes]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:145]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.MatMulBoundary1x1AndVectorShapes]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.MatMulBoundary1x1AndVectorShapes]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.MatMulBoundary1x1AndVectorShapes]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:182]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.NonSquareMatMul17x19x23]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.NonSquareMatMul17x19x23]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.NonSquareMatMul17x19x23]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:200]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.BaselineEndToEndMatMulAddRelu]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.BaselineEndToEndMatMulAddRelu]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.BaselineEndToEndMatMulAddRelu]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:247]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[CPUReferenceTest.FailureCasesMissingOrMismatchedInputs]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_cpu_reference_test [==[--gtest_filter=CPUReferenceTest.FailureCasesMissingOrMismatchedInputs]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[CPUReferenceTest.FailureCasesMissingOrMismatchedInputs]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/cpu_reference_test.cpp:279]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(tinytensor_cpu_reference_test_TESTS [==[CPUReferenceTest.ElementwiseUnaryAndBinary]==] [==[CPUReferenceTest.ActivationReluAndExp]==] [==[CPUReferenceTest.ScalarAndRank1TrailingBroadcast]==] [==[CPUReferenceTest.NonSquareMatMulAndBoundaryShapes]==] [==[CPUReferenceTest.MatMulBoundary1x1AndVectorShapes]==] [==[CPUReferenceTest.NonSquareMatMul17x19x23]==] [==[CPUReferenceTest.BaselineEndToEndMatMulAddRelu]==] [==[CPUReferenceTest.FailureCasesMissingOrMismatchedInputs]==])
