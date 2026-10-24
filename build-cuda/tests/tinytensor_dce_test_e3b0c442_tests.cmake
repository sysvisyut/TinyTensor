add_test([=[DCETest.CanonicalBaselineDCEAndSemanticEquivalence]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_dce_test [==[--gtest_filter=DCETest.CanonicalBaselineDCEAndSemanticEquivalence]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DCETest.CanonicalBaselineDCEAndSemanticEquivalence]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/dce_test.cpp:9]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[DCETest.DeadChainRemoval]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_dce_test [==[--gtest_filter=DCETest.DeadChainRemoval]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DCETest.DeadChainRemoval]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/dce_test.cpp:84]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[DCETest.MultipleIndependentLiveOutputs]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_dce_test [==[--gtest_filter=DCETest.MultipleIndependentLiveOutputs]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DCETest.MultipleIndependentLiveOutputs]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/dce_test.cpp:121]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[DCETest.RetainSharedProducerWithLiveConsumer]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_dce_test [==[--gtest_filter=DCETest.RetainSharedProducerWithLiveConsumer]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DCETest.RetainSharedProducerWithLiveConsumer]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/dce_test.cpp:160]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[DCETest.NoOpIdempotencyAndConvergence]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_dce_test [==[--gtest_filter=DCETest.NoOpIdempotencyAndConvergence]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DCETest.NoOpIdempotencyAndConvergence]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/dce_test.cpp:190]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[DCETest.PassManagerRunsDCEAndCatchesCorruptPass]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_dce_test [==[--gtest_filter=DCETest.PassManagerRunsDCEAndCatchesCorruptPass]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[DCETest.PassManagerRunsDCEAndCatchesCorruptPass]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/dce_test.cpp:208]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(tinytensor_dce_test_TESTS [==[DCETest.CanonicalBaselineDCEAndSemanticEquivalence]==] [==[DCETest.DeadChainRemoval]==] [==[DCETest.MultipleIndependentLiveOutputs]==] [==[DCETest.RetainSharedProducerWithLiveConsumer]==] [==[DCETest.NoOpIdempotencyAndConvergence]==] [==[DCETest.PassManagerRunsDCEAndCatchesCorruptPass]==])
