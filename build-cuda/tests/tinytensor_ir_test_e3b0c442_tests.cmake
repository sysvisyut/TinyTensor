add_test([=[IRTest.BaselineMatMulAddReluAndCanonicalPrint]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.BaselineMatMulAddReluAndCanonicalPrint]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.BaselineMatMulAddReluAndCanonicalPrint]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:6]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[IRTest.EraseInstructionPreservesLiveIDsAndReciprocalLinks]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.EraseInstructionPreservesLiveIDsAndReciprocalLinks]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.EraseInstructionPreservesLiveIDsAndReciprocalLinks]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:94]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[IRTest.EraseSafetyRejectsLiveOrOutputInstructions]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.EraseSafetyRejectsLiveOrOutputInstructions]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.EraseSafetyRejectsLiveOrOutputInstructions]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:141]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[IRTest.ReplaceAllUsesWithUpdatesOperandsAndOutputs]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.ReplaceAllUsesWithUpdatesOperandsAndOutputs]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.ReplaceAllUsesWithUpdatesOperandsAndOutputs]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:159]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[IRTest.OperandMultiplicityAndSetOperand]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.OperandMultiplicityAndSetOperand]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.OperandMultiplicityAndSetOperand]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:190]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[IRTest.InvalidShapeAndMatMulChecks]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.InvalidShapeAndMatMulChecks]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.InvalidShapeAndMatMulChecks]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:216]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[IRTest.ConstantsAndScalarBroadcasting]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_ir_test [==[--gtest_filter=IRTest.ConstantsAndScalarBroadcasting]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[IRTest.ConstantsAndScalarBroadcasting]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/ir_test.cpp:235]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(tinytensor_ir_test_TESTS [==[IRTest.BaselineMatMulAddReluAndCanonicalPrint]==] [==[IRTest.EraseInstructionPreservesLiveIDsAndReciprocalLinks]==] [==[IRTest.EraseSafetyRejectsLiveOrOutputInstructions]==] [==[IRTest.ReplaceAllUsesWithUpdatesOperandsAndOutputs]==] [==[IRTest.OperandMultiplicityAndSetOperand]==] [==[IRTest.InvalidShapeAndMatMulChecks]==] [==[IRTest.ConstantsAndScalarBroadcasting]==])
