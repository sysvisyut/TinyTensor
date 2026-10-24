add_test([=[NVPTXSmokeTest.TargetInitializationAndSupport]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_nvptx_smoke_test [==[--gtest_filter=NVPTXSmokeTest.TargetInitializationAndSupport]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NVPTXSmokeTest.TargetInitializationAndSupport]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/nvptx_smoke_test.cpp:30]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[NVPTXSmokeTest.TargetMachineCreationRejectsInvalidTriple]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_nvptx_smoke_test [==[--gtest_filter=NVPTXSmokeTest.TargetMachineCreationRejectsInvalidTriple]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NVPTXSmokeTest.TargetMachineCreationRejectsInvalidTriple]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/nvptx_smoke_test.cpp:37]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[NVPTXSmokeTest.EmitSmokeTestModuleVerification]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_nvptx_smoke_test [==[--gtest_filter=NVPTXSmokeTest.EmitSmokeTestModuleVerification]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NVPTXSmokeTest.EmitSmokeTestModuleVerification]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/nvptx_smoke_test.cpp:42]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[NVPTXSmokeTest.EmitPTXAssemblyVerification]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_nvptx_smoke_test [==[--gtest_filter=NVPTXSmokeTest.EmitPTXAssemblyVerification]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NVPTXSmokeTest.EmitPTXAssemblyVerification]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/nvptx_smoke_test.cpp:88]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[NVPTXSmokeTest.VerifyModuleCatchesDeliberatelyMalformedModule]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_nvptx_smoke_test [==[--gtest_filter=NVPTXSmokeTest.VerifyModuleCatchesDeliberatelyMalformedModule]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NVPTXSmokeTest.VerifyModuleCatchesDeliberatelyMalformedModule]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/nvptx_smoke_test.cpp:124]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[NVPTXSmokeTest.SaveArtifactHelper]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_nvptx_smoke_test [==[--gtest_filter=NVPTXSmokeTest.SaveArtifactHelper]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[NVPTXSmokeTest.SaveArtifactHelper]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/nvptx_smoke_test.cpp:148]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(tinytensor_nvptx_smoke_test_TESTS [==[NVPTXSmokeTest.TargetInitializationAndSupport]==] [==[NVPTXSmokeTest.TargetMachineCreationRejectsInvalidTriple]==] [==[NVPTXSmokeTest.EmitSmokeTestModuleVerification]==] [==[NVPTXSmokeTest.EmitPTXAssemblyVerification]==] [==[NVPTXSmokeTest.VerifyModuleCatchesDeliberatelyMalformedModule]==] [==[NVPTXSmokeTest.SaveArtifactHelper]==])
