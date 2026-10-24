add_test([=[TinyTensorSanityTest.VersionString]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_placeholder_test [==[--gtest_filter=TinyTensorSanityTest.VersionString]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TinyTensorSanityTest.VersionString]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/placeholder_test.cpp:6]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[TinyTensorSanityTest.Cpp17FeaturesAvailable]=]  /Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests/tinytensor_placeholder_test [==[--gtest_filter=TinyTensorSanityTest.Cpp17FeaturesAvailable]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TinyTensorSanityTest.Cpp17FeaturesAvailable]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/tests/placeholder_test.cpp:10]==]
    WORKING_DIRECTORY [==[/Users/sharvajvidyutgmail.com/Desktop/files/curriculum/projects/TinyTensor/build-cuda/tests]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(tinytensor_placeholder_test_TESTS [==[TinyTensorSanityTest.VersionString]==] [==[TinyTensorSanityTest.Cpp17FeaturesAvailable]==])
