cmake_minimum_required(VERSION 3.1)
project(TEST LANGUAGES CXX CUDA)

add_executable(unit-test src/test.cu)

add_library(unit-test-ptx OBJECT src/test.cu)
set_property(TARGET unit-test-ptx PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(unit-test PRIVATE ../DSP/include ../doctest)
target_compile_definitions(unit-test PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_include_directories(unit-test-ptx PUBLIC ../DSP/include ../doctest ../error-handling/include)
target_compile_definitions(unit-test-ptx PUBLIC $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(unit-test dsp)