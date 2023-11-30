cmake_minimum_required(VERSION 3.1)
project(TEST LANGUAGES CXX)

add_executable(unit-test src/test.cpp)

target_include_directories(unit-test PRIVATE ../DSP/include ../doctest)
target_compile_definitions(unit-test PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(unit-test dsp)