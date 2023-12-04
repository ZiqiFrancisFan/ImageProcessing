cmake_minimum_required(VERSION 3.1)
project(error_handling LANGUAGES CUDA CXX)

add_library(error_handling src/error_handling.cpp)
target_include_directories(error_handling PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "CUDA include dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_compile_definitions(error_handling PUBLIC $<$<CONFIG:Debug>:DEBUG>)

set_directory_properties(PROPERTIES CMAKE_CXX_STANDARD 11)