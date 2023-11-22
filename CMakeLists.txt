cmake_minimum_required(VERSION 3.1)
project(ImageProcessing LANGUAGES CUDA CXX)

add_executable(app src/fundamentals.cu)
target_include_directories(app PRIVATE include)
target_compile_definitions(app PRIVATE $<$<CONFIG:Debug>:DEBUG>)

# Debug properties of app
get_property(target_property TARGET app PROPERTY COMPILE_DEFINITIONS)
message(STATUS "compile flags: ${target_property}")