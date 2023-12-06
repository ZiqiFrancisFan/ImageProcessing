cmake_minimum_required(VERSION 3.1)
project(DSP LANGUAGES CUDA CXX)

add_library(dsp src/dsp.cu src/SignalProcessing.cpp)
add_library(dspptx OBJECT src/dsp.cu)
set_target_properties(dspptx PROPERTIES CUDA_PTX_COMPILATION ON)

target_include_directories(dsp PUBLIC include PUBLIC ../error-handling/include)
target_compile_definitions(dsp PUBLIC $<$<CONFIG:Debug>:DEBUG>)

target_link_libraries(dsp error_handling -lcublas -lcurand -lcusolver)

target_include_directories(dspptx PUBLIC include PUBLIC ../error-handling/include)
target_compile_definitions(dspptx PUBLIC $<$<CONFIG:Debug>:DEBUG>)
# target_compile_definitions(dspptx PUBLIC src-in-ptx)

get_property(target_property TARGET dsp PROPERTY COMPILE_DEFINITIONS)
message(STATUS "compile flags with dsp: ${target_property}")

add_custom_target(genexdebugdsp COMMAND ${CMAKE_COMMAND} -E echo "$<$<CONFIG:Debug>:DEBUG>")

set_directory_properties(PROPERTIES CMAKE_CXX_STANDARD 11)