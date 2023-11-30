cmake_minimum_required(VERSION 3.1)
project(ImageProcessing LANGUAGES CUDA CXX)

add_subdirectory(DSP)
add_subdirectory(unit-test)

add_executable(app src/fundamentals.cu)
target_include_directories(app PRIVATE include)
target_compile_definitions(app PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(app dsp)

set_directory_properties(PROPERTIES CMAKE_CXX_STANDARD 11)

# Debug properties of app
get_property(target_property TARGET app PROPERTY COMPILE_DEFINITIONS)
message(STATUS "compile flags: ${target_property}")

add_custom_target(genexdebug COMMAND ${CMAKE_COMMAND} -E echo "$<CONFIG>, $<$<CONFIG:Debug>:DEBUG>")

enable_testing()

add_test(NAME test COMMAND $<TARGET_FILE:unit-test>)