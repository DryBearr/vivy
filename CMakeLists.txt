cmake_minimum_required(VERSION 3.16)
project(vivy LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

add_executable(vivy main.cpp)
target_compile_features(vivy PRIVATE cxx_std_20)
target_compile_options(vivy PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(vivy PRIVATE ggml)

configure_file(third_party/ggml/LICENSE ${CMAKE_BINARY_DIR}/GGML-LICENSE COPYONLY)
