cmake_minimum_required(VERSION 3.20)
project(gemmaedge VERSION 0.1.0 LANGUAGES CXX)

option(GEMMAEDGE_BUILD_TESTS "Build host-side tests" ON)
option(GEMMAEDGE_BUILD_CLI "Build command-line executable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check and enable CUDA optionally
enable_language(CUDA OPTIONAL)

set(GEMMAEDGE_SOURCES
    src/model_format.cpp
    src/expert_cache.cpp
    src/kv_store.cpp
    src/tensor.cpp
    src/gguf.cpp
    src/mapped_file.cpp
    src/tokenizer.cpp
    src/gemma4_model.cpp
    src/attention.cpp
    src/feed_forward.cpp
    src/vision.cpp
    src/generation.cpp
    src/vulkan_backend.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND GEMMAEDGE_SOURCES src/cuda_backend.cu)
    message(STATUS "CUDA compilation enabled.")
else()
    list(APPEND GEMMAEDGE_SOURCES src/cuda_backend_stub.cpp)
    message(STATUS "CUDA compilation disabled, compiling CPU stub.")
endif()

add_library(gemmaedge_core ${GEMMAEDGE_SOURCES})
if(CMAKE_CUDA_COMPILER)
    set_target_properties(gemmaedge_core PROPERTIES CUDA_STANDARD 17)
    target_compile_definitions(gemmaedge_core PRIVATE GEMMAEDGE_USE_CUDA=1)
    target_compile_options(gemmaedge_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_75 --use_fast_math -DNDEBUG>)
endif()
target_include_directories(gemmaedge_core PUBLIC include)
target_compile_definitions(gemmaedge_core PUBLIC
    $<$<PLATFORM_ID:Android>:GEMMAEDGE_ANDROID=1>
    $<$<PLATFORM_ID:Linux>:GEMMAEDGE_LINUX=1>
    $<$<PLATFORM_ID:Windows>:GEMMAEDGE_WINDOWS=1>
)
if(MSVC)
    target_compile_options(gemmaedge_core PRIVATE /W4 /permissive- /arch:AVX2)
elseif(NOT ANDROID AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_options(gemmaedge_core PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma)
else()
    target_compile_options(gemmaedge_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GEMMAEDGE_BUILD_CLI)
    add_executable(gemmaedge src/main.cpp)
    target_link_libraries(gemmaedge PRIVATE gemmaedge_core)
endif()

if(ANDROID)
    add_library(gemmaedge_jni SHARED android/jni.cpp)
    target_link_libraries(gemmaedge_jni PRIVATE gemmaedge_core log android)
endif()

if(GEMMAEDGE_BUILD_TESTS AND NOT ANDROID)
    enable_testing()
    add_executable(gemmaedge_tests tests/core_tests.cpp)
    target_link_libraries(gemmaedge_tests PRIVATE gemmaedge_core)
    add_test(NAME gemmaedge_core_tests COMMAND gemmaedge_tests)
endif()
