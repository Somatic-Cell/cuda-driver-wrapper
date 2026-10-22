# =========================================
# File Name : CMakeLists.txt
# Encoding  : UTF-8
# =========================================

cmake_minimum_required(VERSION 3.27)

# このライブラリはホスト側の C++ だけで構成
# 使用する CUDA や GPU のアーキテクチャに関しては，カーネルをビルドする側が設定する
project(
    cuda_driver_wrapper
    VERSION 0.1
    LANGUAGES CXX
)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(${PROJECT_NAME} STATIC
    src/error.cpp
    src/device.cpp
    src/context.cpp
    src/stream.cpp
    src/event.cpp
    src/device_memory.cpp
    src/device_buffer.cpp
    src/read_binary_file.cpp
    src/module.cpp
    src/kernel.cpp
    src/pinned_host_memory.cpp
    src/pinned_host_buffer.cpp
    src/array_2d_memory.cpp
    src/array_2d.cpp
)

add_library(
    cuda_driver_wrapper::cuda_driver_wrapper
    ALIAS ${PROJECT_NAME}
)

if(MSVC)
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            COMPILE_PDB_NAME "${PROJECT_NAME}"
    )
endif()

# このプロジェクトは C++20 の機能を使うため，利用側にも伝える
target_compile_features(${PROJECT_NAME} PUBLIC
    cxx_std_20
)
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# 公開 API には CUresult を含むため， CUDA ヘッダと Driver のリンク依存を公開する
target_link_libraries(
    ${PROJECT_NAME} 
    PUBLIC 
    CUDA::cuda_driver
    Threads::Threads
)

# 警告は，このライブのコンパイルにだけ適用
if(MSVC)
    target_compile_options(${PROJECT_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )   
endif()

# -------------------------
#　test
# -------------------------
# テストは，単体でビルドしたときには ON, add_subdirectory() で取り込まれたときには OFF にする
option(CUDA_DRIVER_WRAPPER_BUILD_TESTS
    "Build cuda_driver_wrapper tests"
    ${PROJECT_IS_TOP_LEVEL}
)


if(CUDA_DRIVER_WRAPPER_BUILD_TESTS)
    option(
        CUDA_DRIVER_WRAPPER_BUILD_KERNEL_TESTS
        "Build kernel tests requiring NVIDIA nvcc"
        ON
    )

    if(CUDA_DRIVER_WRAPPER_BUILD_KERNEL_TESTS)
        enable_language(CUDA)
    endif()

    enable_testing()
    add_subdirectory(tests)
endif()