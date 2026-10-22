# M0.9 は実際の tex2D 読み出しを検証する。カーネルテスト無効時は登録しない。
if(NOT CUDA_DRIVER_WRAPPER_BUILD_KERNEL_TESTS)
    return()
endif()
if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "The M0.9 texture test requires NVIDIA nvcc.")
endif()

set(TEXTURE_MODULE cuda_driver_wrapper_texture_test_module)
add_library(${TEXTURE_MODULE} OBJECT texture_test_kernels.cu)
set_target_properties(${TEXTURE_MODULE} PROPERTIES
    CUDA_FATBIN_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_RUNTIME_LIBRARY None
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(${TEXTURE_MODULE} PRIVATE --generate-line-info)
if(MSVC)
    target_compile_options(${TEXTURE_MODULE} PRIVATE
        -Xcompiler=/utf-8
        -Xcompiler=/Zc:preprocessor
    )
endif()

set(TEXTURE_TEST cuda_driver_wrapper_texture_object_test)
add_executable(${TEXTURE_TEST} texture_object_test.cpp)
target_link_libraries(${TEXTURE_TEST} PRIVATE cuda_driver_wrapper::cuda_driver_wrapper)
set_target_properties(${TEXTURE_TEST} PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(${TEXTURE_TEST} PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${TEXTURE_TEST} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# fatbin は通常の .obj としてリンクしない。読み込むファイルとして引数で渡す。
add_dependencies(${TEXTURE_TEST} ${TEXTURE_MODULE})
add_test(
    NAME cuda_driver_wrapper.texture_object
    COMMAND ${TEXTURE_TEST} "$<TARGET_OBJECTS:${TEXTURE_MODULE}>"
)
set_tests_properties(cuda_driver_wrapper.texture_object PROPERTIES
    LABELS "gpu;cuda_driver;texture;kernel"
    SKIP_RETURN_CODE 77
    TIMEOUT 120
)
