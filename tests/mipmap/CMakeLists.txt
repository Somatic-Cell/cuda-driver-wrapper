# M0.10: 格納済み mip chain のレベル取得と tex2DLod の検証。
set(MIPMAP_TEST cuda_driver_wrapper_mipmapped_array_test)
add_executable(${MIPMAP_TEST} mipmapped_array_test.cpp)
target_link_libraries(${MIPMAP_TEST} PRIVATE cuda_driver_wrapper::cuda_driver_wrapper)
set_target_properties(${MIPMAP_TEST} PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(${MIPMAP_TEST} PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${MIPMAP_TEST} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# CUDAの初期化やGPUは不要。寸法計算と、所有しない空の状態を確認する。
add_test(NAME cuda_driver_wrapper.mipmap_metadata COMMAND ${MIPMAP_TEST} --metadata)
set_tests_properties(cuda_driver_wrapper.mipmap_metadata PROPERTIES LABELS "host;mipmap")

if(CUDA_DRIVER_WRAPPER_BUILD_KERNEL_TESTS)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        message(FATAL_ERROR "The M0.10 mipmap kernel test requires NVIDIA nvcc.")
    endif()
    set(MIPMAP_MODULE cuda_driver_wrapper_mipmap_test_module)
    add_library(${MIPMAP_MODULE} OBJECT mipmap_test_kernels.cu)
    set_target_properties(${MIPMAP_MODULE} PROPERTIES
        CUDA_FATBIN_COMPILATION ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RUNTIME_LIBRARY None
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(${MIPMAP_MODULE} PRIVATE --generate-line-info)
    if(MSVC)
        target_compile_options(${MIPMAP_MODULE} PRIVATE
            -Xcompiler=/utf-8
            -Xcompiler=/Zc:preprocessor
        )
    endif()
    # 通常の .obj としてリンクしない。実行時に読む一つの fatbin。
    add_dependencies(${MIPMAP_TEST} ${MIPMAP_MODULE})
    add_test(
        NAME cuda_driver_wrapper.mipmap
        COMMAND ${MIPMAP_TEST} "$<TARGET_OBJECTS:${MIPMAP_MODULE}>"
    )
    set_tests_properties(cuda_driver_wrapper.mipmap PROPERTIES
        LABELS "gpu;cuda_driver;mipmap;texture;kernel"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()
