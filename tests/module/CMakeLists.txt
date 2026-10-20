# M0.6 に固有のテストと GPU module 生成をここにまとめる。
set(READ_TEST cuda_driver_wrapper_read_binary_file_test)
add_executable(${READ_TEST} read_binary_file_test.cpp)
set(HOST_TEST_TARGETS ${READ_TEST})

add_test(NAME cuda_driver_wrapper.read_binary_file COMMAND ${READ_TEST})
set_tests_properties(cuda_driver_wrapper.read_binary_file PROPERTIES LABELS "host;io")

if(CUDA_DRIVER_WRAPPER_BUILD_KERNEL_TESTS)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        message(FATAL_ERROR "The M0.6 fatbin test currently requires NVIDIA nvcc.")
    endif()

    set(KERNEL_MODULE cuda_driver_wrapper_test_kernel_module)
    add_library(${KERNEL_MODULE} OBJECT test_kernels.cu)
    set_target_properties(${KERNEL_MODULE} PROPERTIES
        CUDA_FATBIN_COMPILATION ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_RUNTIME_LIBRARY None
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(${KERNEL_MODULE} PRIVATE --generate-line-info)
    if(MSVC)
        target_compile_options(${KERNEL_MODULE} PRIVATE
            -Xcompiler=/utf-8
            -Xcompiler=/Zc:preprocessor
        )
    endif()

    set(KERNEL_TEST cuda_driver_wrapper_module_kernel_test)
    add_executable(${KERNEL_TEST} module_kernel_test.cpp)
    list(APPEND HOST_TEST_TARGETS ${KERNEL_TEST})

    # module はホストの .obj ではない。リンクせず、先に生成するだけ。
    add_dependencies(${KERNEL_TEST} ${KERNEL_MODULE})
    add_test(
        NAME cuda_driver_wrapper.module_kernel
        COMMAND ${KERNEL_TEST} "$<TARGET_OBJECTS:${KERNEL_MODULE}>"
    )
    # 上の TARGET_OBJECTS は、test_kernels.cu 一つに対応する fatbin 一つ。
    # CTest が構成ごとの実際のパスを引数に渡す。CWD や手動コピーに依存しない。
    set_tests_properties(cuda_driver_wrapper.module_kernel PROPERTIES
        LABELS "gpu;cuda_driver;module;kernel"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
    )
endif()

foreach(TEST_TARGET IN LISTS HOST_TEST_TARGETS)
    target_link_libraries(${TEST_TARGET} PRIVATE cuda_driver_wrapper::cuda_driver_wrapper)
    set_target_properties(${TEST_TARGET} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${TEST_TARGET} PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TEST_TARGET} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()
