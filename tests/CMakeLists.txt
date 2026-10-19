set(TEST_NAME "cuda_driver_wrapper_error_test")
add_executable(${TEST_NAME} error_test.cpp)

target_link_libraries(${TEST_NAME}
    PRIVATE
    cuda_driver_wrapper::cuda_driver_wrapper
)

set_target_properties(${TEST_NAME}
    PROPERTIES CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(${TEST_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${TEST_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

add_test(
    NAME cuda_driver_wrapper.error
    COMMAND cuda_driver_wrapper_error_test
)
set_tests_properties(cuda_driver_wrapper.error 
    PROPERTIES 
    LABELS "host;cuda_driver"
)

# M0.2: Driver initialization and device queries
set(DEVICE_TEST_NAME "cuda_driver_wrapper_device_test")

add_executable(${DEVICE_TEST_NAME} device_test.cpp)

target_link_libraries(${DEVICE_TEST_NAME}
    PRIVATE cuda_driver_wrapper::cuda_driver_wrapper
)

set_target_properties(${DEVICE_TEST_NAME}
    PROPERTIES CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(${DEVICE_TEST_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${DEVICE_TEST_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

add_test(
    NAME cuda_driver_wrapper.device_uninitialized
    COMMAND ${DEVICE_TEST_NAME} --uninitialized
)
set_tests_properties(cuda_driver_wrapper.device_uninitialized
    PROPERTIES LABELS "host;cuda_driver"
)

add_test(
    NAME cuda_driver_wrapper.device
    COMMAND ${DEVICE_TEST_NAME}
)
set_tests_properties(cuda_driver_wrapper.device
    PROPERTIES
        LABELS "gpu;cuda_driver"
        SKIP_RETURN_CODE 77
)

# M0.3: Primary context lifetime and scoped current context
set(CONTEXT_TEST_NAME "cuda_driver_wrapper_context_test")

add_executable(${CONTEXT_TEST_NAME} context_test.cpp)

target_link_libraries(${CONTEXT_TEST_NAME}
    PRIVATE cuda_driver_wrapper::cuda_driver_wrapper
)

set_target_properties(${CONTEXT_TEST_NAME}
    PROPERTIES CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(${CONTEXT_TEST_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${CONTEXT_TEST_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

add_test(
    NAME cuda_driver_wrapper.context
    COMMAND ${CONTEXT_TEST_NAME}
)

set_tests_properties(cuda_driver_wrapper.context
    PROPERTIES
        LABELS "gpu;cuda_driver;context"
        SKIP_RETURN_CODE 77
)

# M0.4: Stream and event ownership / synchronization
set(STREAM_EVENT_TEST_NAME "cuda_driver_wrapper_stream_event_test")

add_executable(${STREAM_EVENT_TEST_NAME}
    stream_event_test.cpp
)

target_link_libraries(${STREAM_EVENT_TEST_NAME}
    PRIVATE cuda_driver_wrapper::cuda_driver_wrapper
)

set_target_properties(${STREAM_EVENT_TEST_NAME}
    PROPERTIES CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(${STREAM_EVENT_TEST_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${STREAM_EVENT_TEST_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

add_test(
    NAME cuda_driver_wrapper.stream_event
    COMMAND ${STREAM_EVENT_TEST_NAME}
)

set_tests_properties(cuda_driver_wrapper.stream_event
    PROPERTIES
        LABELS "gpu;cuda_driver;stream;event"
        SKIP_RETURN_CODE 77
)

# M0.5: Device allocation, borrowed views and memory copies
set(DEVICE_BUFFER_TEST_NAME "cuda_driver_wrapper_device_buffer_test")

add_executable(${DEVICE_BUFFER_TEST_NAME}
    device_buffer_test.cpp
)

target_link_libraries(${DEVICE_BUFFER_TEST_NAME}
    PRIVATE cuda_driver_wrapper::cuda_driver_wrapper
)

set_target_properties(${DEVICE_BUFFER_TEST_NAME}
    PROPERTIES CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(${DEVICE_BUFFER_TEST_NAME}
        PRIVATE /W4 /utf-8 /permissive- /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${DEVICE_BUFFER_TEST_NAME}
        PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

add_test(
    NAME cuda_driver_wrapper.device_buffer
    COMMAND ${DEVICE_BUFFER_TEST_NAME}
)

set_tests_properties(cuda_driver_wrapper.device_buffer
    PROPERTIES
        LABELS "gpu;cuda_driver;memory"
        SKIP_RETURN_CODE 77
)