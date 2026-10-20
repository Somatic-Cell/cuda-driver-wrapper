#include <cuda_driver_wrapper/module.hpp>
#include <cuda_driver_wrapper/kernel.hpp>
#include <cuda_driver_wrapper/device_buffer.hpp>
#include <cuda_driver_wrapper/device_memory.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/event.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace cdw = cuda_driver_wrapper;

void require(bool condition, std::string_view message)
{
    if(!condition) { throw std::runtime_error(std::string(message)); }
}

template<class Exception, class Function>
void require_throws(Function&& function, std::string_view message)
{
    bool caught = false;
    try { function(); }
    catch(const Exception&) { caught = true; }
    require(caught, message);
}

void require_current(CUcontext expected)
{
    CUcontext actual = nullptr;
    cdw::check_cuda(cuCtxGetCurrent(&actual), "cuCtxGetCurrent (test)");
    require(actual == expected, "an operation changed the caller's current context");
}

// テストが途中で失敗しても、外側の buffer/module より先に stream を待つ。
// ラッパーの launch / destructor に自動同期を追加するものではない。
class DrainStream final
{
public:
    explicit DrainStream(const cdw::Stream& stream) noexcept : stream_(stream) {}
    ~DrainStream() noexcept
    {
        try { cdw::synchronize_stream(stream_); }
        catch(const std::exception& error)
        {
            static_cast<void>(std::fprintf(stderr, "[test drain] %s\n", error.what()));
        }
        catch(...) { static_cast<void>(std::fprintf(stderr, "[test drain] unknown failure\n")); }
    }
    DrainStream(const DrainStream&) = delete;
    DrainStream& operator=(const DrainStream&) = delete;
private:
    const cdw::Stream& stream_;
};

// 1 GPU で異なる context との共存を調べる、テスト内だけの所有型。
class ExternalContext final
{
public:
    explicit ExternalContext(CUdevice device)
    {
#if CUDA_VERSION >= 13000
        CUctxCreateParams parameters{};
        cdw::check_cuda(cuCtxCreate(&context_, &parameters, CU_CTX_SCHED_AUTO, device), "cuCtxCreate (test)");
#else
        cdw::check_cuda(cuCtxCreate(&context_, CU_CTX_SCHED_AUTO, device), "cuCtxCreate (test)");
#endif
    }
    ~ExternalContext() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(cuCtxDestroy(context_), "cuCtxDestroy (test)"));
    }
    ExternalContext(const ExternalContext&) = delete;
    ExternalContext& operator=(const ExternalContext&) = delete;
    CUcontext handle() const noexcept { return context_; }
private:
    CUcontext context_ = nullptr;
};

void test_ownership(CUcontext context, const std::filesystem::path& file, CUcontext previous)
{
    cdw::Module empty;
    require(!empty.is_valid(), "default module must be empty");
    empty.close();
    require_throws<std::logic_error>([&] {
        static_cast<void>(cdw::get_kernel(empty, "cdw_noop"));
    }, "lookup on an empty module must fail");
    require_throws<std::invalid_argument>([&] {
        cdw::Module invalid(nullptr, file);
    }, "a null context must fail before loading");
    require_throws<std::invalid_argument>([&] {
        cdw::Module invalid(context, {});
    }, "an empty file path must fail");
    require_current(previous);

    cdw::Module first(context, file);
    require(first.is_valid() && first.context() == context, "module must preserve its context");
    const auto saved = cdw::get_kernel(first, "cdw_noop");
    const CUmodule original = first.handle();
    cdw::Module second(std::move(first));
    require(!first.is_valid() && first.context() == nullptr, "move must empty the source");
    require(second.handle() == original, "move must not reload the module");
    first.close();

    cdw::Module destination(context, file);
    destination = std::move(second);
    require(!second.is_valid() && destination.handle() == original, "move assignment must transfer ownership");
    cdw::Stream stream(context);
    {
        DrainStream drain(stream);
        cdw::launch_kernel(saved, {}, stream, {});
        cdw::synchronize_stream(stream);
    }
    // saved は同じ CUmodule のままなので move 後も使える。Unload 後には使わない。
    destination.close();
    destination.close();
    require_throws<std::logic_error>([&] {
        static_cast<void>(cdw::get_kernel(destination, "cdw_noop"));
    }, "lookup on a closed module must fail");
    require_current(previous);
}

void test_rejections(CUcontext context, const std::filesystem::path& file, CUcontext previous)
{
    cdw::Module module(context, file);
    cdw::Stream stream(context);
    const auto noop = cdw::get_kernel(module, "cdw_noop");
    require_throws<std::invalid_argument>([&] {
        static_cast<void>(cdw::get_kernel(module, ""));
    }, "empty kernel name must fail");
    const std::string embedded("cdw_noop\0suffix", 15);
    require_throws<std::invalid_argument>([&] {
        static_cast<void>(cdw::get_kernel(module, embedded));
    }, "an embedded NUL in a kernel name must fail");

    bool caught = false;
    std::uint_least32_t expected_line = 0;
    const char* expected_file = std::source_location::current().file_name();
    try
    {
        expected_line = std::source_location::current().line() + 1;
        static_cast<void>(cdw::get_kernel(module, "cdw_function_that_does_not_exist"));
    }
    catch(const cdw::CudaError& error)
    {
        caught = true;
        require(error.result() == CUDA_ERROR_NOT_FOUND, "lookup must preserve CUDA_ERROR_NOT_FOUND");
        const std::string expected = std::string(expected_file) + ": " + std::to_string(expected_line);
        require(std::string_view(error.what()).find(expected) != std::string_view::npos,
                "lookup error must preserve the caller's location");
    }
    require(caught, "missing kernel lookup must fail");
    require_current(previous);

    cdw::LaunchConfig zero;
    zero.block_x = 0;
    require_throws<std::invalid_argument>([&] {
        cdw::launch_kernel(noop, zero, stream, {});
    }, "zero block dimension must fail before launch");
    require_throws<std::invalid_argument>([&] {
        cdw::launch_kernel({}, {}, stream, {});
    }, "empty kernel descriptor must fail");
    void* bad_arguments[] = {nullptr};
    require_throws<std::invalid_argument>([&] {
        cdw::launch_kernel(noop, {}, stream, bad_arguments);
    }, "null argument storage must fail before the Driver reads it");
    stream.close();
    require_throws<std::logic_error>([&] {
        cdw::launch_kernel(noop, {}, stream, {});
    }, "a closed stream must fail");
    require_current(previous);
}

// ホスト上の引数領域は cuLaunchKernel の呼び出し中にだけ必要。
// GPU メモリの所有者は、この関数の外で完了まで保持する。
void enqueue_affine(cdw::Kernel kernel, const cdw::LaunchConfig& config,
                    const cdw::Stream& stream, CUdeviceptr data,
                    unsigned int count, unsigned int multiplier, unsigned int increment)
{
    void* arguments[] = {&data, &count, &multiplier, &increment};
    cdw::launch_kernel(kernel, config, stream, arguments);
}

void test_computation(CUcontext context, const std::filesystem::path& file, CUcontext previous)
{
    constexpr unsigned int count = 257;
    constexpr unsigned int block_size = 128;
    constexpr unsigned int blocks = (count + block_size - 1) / block_size;
    constexpr unsigned int guard_value = 0xdeadbeefu;
    static_assert(sizeof(unsigned int) == 4);
    static_assert(sizeof(CUdeviceptr) == sizeof(void*));

    cdw::Module module(context, file);
    cdw::Stream stream(context);
    const auto affine = cdw::get_kernel(module, "cdw_affine_u32");
    const auto reduce = cdw::get_kernel(module, "cdw_block_sum_u32");
    std::vector<unsigned int> input(count + 2, guard_value);
    for(unsigned int i = 0; i < count; ++i) { input[i] = i + 1; }
    std::vector<unsigned int> output(input.size());
    std::vector<unsigned int> sums(blocks);

    cdw::DeviceBuffer data(context, cdw::checked_byte_size(input.size(), sizeof(unsigned int)));
    cdw::DeviceBuffer sum_buffer(context, cdw::checked_byte_size(sums.size(), sizeof(unsigned int)));
    cdw::Event finished(context);
    DrainStream drain(stream); // module / buffer / event より先に完了確認する。

    cdw::copy_host_to_device(data.view(), std::as_bytes(std::span{input}));
    cdw::LaunchConfig config;
    config.grid_x = blocks;
    config.block_x = block_size;
    enqueue_affine(affine, config, stream, data.device_ptr(), count, 3, 7);
    enqueue_affine(affine, config, stream, data.device_ptr(), count, 2, 1);
    require_current(previous);

    config.shared_memory_bytes = block_size * static_cast<unsigned int>(sizeof(unsigned int));
    CUdeviceptr data_argument = data.device_ptr();
    CUdeviceptr sums_argument = sum_buffer.device_ptr();
    unsigned int count_argument = count;
    void* arguments[] = {&data_argument, &sums_argument, &count_argument};
    cdw::launch_kernel(reduce, config, stream, arguments);
    cdw::record_event(finished, stream);
    cdw::synchronize_event(finished);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), data.view());
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{sums}), sum_buffer.view());
    require_current(previous);

    for(unsigned int i = 0; i < count; ++i)
    {
        require(output[i] == input[i] * 6u + 15u, "kernel arithmetic or stream ordering disagrees");
    }
    require(output[count] == guard_value && output[count + 1] == guard_value,
            "threads past count must not overwrite guard values");
    for(unsigned int block = 0; block < blocks; ++block)
    {
        unsigned int expected = 0;
        const auto end = (std::min)(count, (block + 1) * block_size);
        for(unsigned int i = block * block_size; i < end; ++i) { expected += output[i]; }
        require(sums[block] == expected, "dynamic shared memory block sum disagrees");
    }
}

void test_external(CUdevice device, CUcontext primary, const std::filesystem::path& file)
{
    ExternalContext external(device); // この構築で external が current になる。
    require_current(external.handle());
    test_ownership(primary, file, external.handle());
    test_computation(primary, file, external.handle());
    require_current(external.handle());

    cdw::Module module(primary, file);
    cdw::Stream other_stream(external.handle());
    const auto kernel = cdw::get_kernel(module, "cdw_noop");
    require_throws<std::invalid_argument>([&] {
        cdw::launch_kernel(kernel, {}, other_stream, {});
    }, "launch on another context's stream must be rejected");
    require_current(external.handle());
}

} // namespace

int main(int argc, char** argv)
{
    static_assert(!std::is_copy_constructible_v<cdw::Module>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::Module>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::Module>);
    static_assert(std::is_nothrow_destructible_v<cdw::Module>);
    static_assert(std::is_trivially_copyable_v<cdw::Kernel>);
    try
    {
        if(argc != 2) { throw std::invalid_argument("Usage: module_kernel_test <test_kernels.fatbin>"); }
        const std::filesystem::path file(argv[1]);
        require(std::filesystem::is_regular_file(file), "test module artifact is missing");
        int count = 0;
        try
        {
            cdw::initialize_driver();
            count = cdw::get_device_count();
        }
        catch(const cdw::CudaError& error)
        {
            if(error.result() != CUDA_ERROR_NO_DEVICE) { throw; }
            std::cout << "SKIP: no CUDA device.\n";
            return 77;
        }
        if(count == 0) { std::cout << "SKIP: no visible CUDA device.\n"; return 77; }
        const CUdevice device = cdw::get_device(0);
        std::cout << "Module/kernel tests on: " << cdw::get_device_info(device).name << '\n';
        std::cout << "Module file: " << file.string() << '\n';
        {
            cdw::RetainPrimaryContext primary(device);
            require_current(nullptr);
            test_ownership(primary.handle(), file, nullptr);
            test_rejections(primary.handle(), file, nullptr);
            test_computation(primary.handle(), file, nullptr);
            test_external(device, primary.handle(), file);
            require_current(nullptr);
        }
        std::cout << "All module/kernel tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
