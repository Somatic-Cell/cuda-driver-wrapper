#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/device_buffer.hpp>
#include <cuda_driver_wrapper/device_memory.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/event.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
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

void require(const bool condition, const std::string_view message)
{
    if(!condition) { throw std::runtime_error(std::string(message)); }
}

template<class Exception, class Function>
void require_throws(Function&& operation, const std::string_view message)
{
    bool caught = false;
    try { std::forward<Function>(operation)(); }
    catch(const Exception&) { caught = true; }
    require(caught, message);
}

void require_current(const CUcontext expected)
{
    CUcontext actual = nullptr;
    cdw::check_cuda(cuCtxGetCurrent(&actual), "cuCtxGetCurrent (test)");
    require(actual == expected, "operation must preserve the caller's current context");
}

// テストが例外で中断しても、バッファの破棄より先に投入済み処理の完了を試みる。
// ライブラリの自動同期ではなく、テスト側が持つ寿命管理の責任。
class DrainStream final
{
public:
    explicit DrainStream(const cdw::Stream& stream) : stream_(stream) {}
    ~DrainStream() noexcept
    {
        if(!stream_.is_valid()) { return; }
        try { cdw::synchronize_stream(stream_); }
        catch(const std::exception& e) { static_cast<void>(std::fprintf(stderr, "Test cleanup failed: %s\n", e.what())); }
        catch(...) { static_cast<void>(std::fprintf(stderr, "Test cleanup failed.\n")); }
    }
    DrainStream(const DrainStream&) = delete;
    DrainStream& operator=(const DrainStream&) = delete;
private:
    const cdw::Stream& stream_;
};

// 一台の GPU でも別 context との共存を検証する。公開ライブラリの型ではない。
class ExternalContext final
{
public:
    explicit ExternalContext(const CUdevice device)
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

void test_host_contracts()
{
    const std::size_t maximum = (std::numeric_limits<std::size_t>::max)();
    require(cdw::checked_byte_size(0, sizeof(float)) == 0, "zero elements must have zero bytes");
    require(cdw::checked_byte_size(17, sizeof(float)) == 17 * sizeof(float), "byte size must match");
    require(cdw::checked_byte_size(maximum, 1) == maximum, "representable product must not be rejected");
    require_throws<std::length_error>([&] { static_cast<void>(cdw::checked_byte_size(maximum / 8 + 1, 8)); },
                                     "overflow must be rejected before multiplication");
    require_throws<std::invalid_argument>([] { static_cast<void>(cdw::checked_byte_size(1, 0)); },
                                         "zero element size must be rejected");

    cdw::DeviceBuffer empty;
    require(empty.empty() && empty.device_ptr() == 0 && empty.context() == nullptr, "default must be empty");
    empty.close();
    empty.close();
    const auto zero = cdw::subview(empty.view(), 0, 0);
    cdw::copy_host_to_device(zero, {});
    cdw::copy_device_to_host({}, zero);
    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::subview(empty.view(), 1, 0)); },
                                     "offset past an empty view must fail");
    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::subview(empty.view(), 0, maximum)); },
                                     "oversized subview must fail");
    require_throws<std::invalid_argument>([] { cdw::DeviceBuffer invalid(nullptr, 0); },
                                         "null context must fail even for zero bytes");
    require_throws<std::invalid_argument>([] { cdw::DeviceBuffer invalid(nullptr, 16); },
                                         "null context must fail for an allocation");
}

void test_allocation_and_partial_transfer(const CUcontext context)
{
    const std::array<std::uint32_t, 8> input{10, 20, 30, 40, 50, 60, 70, 80};
    auto expected = input;
    std::array<std::uint32_t, 8> output{};
    const auto input_bytes = std::as_bytes(std::span{input});
    cdw::DeviceBuffer buffer(context, input_bytes.size_bytes());
    require(buffer.size_bytes() == input_bytes.size_bytes() && !buffer.empty(), "buffer size must be in bytes");
    require(buffer.context() == context && buffer.device_ptr() != 0, "allocation must preserve context");
    require_current(nullptr);

    {
        cdw::ScopedCurrentContext current(context);
        CUdeviceptr base = 0;
        std::size_t bytes = 0;
        cdw::check_cuda(cuMemGetAddressRange(&base, &bytes, buffer.device_ptr()), "cuMemGetAddressRange (test)");
        require(base == buffer.device_ptr() && bytes >= buffer.size_bytes(), "view must identify real storage");
        current.restore();
    }

    cdw::copy_host_to_device(buffer.view(), std::as_bytes(std::span{input}));
    const std::array<std::uint32_t, 2> replacement{301, 401};
    const auto part = cdw::subview(buffer.view(), 2 * sizeof(std::uint32_t), std::as_bytes(std::span{replacement}).size_bytes());
    require(part.data == buffer.device_ptr() + 2 * sizeof(std::uint32_t), "subview offset is in bytes");
    cdw::copy_host_to_device(part, std::as_bytes(std::span{replacement}));
    expected[2] = replacement[0];
    expected[3] = replacement[1];

    const cdw::DeviceBuffer& read_only = buffer;
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), read_only.view());
    require(output == expected, "partial upload must preserve the surrounding bytes");

    std::array<std::uint32_t, 2> part_output{};
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{part_output}),
                            cdw::subview(read_only.view(), 2 * sizeof(std::uint32_t), std::as_writable_bytes(std::span{part_output}).size_bytes()));
    require(part_output == replacement, "partial download must match");
    require_current(nullptr);
    buffer.close();
    buffer.close();
    require(buffer.empty() && buffer.context() == nullptr && buffer.device_ptr() == 0, "close must clear metadata");
}

void test_stream_ordering(const CUcontext context)
{
    // buffer を stream / guard より前に作り、guard の同期が最初に行われる順序にする。
    constexpr std::size_t count = 4096;
    std::vector<std::uint32_t> input(count);
    for(std::size_t i = 0; i < count; ++i) { input[i] = static_cast<std::uint32_t>(i * 37 + 11); }
    std::vector<std::uint32_t> output(count);
    const auto bytes = cdw::checked_byte_size(count, sizeof(std::uint32_t));
    cdw::DeviceBuffer a(context, bytes);
    cdw::DeviceBuffer b(context, bytes);
    cdw::Stream producer(context);
    cdw::Stream consumer(context);
    cdw::Event ready(context);
    DrainStream drain_producer(producer);
    DrainStream drain_consumer(consumer);

    // upload の正常終了後、独立した non-blocking stream から直ちに読み出してよい。
    cdw::copy_host_to_device(a.view(), std::as_bytes(std::span{input}));
    cdw::copy_device_to_device_async(b.view(), a.view(), consumer);
    cdw::synchronize_stream(consumer);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), b.view());
    require(output == input, "upload must be device-complete before returning");

    // 書込み -> event -> 他 stream でのコピー -> CPU での読戻し。
    cdw::fill_device_bytes_async(a.view(), 0xA5, producer);
    cdw::record_event(ready, producer);
    cdw::stream_wait_event(consumer, ready);
    cdw::copy_device_to_device_async(b.view(), a.view(), consumer);
    cdw::synchronize_stream(consumer);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), b.view());
    require(std::all_of(output.begin(), output.end(), [](std::uint32_t v) { return v == 0xA5A5A5A5u; }),
            "event dependency must order the producer write before the copy");

    // 同じ allocation 内でも、重ならない二つの区間ならコピー可能。
    const auto left = cdw::subview(a.view(), 0, bytes / 2);
    const auto right = cdw::subview(a.view(), bytes / 2, bytes / 2);
    cdw::fill_device_bytes_async(left, 0x3C, producer);
    cdw::copy_device_to_device_async(right, left, producer);
    cdw::synchronize_stream(producer);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), a.view());
    require(std::all_of(output.begin(), output.end(), [](std::uint32_t v) { return v == 0x3C3C3C3Cu; }),
            "non-overlapping subviews must be usable in one stream");
    require_current(nullptr);
}

void test_move_and_empty(const CUcontext context)
{
    cdw::DeviceBuffer zero(context, 0);
    require(zero.empty() && zero.device_ptr() == 0, "zero bytes must not allocate");
    require(zero.context() == context, "zero-sized construction retains the borrowed context value");

    const std::array<float, 4> input{0.25f, 1.5f, -2.0f, 8.0f};
    std::array<float, 4> output{};
    cdw::DeviceBuffer source(context, std::as_bytes(std::span{input}).size_bytes());
    cdw::copy_host_to_device(source.view(), std::as_bytes(std::span{input}));
    const auto borrowed = source.view();
    const CUdeviceptr original = source.device_ptr();

    cdw::DeviceBuffer moved(std::move(source));
    require(source.empty() && source.context() == nullptr, "moved-from owner must be empty");
    require(moved.device_ptr() == original, "move construction must preserve the allocation address");
    source.close();
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), borrowed);
    require(output == input, "a pre-move view remains valid while the new owner is alive");

    cdw::DeviceBuffer destination(context, 7);
    destination = std::move(moved);
    require(moved.empty() && destination.device_ptr() == original, "move assignment must transfer ownership");
    cdw::DeviceBuffer& same_object = destination;
    destination = std::move(same_object);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), destination.view());
    require(output == input, "self move must preserve the allocation");

    cdw::DeviceBuffer empty;
    destination = std::move(empty);
    require(destination.empty() && destination.context() == nullptr, "move from empty must release old storage");
    // borrowed はここでは無効。解放済みアドレスを CUDA API で試しに参照しない。

    cdw::Stream stream(context);
    cdw::fill_device_bytes_async(zero.view(), 0, stream);
    cdw::copy_device_to_device_async(zero.view(), zero.view(), stream);
    cdw::copy_host_to_device(zero.view(), {});
    cdw::copy_device_to_host({}, zero.view());
    zero.close();
    require_current(nullptr);
}

void test_invalid_ranges(const CUcontext context)
{
    cdw::DeviceBuffer buffer(context, 64);
    cdw::Stream stream(context);
    std::array<std::byte, 8> host{};
    const auto maximum = (std::numeric_limits<std::size_t>::max)();

    require_throws<std::invalid_argument>([&] { cdw::copy_host_to_device(buffer.view(), host); },
                                         "copy sizes must match exactly");
    require_throws<std::invalid_argument>([&] { cdw::copy_device_to_host(host, buffer.view()); },
                                         "download sizes must match exactly");
    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::subview(buffer.view(), 65, 0)); },
                                     "subview offset must be bounded");
    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::subview(buffer.view(), 63, maximum)); },
                                     "subview count must be checked without overflow");
    require(cdw::subview(buffer.view(), 64, 0).size_bytes == 0, "empty subview at the end is allowed");

    const auto first = cdw::subview(buffer.view(), 0, 32);
    const auto overlap = cdw::subview(buffer.view(), 1, 32);
    require_throws<std::invalid_argument>([&] { cdw::copy_device_to_device_async(overlap, first, stream); },
                                         "partially overlapping ranges must be rejected");
    cdw::copy_device_to_device_async(first, first, stream); // exact self-copy is a no-op

    const cdw::DeviceMemoryView no_address{context, 0, host.size()};
    require_throws<std::invalid_argument>([&] { cdw::copy_host_to_device(no_address, host); },
                                         "nonempty view with address zero must be rejected");
    const cdw::DeviceMemoryView no_context{nullptr, buffer.device_ptr(), host.size()};
    require_throws<std::invalid_argument>([&] { cdw::copy_host_to_device(no_context, host); },
                                         "nonempty view with context null must be rejected");

    // 表現可能性の検証用メタデータ。CUDA API に渡す前に拒否されることを確認する。
    const cdw::ConstDeviceMemoryView overflow{context, (std::numeric_limits<CUdeviceptr>::max)() - 3, 8};
    require_throws<std::overflow_error>([&] { static_cast<void>(cdw::subview(overflow, 0, 1)); },
                                       "address arithmetic overflow must be rejected");

    stream.close();
    require_throws<std::logic_error>([&] { cdw::fill_device_bytes_async({}, 0, stream); },
                                    "closed stream must not be interpreted as the default stream");
    require_current(nullptr);
}

void test_external_context_and_borrowing(const CUdevice device, const CUcontext primary)
{
    require_current(nullptr);
    {
        ExternalContext external(device); // now current
        cdw::DeviceBuffer buffer(primary, 16);
        cdw::Stream stream(primary);
        cdw::Stream foreign_stream(external.handle());
        cdw::DeviceBuffer foreign_buffer(external.handle(), 16);
        DrainStream drain(stream);
        std::array<std::byte, 16> bytes{};
        bytes.fill(std::byte{0x5A});
        require_current(external.handle());

        cdw::copy_host_to_device(buffer.view(), bytes);
        cdw::fill_device_bytes_async(buffer.view(), 0x2A, stream);
        cdw::synchronize_stream(stream);
        cdw::copy_device_to_host(bytes, buffer.view());
        require(std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0x2A}; }),
                "operations must select the allocation's context");
        require_current(external.handle());

        require_throws<std::invalid_argument>([&] { cdw::fill_device_bytes_async(buffer.view(), 0, foreign_stream); },
                                             "wrong-context stream must be rejected");
        require_throws<std::invalid_argument>([&] { cdw::copy_device_to_device_async(foreign_buffer.view(), buffer.view(), stream); },
                                             "cross-context copy is outside M0.5");
        require_current(external.handle());

        // 生 API で確保した allocation を借用する（所有権は移さない）。
        CUdeviceptr raw = 0;
        cdw::check_cuda(cuMemAlloc(&raw, bytes.size()), "cuMemAlloc (external owner)");
        try
        {
            const cdw::DeviceMemoryView borrowed{external.handle(), raw, bytes.size()};
            cdw::copy_host_to_device(borrowed, bytes);
            std::array<std::byte, 16> copied{};
            cdw::copy_device_to_host(copied, borrowed);
            require(copied == bytes, "a non-owning view must work with external allocations");
            require_current(external.handle());
        }
        catch(...)
        {
            static_cast<void>(cdw::report_cuda_cleanup_result(cuMemFree(raw), "cuMemFree (test cleanup)"));
            throw;
        }
        cdw::check_cuda(cuMemFree(raw), "cuMemFree (external owner)");
        buffer.close();
        require_current(external.handle());
    }
    require_current(nullptr);
}

} // namespace

int main()
{
    static_assert(!std::is_copy_constructible_v<cdw::DeviceBuffer>);
    static_assert(!std::is_copy_assignable_v<cdw::DeviceBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::DeviceBuffer>);
    static_assert(std::is_move_assignable_v<cdw::DeviceBuffer>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::DeviceBuffer>);
    static_assert(std::is_nothrow_destructible_v<cdw::DeviceBuffer>);
    static_assert(std::is_trivially_copyable_v<cdw::DeviceMemoryView>);
    static_assert(std::is_convertible_v<cdw::DeviceMemoryView, cdw::ConstDeviceMemoryView>);
    static_assert(!std::is_convertible_v<cdw::ConstDeviceMemoryView, cdw::DeviceMemoryView>);
    static_assert(std::is_same_v<decltype(std::declval<const cdw::DeviceBuffer&>().view()), cdw::ConstDeviceMemoryView>);

    try
    {
        test_host_contracts();
        int count = 0;
        try { cdw::initialize_driver(); count = cdw::get_device_count(); }
        catch(const cdw::CudaError& error)
        {
            if(error.result() != CUDA_ERROR_NO_DEVICE) { throw; }
            std::cout << "SKIP: no CUDA device.\n";
            return 77;
        }
        if(count == 0) { std::cout << "SKIP: no visible CUDA device.\n"; return 77; }
        const CUdevice device = cdw::get_device(0);
        std::cout << "Device memory tests on: " << cdw::get_device_info(device).name << '\n';
        {
            cdw::RetainPrimaryContext primary(device);
            test_allocation_and_partial_transfer(primary.handle());
            test_stream_ordering(primary.handle());
            test_move_and_empty(primary.handle());
            test_invalid_ranges(primary.handle());
            test_external_context_and_borrowing(device, primary.handle());
        }
        require_current(nullptr);
        std::cout << "All device memory tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
