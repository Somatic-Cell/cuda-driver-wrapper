#include <cuda_driver_wrapper/pinned_host_buffer.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/device_buffer.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/event.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace
{
namespace cdw = cuda_driver_wrapper;

void require(const bool condition, const std::string_view message)
{
    if(!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

template<class Exception, class Function>
void require_throws(Function&& function, const std::string_view message)
{
    try { std::forward<Function>(function)(); }
    catch(const Exception&) { return; }
    throw std::runtime_error(std::string(message));
}

void require_current(const CUcontext expected)
{
    require(cdw::get_current_context() == expected, "The current context was not restored");
}

// テスト用。必ずバッファ・イベント・stream の後に構築する。
// 途中で assertion / CUDA API が失敗しても、転送中のバッファを先に破棄しない。
// 同期すら失敗した場合は安全な解放を仮定せず、そのプロセスを終了する。
class FinishStreamBeforeBuffers final
{
public:
    explicit FinishStreamBeforeBuffers(const cdw::Stream& stream) noexcept : stream_(stream) {}
    ~FinishStreamBeforeBuffers() noexcept
    {
        try { cdw::synchronize_stream(stream_); }
        catch(...)
        {
            std::cerr << "Fatal: test cleanup could not establish stream completion.\n";
            std::terminate();
        }
    }
    FinishStreamBeforeBuffers(const FinishStreamBeforeBuffers&) = delete;
    FinishStreamBeforeBuffers& operator=(const FinishStreamBeforeBuffers&) = delete;

private:
    const cdw::Stream& stream_;
};

// GPU 一台で別 context との共存を確認するための、テスト専用所有型。
class ExternalContext final
{
public:
    explicit ExternalContext(const CUdevice device)
    {
#if CUDA_VERSION >= 13000
        CUctxCreateParams parameters{};
        cdw::check_cuda(cuCtxCreate(&context_, &parameters, 0, device), "cuCtxCreate (test)");
#else
        cdw::check_cuda(cuCtxCreate(&context_, 0, device), "cuCtxCreate (test)");
#endif
        try
        {
            CUcontext popped = nullptr;
            cdw::check_cuda(cuCtxPopCurrent(&popped), "cuCtxPopCurrent (test setup)");
            require(popped == context_, "Unexpected context after cuCtxCreate");
        }
        catch(...)
        {
            static_cast<void>(cdw::report_cuda_cleanup_result(
                cuCtxDestroy(context_), "cuCtxDestroy (test rollback)"
            ));
            throw;
        }
    }
    ~ExternalContext() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(
            cuCtxDestroy(context_), "cuCtxDestroy (test)"
        ));
    }
    ExternalContext(const ExternalContext&) = delete;
    ExternalContext& operator=(const ExternalContext&) = delete;
    [[nodiscard]] CUcontext handle() const noexcept { return context_; }

private:
    CUcontext context_ = nullptr;
};

std::byte pattern(const std::size_t index, const unsigned int seed)
{
    return static_cast<std::byte>((index * 37u + seed) % 256u);
}

void fill_pattern(const std::span<std::byte> bytes, const unsigned int seed)
{
    for(std::size_t i = 0; i < bytes.size(); ++i) { bytes[i] = pattern(i, seed); }
}

void require_pattern(const std::span<const std::byte> bytes, const unsigned int seed)
{
    for(std::size_t i = 0; i < bytes.size(); ++i)
    {
        require(bytes[i] == pattern(i, seed), "Host/device copy changed the byte pattern");
    }
}

void test_empty_and_views(const CUcontext context)
{
    cdw::PinnedHostBuffer empty;
    require(empty.empty() && empty.bytes().empty(), "Default buffer must be empty");
    require(empty.view().data == nullptr && empty.context() == nullptr, "Default metadata must be null");
    require(cdw::subview(empty.view(), 0, 0).data == nullptr, "Empty subview must stay null");
    empty.close();
    empty.close();

    require_throws<std::invalid_argument>([] { cdw::PinnedHostBuffer bad(nullptr, 0); },
        "Even a zero-byte constructor must reject a null context");

    cdw::PinnedHostBuffer zero(context, 0);
    require(zero.context() == context && zero.empty(), "Zero-size buffer must retain only context metadata");
    zero.close();
    require(zero.context() == nullptr, "Closing a zero-size buffer must clear context metadata");

    cdw::PinnedHostBuffer buffer(context, 41);
    const auto whole = buffer.view();
    const auto part = cdw::subview(whole, 7, 13);
    require(part.data == whole.data + 7 && part.size_bytes == 13, "Wrong mutable subview");
    const auto& constant_buffer = buffer;
    const auto constant_part = cdw::subview(constant_buffer.view(), 7, 13);
    require(constant_part.data == part.data, "Const subview must preserve the address");
    const auto tail = cdw::subview(whole, 41, 0);
    require(tail.size_bytes == 0 && tail.data == whole.data + 41, "One-past empty subview must be allowed");

    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::subview(whole, 42, 0)); },
        "Subview past end was accepted");
    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::subview(whole, 40, 2)); },
        "Subview beyond the remaining bytes was accepted");
    require_throws<std::out_of_range>([&]
    {
        static_cast<void>(cdw::subview(whole, 1, (std::numeric_limits<std::size_t>::max)()));
    }, "Subview count overflow was accepted");
    require_throws<std::invalid_argument>([&]
    {
        static_cast<void>(cdw::subview(cdw::PinnedHostMemoryView{context, nullptr, 1}, 0, 1));
    }, "Nonempty null host view was accepted");
    require_throws<std::invalid_argument>([&]
    {
        static_cast<void>(cdw::subview(cdw::PinnedHostMemoryView{nullptr, whole.data, 1}, 0, 1));
    }, "Host address without a context was accepted");
    require_throws<std::overflow_error>([&]
    {
        const cdw::ConstPinnedHostMemoryView bad{
            context, whole.data, (std::numeric_limits<std::size_t>::max)()
        };
        static_cast<void>(cdw::subview(bad, 0, 0));
    }, "Address-range overflow was accepted");
}

void test_allocation_and_move(const CUcontext context)
{
    const CUcontext before = cdw::get_current_context();
    {
        cdw::PinnedHostBuffer source(context, 257);
        require_current(before);
        require(source.size_bytes() == 257 && source.context() == context, "Wrong allocation metadata");
        {
            cdw::ScopedCurrentContext current(context);
            unsigned int flags = 1;
            cdw::check_cuda(cuMemHostGetFlags(&flags, source.bytes().data()), "cuMemHostGetFlags (test)");
            require(flags == 0, "PinnedHostBuffer must allocate with flags=0");
        }
        require_current(before);
        fill_pattern(source.bytes(), 19);
        const auto borrowed = source.view();

        cdw::PinnedHostBuffer moved(std::move(source));
        require(source.empty() && source.context() == nullptr, "Move must empty the source");
        require(moved.view().data == borrowed.data, "Move must not relocate host memory");
        require_pattern(moved.bytes(), 19);
        source.close();

        cdw::PinnedHostBuffer destination(context, 31);
        destination = std::move(moved);
        require(moved.empty(), "Move assignment must empty the source");
        require(destination.view().data == borrowed.data, "Move assignment must preserve the allocation");
        require_pattern(destination.bytes(), 19);
        require_current(before);

        // 別名経由で、自己ムーブを意図的に確認する。
        auto* alias = &destination;
        destination = std::move(*alias);
        require_pattern(destination.bytes(), 19);
        destination.close();
        destination.close();
        require(destination.empty() && destination.context() == nullptr, "Close must clear metadata");
    }
    require_current(before);
}

void test_roundtrip(const CUcontext context, const std::size_t bytes)
{
    const CUcontext before = cdw::get_current_context();
    cdw::PinnedHostBuffer input(context, bytes);
    cdw::PinnedHostBuffer output(context, bytes);
    cdw::DeviceBuffer device(context, bytes);
    cdw::Stream stream(context);
    cdw::Event done(context);
    FinishStreamBeforeBuffers finish(stream);

    for(unsigned int seed : {17u, 89u, 213u})
    {
        fill_pattern(input.bytes(), seed);
        std::fill(output.bytes().begin(), output.bytes().end(), std::byte{0xCC});
        cdw::copy_host_to_device_async(device.view(), input.view(), stream);
        cdw::copy_device_to_host_async(output.view(), device.view(), stream);
        cdw::record_event(done, stream);
        require_current(before);
        // ここでは input を変更せず、output を読まない。
        // 短い転送は既に完了し得るため、query == false は要求しない。
        static_cast<void>(cdw::is_event_ready(done));
        cdw::synchronize_event(done);
        require(cdw::is_event_ready(done), "Synchronized event must be ready");
        require_pattern(output.bytes(), seed);
        require_current(before);
        // 完了確認後だけ、同じ pinned buffer を次の反復で再利用する。
    }
}

void test_partial_copy(const CUcontext context)
{
    constexpr std::size_t total = 512;
    constexpr std::size_t count = 257;
    constexpr std::size_t src_offset = 7;
    constexpr std::size_t dev_offset = 11;
    constexpr std::size_t dst_offset = 13;
    cdw::PinnedHostBuffer input(context, total);
    cdw::PinnedHostBuffer output(context, total);
    cdw::PinnedHostBuffer full_device_copy(context, total);
    cdw::DeviceBuffer device(context, total);
    cdw::Stream stream(context);
    FinishStreamBeforeBuffers finish(stream);

    fill_pattern(input.bytes(), 41);
    std::fill(output.bytes().begin(), output.bytes().end(), std::byte{0xA5});
    cdw::fill_device_bytes_async(device.view(), 0xCD, stream);
    cdw::copy_host_to_device_async(
        cdw::subview(device.view(), dev_offset, count),
        cdw::subview(input.view(), src_offset, count), stream
    );
    cdw::copy_device_to_host_async(
        cdw::subview(output.view(), dst_offset, count),
        cdw::subview(device.view(), dev_offset, count), stream
    );
    cdw::copy_device_to_host_async(full_device_copy.view(), device.view(), stream);
    cdw::synchronize_stream(stream);

    for(std::size_t i = 0; i < total; ++i)
    {
        const auto expected_host = i >= dst_offset && i < dst_offset + count
            ? pattern(src_offset + i - dst_offset, 41) : std::byte{0xA5};
        const auto expected_device = i >= dev_offset && i < dev_offset + count
            ? pattern(src_offset + i - dev_offset, 41) : std::byte{0xCD};
        require(output.bytes()[i] == expected_host, "Partial download damaged host bytes");
        require(full_device_copy.bytes()[i] == expected_device, "Partial upload damaged device bytes");
    }
}

void test_two_streams_and_input_reuse(const CUcontext context)
{
    constexpr std::size_t bytes = 262157;
    cdw::PinnedHostBuffer input(context, bytes);
    cdw::PinnedHostBuffer output(context, bytes);
    cdw::DeviceBuffer first(context, bytes);
    cdw::DeviceBuffer second(context, bytes);
    cdw::Stream upload(context);
    cdw::Stream consumer(context);
    cdw::Event uploaded(context);
    cdw::Event downloaded(context);
    FinishStreamBeforeBuffers finish_upload(upload);
    FinishStreamBeforeBuffers finish_consumer(consumer);

    fill_pattern(input.bytes(), 53);
    cdw::copy_host_to_device_async(first.view(), input.view(), upload);
    cdw::record_event(uploaded, upload);

    cdw::stream_wait_event(consumer, uploaded);
    cdw::copy_device_to_device_async(second.view(), first.view(), consumer);
    cdw::copy_device_to_host_async(output.view(), second.view(), consumer);
    cdw::record_event(downloaded, consumer);

    // input の内容保持が必要なのは HtoD が終わるまで。
    // consumer の完了を待つ前でも、uploaded の完了確認後は input を書き換えられる。
    cdw::synchronize_event(uploaded);
    std::fill(input.bytes().begin(), input.bytes().end(), std::byte{0});

    cdw::synchronize_event(downloaded);
    require_pattern(output.bytes(), 53);
}

void test_typed_payload_bytes(const CUcontext context)
{
    const std::array<float, 5> input{0.25f, -2.0f, 3.5f, 0.0f, 16.0f};
    std::array<float, 5> output{};
    const auto input_bytes = std::as_bytes(std::span{input});
    const auto output_bytes = std::as_writable_bytes(std::span{output});
    const std::size_t bytes = input_bytes.size_bytes();
    cdw::PinnedHostBuffer staging_in(context, bytes);
    cdw::PinnedHostBuffer staging_out(context, bytes);
    cdw::DeviceBuffer device(context, bytes);
    cdw::Stream stream(context);
    FinishStreamBeforeBuffers finish(stream);

    // pinned な生ストレージへ float* を作って間接参照せず、object representation をコピーする。
    std::memcpy(staging_in.bytes().data(), input_bytes.data(), bytes);
    cdw::copy_host_to_device_async(device.view(), staging_in.view(), stream);
    cdw::copy_device_to_host_async(staging_out.view(), device.view(), stream);
    cdw::synchronize_stream(stream);
    std::memcpy(output_bytes.data(), staging_out.bytes().data(), bytes);
    require(std::memcmp(input_bytes.data(), output_bytes.data(), bytes) == 0, "Typed payload bytes changed");
}

void test_rejected_copies(const CUcontext context)
{
    cdw::PinnedHostBuffer host(context, 32);
    cdw::DeviceBuffer device(context, 32);
    cdw::Stream stream(context);
    FinishStreamBeforeBuffers finish(stream);

    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_device_async(device.view(), cdw::subview(host.view(), 0, 31), stream);
    }, "Unequal upload sizes must be rejected");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_device_to_host_async(cdw::subview(host.view(), 0, 31), device.view(), stream);
    }, "Unequal download sizes must be rejected");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_device_async(device.view(), {context, nullptr, 32}, stream);
    }, "Null upload source must be rejected");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_device_to_host_async({context, nullptr, 32}, device.view(), stream);
    }, "Null download destination must be rejected");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_device_async({context, 0, 32}, host.view(), stream);
    }, "Null device destination must be rejected");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_device_to_host_async(host.view(), {context, 0, 32}, stream);
    }, "Null device source must be rejected");

    const cdw::DeviceMemoryView no_device{};
    const cdw::PinnedHostMemoryView no_host{};
    cdw::copy_host_to_device_async(no_device, no_host, stream);
    cdw::copy_device_to_host_async(no_host, no_device, stream);

    cdw::Stream closed(context);
    closed.close();
    require_throws<std::logic_error>([&]
    {
        cdw::copy_host_to_device_async(no_device, no_host, closed);
    }, "Closed stream must be rejected even for an empty upload");
    require_throws<std::logic_error>([&]
    {
        cdw::copy_device_to_host_async(no_host, no_device, closed);
    }, "Closed stream must be rejected even for an empty download");
}

void test_external_context(const CUdevice device_id, const CUcontext primary)
{
    const CUcontext before = cdw::get_current_context();
    ExternalContext other(device_id);
    require_current(before);
    {
        cdw::ScopedCurrentContext current(other.handle());
        test_allocation_and_move(primary);
        test_roundtrip(primary, 257);
        require_current(other.handle());
        current.restore();
    }
    require_current(before);

    cdw::PinnedHostBuffer host_primary(primary, 32);
    cdw::PinnedHostBuffer host_other(other.handle(), 32);
    cdw::DeviceBuffer device_primary(primary, 32);
    cdw::DeviceBuffer device_other(other.handle(), 32);
    cdw::Stream stream(primary);
    FinishStreamBeforeBuffers finish(stream);

    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_device_async(device_primary.view(), host_other.view(), stream);
    }, "M0.7 must reject a different host allocation context");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_device_to_host_async(host_other.view(), device_primary.view(), stream);
    }, "M0.7 must reject a different host allocation context on download");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_device_async(device_other.view(), host_primary.view(), stream);
    }, "Different device context must be rejected on upload");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_device_to_host_async(host_primary.view(), device_other.view(), stream);
    }, "Different device context must be rejected on download");
    require_current(before);
}

} // namespace

int main()
{
    static_assert(!std::is_copy_constructible_v<cdw::PinnedHostBuffer>);
    static_assert(!std::is_copy_assignable_v<cdw::PinnedHostBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::PinnedHostBuffer>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::PinnedHostBuffer>);
    static_assert(std::is_nothrow_destructible_v<cdw::PinnedHostBuffer>);
    static_assert(std::is_convertible_v<cdw::PinnedHostMemoryView, cdw::ConstPinnedHostMemoryView>);
    static_assert(!std::is_convertible_v<cdw::ConstPinnedHostMemoryView, cdw::PinnedHostMemoryView>);
    static_assert(!std::is_convertible_v<std::span<const std::byte>, cdw::ConstPinnedHostMemoryView>);
    static_assert(std::is_same_v<decltype(std::declval<const cdw::PinnedHostBuffer&>().bytes()),
        std::span<const std::byte>>);
    static_assert(std::is_same_v<decltype(std::declval<const cdw::PinnedHostBuffer&>().view()),
        cdw::ConstPinnedHostMemoryView>);

    try
    {
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
        if(count == 0)
        {
            std::cout << "SKIP: no visible CUDA device.\n";
            return 77;
        }

        const CUdevice device = cdw::get_device(0);
        std::cout << "Pinned host memory tests on: " << cdw::get_device_info(device).name << '\n';
        const CUcontext before = cdw::get_current_context();
        {
            cdw::RetainPrimaryContext primary(device);
            const CUcontext context = primary.handle();
            test_empty_and_views(context);
            test_allocation_and_move(context);
            for(std::size_t bytes : {std::size_t{1}, std::size_t{257}, std::size_t{262157}})
            {
                test_roundtrip(context, bytes);
            }
            test_partial_copy(context);
            test_two_streams_and_input_reuse(context);
            test_typed_payload_bytes(context);
            test_rejected_copies(context);
            test_external_context(device, context);
            require_current(before);
        }
        require_current(before);
        std::cout << "All pinned host memory tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
