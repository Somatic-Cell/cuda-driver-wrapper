#include <cuda_driver_wrapper/array_2d.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/device_buffer.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/event.hpp>
#include <cuda_driver_wrapper/pinned_host_buffer.hpp>
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
#include <vector>

namespace
{
namespace cdw = cuda_driver_wrapper;

void require(const bool condition, const std::string_view message)
{
    if(!condition) { throw std::runtime_error(std::string(message)); }
}

template<class Exception, class Function>
void require_throws(Function&& function, const std::string_view message)
{
    try { std::forward<Function>(function)(); }
    catch(const Exception&) { return; }
    throw std::runtime_error(std::string(message));
}

void require_current(const CUcontext context)
{
    require(cdw::get_current_context() == context, "Current context was not restored");
}

void require_equal(const std::span<const std::byte> actual,
                   const std::span<const std::byte> expected, const std::string_view message)
{
    require(actual.size() == expected.size(), "Comparison sizes differ");
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        if(actual[i] != expected[i])
        {
            throw std::runtime_error(std::string(message) + " at byte " + std::to_string(i));
        }
    }
}

void fill_pattern(const std::span<std::byte> data)
{
    for(std::size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<std::byte>((i * 37 + 19) % 256);
    }
}

// 非同期転送より後に構築するのではなく、投入前かつ資源の宣言より後に置く。
// 途中で検証に失敗しても、転送中のメモリ/配列を先に破棄しない。
class FinishStream final
{
public:
    explicit FinishStream(const cdw::Stream& stream) noexcept : stream_(stream) {}
    ~FinishStream() noexcept
    {
        try { cdw::synchronize_stream(stream_); }
        catch(...)
        {
            std::cerr << "Fatal: test cleanup could not establish GPU completion.\n";
            std::terminate();
        }
    }
    FinishStream(const FinishStream&) = delete;
    FinishStream& operator=(const FinishStream&) = delete;
private:
    const cdw::Stream& stream_;
};

// 独立した context を借用するテスト用。公開ライブラリの API ではない。
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
            require(popped == context_, "Unexpected external context");
        }
        catch(...)
        {
            static_cast<void>(cdw::report_cuda_cleanup_result(cuCtxDestroy(context_), "cuCtxDestroy (rollback)"));
            throw;
        }
    }
    ~ExternalContext() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(cuCtxDestroy(context_), "cuCtxDestroy (test)"));
    }
    ExternalContext(const ExternalContext&) = delete;
    ExternalContext& operator=(const ExternalContext&) = delete;
    [[nodiscard]] CUcontext handle() const noexcept { return context_; }
private:
    CUcontext context_ = nullptr;
};

void test_metadata()
{
    require(cdw::texel_size_bytes(CU_AD_FORMAT_UNSIGNED_INT8, 4) == 4, "RGBA8 size");
    require(cdw::texel_size_bytes(CU_AD_FORMAT_FLOAT, 4) == 16, "RGBA32F size");
    require(cdw::required_2d_bytes(28, 3, 32) == 92, "Footprint must exclude final padding");
    require(cdw::required_2d_bytes(28, 1, 100) == 28, "One-row footprint");
    require(cdw::required_2d_bytes(0, 5, 0) == 0, "Empty width footprint");
    require(cdw::required_2d_bytes(28, 0, 0) == 0, "Empty height footprint");
    require_throws<std::invalid_argument>([] { static_cast<void>(cdw::required_2d_bytes(28, 2, 27)); },
        "Short pitch accepted");
    require_throws<std::length_error>([]
    {
        static_cast<void>(cdw::required_2d_bytes(1, 2, (std::numeric_limits<std::size_t>::max)()));
    }, "Footprint overflow accepted");
    require_throws<std::length_error>([]
    {
        static_cast<void>(cdw::row_size_bytes({(std::numeric_limits<std::size_t>::max)(), 2,
            CU_AD_FORMAT_FLOAT, 4}));
    }, "Descriptor overflow accepted");
    require_throws<std::invalid_argument>([]
    {
        static_cast<void>(cdw::texel_size_bytes(CU_AD_FORMAT_UNSIGNED_INT8, 3));
    }, "Three-channel array accepted");
    require_throws<std::invalid_argument>([]
    {
        static_cast<void>(cdw::texel_size_bytes(CU_AD_FORMAT_HALF, 1));
    }, "Unsupported format accepted");
}

void test_format_roundtrips(const CUcontext context, const cdw::Array2DDescriptor descriptor)
{
    const auto before = cdw::get_current_context();
    {
        cdw::Array2D image(context, descriptor);
        require_current(before);
        {
            cdw::ScopedCurrentContext current(context);
            CUDA_ARRAY_DESCRIPTOR actual{};
            cdw::check_cuda(cuArrayGetDescriptor(&actual, image.handle()), "cuArrayGetDescriptor (test)");
            require(actual.Width == descriptor.width && actual.Height == descriptor.height &&
                    actual.Format == descriptor.format && actual.NumChannels == descriptor.num_channels,
                    "Array descriptor differs from requested descriptor");
        }
        require_current(before);
        const auto row = cdw::row_size_bytes(descriptor);
        const auto source_pitch = row + 5;
        const auto destination_pitch = row + 11;
        const auto source_size = cdw::required_2d_bytes(row, descriptor.height, source_pitch);
        const auto destination_size = cdw::required_2d_bytes(row, descriptor.height, destination_pitch);
        constexpr std::size_t source_prefix = 7, destination_prefix = 13, suffix = 17;
        std::vector<std::byte> source(source_prefix + source_size + suffix);
        fill_pattern(source);
        const auto original_source = source;
        std::vector<std::byte> destination(destination_prefix + destination_size + suffix, std::byte{0xCD});
        auto expected = destination;
        for(std::size_t y = 0; y < descriptor.height; ++y)
        {
            std::memcpy(expected.data() + destination_prefix + y * destination_pitch,
                        source.data() + source_prefix + y * source_pitch, row);
        }
        // 渡す span は最終行末の padding を含まない最小サイズ。
        cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()),
            std::span<const std::byte>{source}.subspan(source_prefix, source_size), source_pitch);
        const auto& const_image = image;
        cdw::copy_array_to_host(std::span{destination}.subspan(destination_prefix, destination_size),
            destination_pitch, const_image.view(), cdw::make_full_array_region(const_image.view()));
        require_equal(destination, expected, "Synchronous 2D transfer changed pixels or padding");
        require_equal(source, original_source, "Upload modified its input");
        require_current(before);

        // 同じ形式・行間隔を pinned memory の非同期転送でも検証する。
        cdw::PinnedHostBuffer pinned_source(context, source.size());
        cdw::PinnedHostBuffer pinned_destination(context, destination.size());
        std::memcpy(pinned_source.bytes().data(), source.data(), source.size());
        std::fill(pinned_destination.bytes().begin(), pinned_destination.bytes().end(), std::byte{0xCD});
        cdw::Stream stream(context);
        FinishStream finish(stream);
        cdw::copy_host_to_array_async(image.view(), cdw::make_full_array_region(image.view()),
            cdw::subview(pinned_source.view(), source_prefix, source_size), source_pitch, stream);
        cdw::copy_array_to_host_async(
            cdw::subview(pinned_destination.view(), destination_prefix, destination_size),
            destination_pitch, image.view(), cdw::make_full_array_region(image.view()), stream);
        cdw::synchronize_stream(stream);
        require_equal(pinned_destination.bytes(), expected, "Async format roundtrip failed");
        require_equal(pinned_source.bytes(), original_source, "Async upload modified input");
        require_current(before);
    }
    require_current(before);
}

void test_ownership(const CUcontext context)
{
    const auto before = cdw::get_current_context();
    cdw::Array2D empty;
    require(!empty.is_valid() && empty.context() == nullptr, "Default array is not empty");
    empty.close(); empty.close();
    require_throws<std::invalid_argument>([&] { static_cast<void>(cdw::make_full_array_region(empty.view())); },
        "Empty array view accepted");
    require_throws<std::invalid_argument>([] { cdw::Array2D bad(nullptr, {1, 1}); }, "Null context accepted");
    require_throws<std::invalid_argument>([&] { cdw::Array2D bad(context, {0, 1}); }, "Zero width accepted");
    require_throws<std::invalid_argument>([&] { cdw::Array2D bad(context, {1, 0}); }, "Zero height accepted");

    cdw::Array2D source(context, {5, 3, CU_AD_FORMAT_UNSIGNED_INT8, 2});
    std::vector<std::byte> input(30);
    fill_pattern(input);
    cdw::copy_host_to_array(source.view(), cdw::make_full_array_region(source.view()), input, 10);
    const auto borrowed = source.view();
    cdw::Array2D moved(std::move(source));
    require(!source.is_valid() && source.context() == nullptr && source.descriptor().width == 0,
        "Move did not clear the source");
    require(moved.handle() == borrowed.array, "Move changed the underlying array");
    cdw::Array2D destination(context, {2, 2});
    destination = std::move(moved);
    require(!moved.is_valid(), "Move assignment did not clear source");
    auto* alias = &destination;
    destination = std::move(*alias);
    std::vector<std::byte> output(30);
    cdw::copy_array_to_host(output, 10, borrowed, cdw::make_full_array_region(borrowed));
    require_equal(output, input, "Move damaged array contents or borrowed view");
    destination.close(); destination.close();
    require(!destination.is_valid() && destination.descriptor().height == 0, "Close did not clear metadata");
    require_current(before);
}

void test_partial_async(const CUcontext context)
{
    const auto before = cdw::get_current_context();
    constexpr std::size_t texel = 4, input_pitch = 23, output_pitch = 53;
    const cdw::ArrayRegion2D source_region{2, 1, 4, 3};
    const cdw::ArrayRegion2D destination_region{3, 2, 4, 3};
    cdw::Array2D first(context, {9, 6, CU_AD_FORMAT_UNSIGNED_INT8, 4});
    cdw::Array2D second(context, {11, 8, CU_AD_FORMAT_UNSIGNED_INT8, 4});
    std::vector<std::byte> initial_first(9 * 6 * texel, std::byte{0x22});
    std::vector<std::byte> initial_second(11 * 8 * texel, std::byte{0x33});
    cdw::copy_host_to_array(first.view(), cdw::make_full_array_region(first.view()), initial_first, 9 * texel);
    cdw::copy_host_to_array(second.view(), cdw::make_full_array_region(second.view()), initial_second, 11 * texel);

    constexpr std::size_t prefix = 7, suffix = 13;
    const auto input_size = cdw::required_2d_bytes(4 * texel, 3, input_pitch);
    const auto output_size = cdw::required_2d_bytes(11 * texel, 8, output_pitch);
    cdw::PinnedHostBuffer input(context, prefix + input_size + suffix);
    cdw::PinnedHostBuffer output(context, prefix + output_size + suffix);
    fill_pattern(input.bytes());
    std::fill(output.bytes().begin(), output.bytes().end(), std::byte{0xA5});
    std::vector<std::byte> expected_first = initial_first;
    std::vector<std::byte> expected_second = initial_second;
    for(std::size_t y = 0; y < 3; ++y)
    {
        std::memcpy(expected_first.data() + ((y + 1) * 9 + 2) * texel,
                    input.bytes().data() + prefix + y * input_pitch, 4 * texel);
        std::memcpy(expected_second.data() + ((y + 2) * 11 + 3) * texel,
                    input.bytes().data() + prefix + y * input_pitch, 4 * texel);
    }
    std::vector<std::byte> expected_output(output.size_bytes(), std::byte{0xA5});
    for(std::size_t y = 0; y < 8; ++y)
    {
        std::memcpy(expected_output.data() + prefix + y * output_pitch,
                    expected_second.data() + y * 11 * texel, 11 * texel);
    }
    cdw::Stream producer(context), consumer(context);
    cdw::Event uploaded(context), downloaded(context);
    FinishStream finish_producer(producer);
    FinishStream finish_consumer(consumer);
    cdw::copy_host_to_array_async(first.view(), source_region,
        cdw::subview(input.view(), prefix, input_size), input_pitch, producer);
    cdw::record_event(uploaded, producer);
    cdw::stream_wait_event(consumer, uploaded);
    cdw::copy_array_to_array_async(second.view(), destination_region, first.view(), source_region, consumer);
    cdw::copy_array_to_host_async(cdw::subview(output.view(), prefix, output_size), output_pitch,
        second.view(), cdw::make_full_array_region(second.view()), consumer);
    cdw::record_event(downloaded, consumer);
    require_current(before);
    cdw::synchronize_event(uploaded);
    std::fill(input.bytes().begin(), input.bytes().end(), std::byte{0}); // HtoA 完了後だけ再利用。
    cdw::synchronize_event(downloaded);
    require_equal(output.bytes(), expected_output, "Async transfer changed pixels/padding/guards");
    std::vector<std::byte> actual_first(initial_first.size());
    cdw::copy_array_to_host(actual_first, 9 * texel, first.view(), cdw::make_full_array_region(first.view()));
    require_equal(actual_first, expected_first, "Partial upload damaged pixels outside its rectangle");

    // 同じ array の非重複矩形間も確認。
    cdw::copy_array_to_array_async(first.view(), {6, 5, 2, 1}, first.view(), {2, 1, 2, 1}, consumer);
    cdw::synchronize_stream(consumer);
    std::memcpy(expected_first.data() + (5 * 9 + 6) * texel,
                expected_first.data() + (1 * 9 + 2) * texel, 2 * texel);
    cdw::copy_array_to_host(actual_first, 9 * texel, first.view(), cdw::make_full_array_region(first.view()));
    require_equal(actual_first, expected_first, "Same-array disjoint copy failed");
    require_current(before);
}

void test_linear_device(const CUcontext context, const CUarray_format format, const unsigned int channels)
{
    const auto before = cdw::get_current_context();
    const auto texel = cdw::texel_size_bytes(format, channels);
    const auto row = 7 * texel;
    const auto src_pitch = row + 17, dst_pitch = row + 29;
    const auto source_size = cdw::required_2d_bytes(row, 3, src_pitch);
    const auto destination_size = cdw::required_2d_bytes(row, 3, dst_pitch);
    // cuMemAllocPitch を使わず、先頭オフセットと奇数 pitch を含むケース。
    constexpr std::size_t src_prefix = 13, dst_prefix = 9, suffix = 19;
    std::vector<std::byte> input(src_prefix + source_size + suffix);
    fill_pattern(input);
    std::vector<std::byte> output(dst_prefix + destination_size + suffix, std::byte{0xB7});
    auto expected_output = output;
    cdw::Array2D image(context, {9, 7, format, channels});
    std::vector<std::byte> expected_image(9 * 7 * texel, std::byte{0x65});
    cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()), expected_image, 9 * texel);
    cdw::DeviceBuffer source(context, input.size()), destination(context, output.size());
    cdw::copy_host_to_device(source.view(), input);
    cdw::copy_host_to_device(destination.view(), output);
    cdw::Stream stream(context);
    FinishStream finish(stream);
    const cdw::ArrayRegion2D region{1, 2, 7, 3};
    cdw::copy_device_to_array_async(image.view(), region,
        cdw::subview(source.view(), src_prefix, source_size), src_pitch, stream);
    cdw::copy_array_to_device_async(cdw::subview(destination.view(), dst_prefix, destination_size),
        dst_pitch, image.view(), region, stream);
    cdw::synchronize_stream(stream);
    cdw::copy_device_to_host(output, destination.view());
    for(std::size_t y = 0; y < 3; ++y)
    {
        std::memcpy(expected_output.data() + dst_prefix + y * dst_pitch,
                    input.data() + src_prefix + y * src_pitch, row);
        std::memcpy(expected_image.data() + ((y + 2) * 9 + 1) * texel,
                    input.data() + src_prefix + y * src_pitch, row);
    }
    require_equal(output, expected_output, "Linear device <-> array transfer failed");
    std::vector<std::byte> actual_image(expected_image.size());
    cdw::copy_array_to_host(actual_image, 9 * texel, image.view(), cdw::make_full_array_region(image.view()));
    require_equal(actual_image, expected_image, "Device upload changed pixels outside region");
    require_current(before);
}

void test_invalid_transfers(const CUcontext context)
{
    cdw::Array2D image(context, {8, 5, CU_AD_FORMAT_UNSIGNED_INT8, 4});
    cdw::Array2D float_image(context, {8, 5, CU_AD_FORMAT_FLOAT, 1}); // 同じ texel byte size、異なる形式。
    std::vector<std::byte> host(8 * 5 * 4, std::byte{0});
    cdw::Stream stream(context);
    FinishStream finish(stream);
    cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()), host, 32);
    require_throws<std::out_of_range>([&]
    {
        cdw::copy_host_to_array(image.view(), {7, 0, 2, 1}, host, 8);
    }, "Out-of-bounds region accepted");
    require_throws<std::out_of_range>([&]
    {
        cdw::copy_host_to_array(image.view(), {1, 0, (std::numeric_limits<std::size_t>::max)(), 1}, host, 32);
    }, "Overflowing region accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()), host, 31);
    }, "Short source pitch accepted");
    require_throws<std::out_of_range>([&]
    {
        cdw::copy_array_to_host(std::span{host}.first(159), 32, image.view(), cdw::make_full_array_region(image.view()));
    }, "One-byte-short destination accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_array_async(image.view(), {0, 0, 1, 1}, {context, nullptr, 4}, 4, stream);
    }, "Null pinned host memory accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_device_to_array_async(image.view(), {0, 0, 1, 1}, {context, 0, 4}, 4, stream);
    }, "Null device memory accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_array_to_array_async(float_image.view(), {0, 0, 1, 1}, image.view(), {0, 0, 1, 1}, stream);
    }, "Different formats accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_array_to_array_async(image.view(), {0, 0, 2, 1}, image.view(), {0, 0, 1, 1}, stream);
    }, "Different rectangle sizes accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_array_to_array_async(image.view(), {1, 1, 4, 3}, image.view(), {0, 0, 4, 3}, stream);
    }, "Overlapping rectangles accepted");
    cdw::copy_array_to_array_async(image.view(), {0, 0, 4, 3}, image.view(), {0, 0, 4, 3}, stream);
    const cdw::ArrayRegion2D empty{8, 5, 0, 0};
    cdw::copy_host_to_array(image.view(), empty, {}, 0);
    cdw::copy_array_to_host({}, 0, image.view(), empty);
    cdw::copy_host_to_array_async(image.view(), empty, {}, 0, stream);
    cdw::copy_array_to_host_async({}, 0, image.view(), empty, stream);
    cdw::copy_device_to_array_async(image.view(), empty, {}, 0, stream);
    cdw::copy_array_to_device_async({}, 0, image.view(), empty, stream);
    cdw::copy_array_to_array_async(image.view(), empty, image.view(), empty, stream);
    cdw::Stream closed(context);
    closed.close();
    require_throws<std::logic_error>([&]
    {
        cdw::copy_host_to_array_async(image.view(), empty, {}, 0, closed);
    }, "Closed stream accepted for empty transfer");
    cdw::Array2D invalid;
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_array(invalid.view(), {}, {}, 0);
    }, "Invalid array accepted for empty transfer");
    std::vector<std::byte> output(host.size());
    cdw::copy_array_to_host(output, 32, image.view(), cdw::make_full_array_region(image.view()));
    require_equal(output, host, "Rejected/empty/self copies modified image");
}

void test_external_context(const CUdevice device_id, const CUcontext context)
{
    const auto before = cdw::get_current_context();
    ExternalContext other(device_id);
    {
        cdw::ScopedCurrentContext current(other.handle());
        test_format_roundtrips(context, {3, 2, CU_AD_FORMAT_FLOAT, 2});
        test_ownership(context);
        test_linear_device(context, CU_AD_FORMAT_UNSIGNED_INT8, 1);
        require_current(other.handle());
    }
    require_current(before);
    cdw::Array2D primary_array(context, {2, 2});
    cdw::Array2D other_array(other.handle(), {2, 2});
    cdw::PinnedHostBuffer other_host(other.handle(), 4);
    cdw::DeviceBuffer other_device(other.handle(), 4);
    cdw::Stream stream(context);
    FinishStream finish(stream);
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_array_to_array_async(primary_array.view(), {0, 0, 2, 2},
            other_array.view(), {0, 0, 2, 2}, stream);
    }, "Cross-context array copy accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_host_to_array_async(primary_array.view(), {0, 0, 2, 2}, other_host.view(), 2, stream);
    }, "Cross-context pinned memory accepted");
    require_throws<std::invalid_argument>([&]
    {
        cdw::copy_array_to_device_async(other_device.view(), 2, primary_array.view(), {0, 0, 2, 2}, stream);
    }, "Cross-context device memory accepted");
    // 外部所有の CUarray も、元の所有者を維持したまま view として渡せる。
    const cdw::Array2DView borrowed{context, primary_array.handle(), primary_array.descriptor()};
    const std::array<std::byte, 4> input{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    std::array<std::byte, 4> output{};
    cdw::copy_host_to_array(borrowed, {0, 0, 2, 2}, input, 2);
    cdw::copy_array_to_host(output, 2, borrowed, {0, 0, 2, 2});
    require_equal(output, input, "Borrowed array copy failed");
    require_current(before);
}

} // namespace

int main()
{
    static_assert(!std::is_copy_constructible_v<cdw::Array2D>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::Array2D>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::Array2D>);
    static_assert(std::is_nothrow_destructible_v<cdw::Array2D>);
    static_assert(std::is_convertible_v<cdw::Array2DView, cdw::ConstArray2DView>);
    static_assert(!std::is_convertible_v<cdw::ConstArray2DView, cdw::Array2DView>);
    static_assert(std::is_same_v<decltype(std::declval<const cdw::Array2D&>().view()), cdw::ConstArray2DView>);
    try
    {
        test_metadata();
        int count = 0;
        try { cdw::initialize_driver(); count = cdw::get_device_count(); }
        catch(const cdw::CudaError& error)
        {
            if(error.result() != CUDA_ERROR_NO_DEVICE) { throw; }
            std::cout << "SKIP: no CUDA device.\n";
            return 77;
        }
        if(count == 0) { std::cout << "SKIP: no visible CUDA device.\n"; return 77; }
        const auto device = cdw::get_device(0);
        std::cout << "2D array tests on: " << cdw::get_device_info(device).name << '\n';
        const auto before = cdw::get_current_context();
        {
            cdw::RetainPrimaryContext primary(device);
            const auto context = primary.handle();
            test_ownership(context);
            for(const auto format : {CU_AD_FORMAT_UNSIGNED_INT8, CU_AD_FORMAT_FLOAT})
            {
                for(const unsigned int channels : {1u, 2u, 4u})
                {
                    for(const auto& extent : {std::pair<std::size_t, std::size_t>{1, 1}, {7, 3}, {1, 5}, {9, 1}})
                    {
                        test_format_roundtrips(context, {extent.first, extent.second, format, channels});
                    }
                    test_linear_device(context, format, channels);
                }
            }
            test_partial_async(context);
            test_invalid_transfers(context);
            test_external_context(device, context);
            require_current(before);
        }
        require_current(before);
        std::cout << "All 2D array tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
