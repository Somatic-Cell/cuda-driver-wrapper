#include <cuda_driver_wrapper/texture_object.hpp>
#include <cuda_driver_wrapper/array_2d.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/device.hpp>
#include <cuda_driver_wrapper/device_buffer.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/event.hpp>
#include <cuda_driver_wrapper/kernel.hpp>
#include <cuda_driver_wrapper/module.hpp>
#include <cuda_driver_wrapper/pinned_host_buffer.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
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

template<class Function>
void require_invalid_argument(Function&& function)
{
    try { std::forward<Function>(function)(); }
    catch(const std::invalid_argument&) { return; }
    throw std::runtime_error("An unsupported texture configuration was accepted");
}

void require_current(const CUcontext expected)
{
    require(cdw::get_current_context() == expected, "Current context was not restored");
}

void require_near(const float actual, const float expected, const std::string_view label)
{
    // 座標は補間重みが二進で正確に表せる位置を選ぶ。
    // 正規化整数 -> float の丸め等は許すが、大きな補間誤差を許す値にはしない。
    const float tolerance = 5.0e-5f * std::max(1.0f, std::abs(expected));
    if(!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
    {
        throw std::runtime_error(std::string(label) + ": expected " +
            std::to_string(expected) + ", actual " + std::to_string(actual));
    }
}

// テスト専用。GPU が使用する所有者より後に宣言し、失敗時も先に同期する。
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
            require(popped == context_, "Wrong external context popped");
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

// 実際に作成された資源の結合と主要な設定を、生の Driver API でも照合する。
void verify_descriptor(
    const cdw::TextureObject& texture, const cdw::ConstArray2DView image,
    const cdw::Texture2DDescriptor& settings
)
{
    const CUcontext before = cdw::get_current_context();
    {
        cdw::ScopedCurrentContext current(texture.context());
        CUDA_RESOURCE_DESC resource{};
        CUDA_TEXTURE_DESC sampling{};
        cdw::check_cuda(cuTexObjectGetResourceDesc(&resource, texture.handle()), "cuTexObjectGetResourceDesc (test)");
        cdw::check_cuda(cuTexObjectGetTextureDesc(&sampling, texture.handle()), "cuTexObjectGetTextureDesc (test)");
        require(resource.resType == CU_RESOURCE_TYPE_ARRAY && resource.res.array.hArray == image.array,
            "Texture must refer to the original array, not a copy");
        require(sampling.addressMode[0] == settings.address_mode_u &&
                sampling.addressMode[1] == settings.address_mode_v &&
                sampling.filterMode == settings.filter_mode, "Texture sampling settings differ");
        // FLOAT に対する READ_AS_INTEGER は読み出しの意味を変えないので比較しない。
        const unsigned int mask = CU_TRSF_NORMALIZED_COORDINATES |
            (image.descriptor.format == CU_AD_FORMAT_UNSIGNED_INT8 ? CU_TRSF_READ_AS_INTEGER : 0u);
        const unsigned int expected =
            (settings.normalized_coordinates ? CU_TRSF_NORMALIZED_COORDINATES : 0u) |
            (settings.read_as_integer ? CU_TRSF_READ_AS_INTEGER : 0u);
        require((sampling.flags & mask) == expected, "Texture coordinate/read flags differ");
        require((sampling.flags & CU_TRSF_SRGB) == 0, "sRGB conversion must not be enabled implicitly");
    }
    require_current(before);
}

// M0.6 のカーネル起動、M0.5 のバッファをそのまま使う。
// このテスト補助関数は結果比較のために同期する。ライブラリの暗黙同期ではない。
template<class Output>
std::vector<Output> sample(
    const cdw::Module& module, const cdw::TextureObject& texture, const cdw::Stream& stream,
    const std::vector<float>& coordinates, const unsigned int channels
)
{
    static_assert(std::is_same_v<Output, float> || std::is_same_v<Output, unsigned int>);
    require(texture.is_valid() && texture.context() == stream.context(), "Invalid texture/stream in test");
    require(!coordinates.empty() && coordinates.size() % 2 == 0, "Expected xy coordinate pairs");
    require(coordinates.size() / 2 <= (std::numeric_limits<unsigned int>::max)(), "Too many test samples");
    const auto before = cdw::get_current_context();
    unsigned int count = static_cast<unsigned int>(coordinates.size() / 2);
    const auto coordinate_bytes = std::as_bytes(std::span{coordinates});
    std::vector<Output> output(cdw::checked_byte_size(count, channels));
    cdw::DeviceBuffer device_coordinates(texture.context(), coordinate_bytes.size_bytes());
    cdw::DeviceBuffer device_output(texture.context(), cdw::checked_byte_size(output.size(), sizeof(Output)));
    FinishStream finish(stream);
    cdw::copy_host_to_device(device_coordinates.view(), coordinate_bytes);
    const std::string name = std::string(std::is_same_v<Output, float> ? "cdw_sample_float_" : "cdw_sample_u8_")
                           + std::to_string(channels);
    const auto kernel = cdw::get_kernel(module, name);
    CUtexObject handle = texture.handle();
    CUdeviceptr input_address = device_coordinates.device_ptr();
    CUdeviceptr output_address = device_output.device_ptr();
    void* arguments[]{&handle, &input_address, &output_address, &count};
    cdw::LaunchConfig launch;
    launch.block_x = 64;
    launch.grid_x = count / 64 + (count % 64 != 0 ? 1u : 0u);
    cdw::launch_kernel(kernel, launch, stream, arguments);
    cdw::synchronize_stream(stream);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), device_output.view());
    require_current(before);
    return output;
}

// CPU 参照解。添字に境界規則を適用する。-1 はゼロ色 border。
int address_index(const int index, const int size, const CUaddress_mode mode)
{
    if(mode == CU_TR_ADDRESS_MODE_CLAMP) { return std::clamp(index, 0, size - 1); }
    if(mode == CU_TR_ADDRESS_MODE_BORDER) { return index < 0 || index >= size ? -1 : index; }
    if(mode == CU_TR_ADDRESS_MODE_WRAP) { return ((index % size) + size) % size; }
    const int period = 2 * size;
    const int folded = ((index % period) + period) % period;
    return folded < size ? folded : period - 1 - folded;
}

float reference_sample(
    const std::vector<float>& values, const int width, const int height, const unsigned int channels,
    const unsigned int channel, const float x, const float y, const cdw::Texture2DDescriptor& settings
)
{
    const auto texel = [&](const int ix, const int iy)
    {
        const int sx = address_index(ix, width, settings.address_mode_u);
        const int sy = address_index(iy, height, settings.address_mode_v);
        if(sx < 0 || sy < 0) { return 0.0f; }
        return values[(static_cast<std::size_t>(sy) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(sx)) * channels + channel];
    };
    if(settings.filter_mode == CU_TR_FILTER_MODE_POINT)
    {
        return texel(static_cast<int>(std::floor(x)), static_cast<int>(std::floor(y)));
    }
    const float fx = x - 0.5f, fy = y - 0.5f;
    const int ix = static_cast<int>(std::floor(fx)), iy = static_cast<int>(std::floor(fy));
    const float a = fx - static_cast<float>(ix), b = fy - static_cast<float>(iy);
    return (1 - b) * ((1 - a) * texel(ix, iy) + a * texel(ix + 1, iy))
         + b * ((1 - a) * texel(ix, iy + 1) + a * texel(ix + 1, iy + 1));
}

void test_sampling(const CUcontext context, const cdw::Module& module)
{
    constexpr int width = 4, height = 4;
    // 中心、内部、境界の補間、境界外。/4 しても正確に表現できる座標。
    const std::vector<float> texel_coordinates{
        0.5f,0.5f, 3.5f,2.5f, 1.0f,1.0f, 3.75f,1.25f, 0.0f,0.0f, 4.0f,4.0f,
        -0.5f,1.5f, 4.5f,1.5f, 1.5f,-0.5f, 1.5f,4.5f, -1.5f,-1.5f, 5.5f,5.5f
    };
    cdw::Stream stream(context);
    for(const auto format : {CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_UNSIGNED_INT8})
    for(const unsigned int channels : {1u, 2u, 4u})
    {
        cdw::Array2D image(context, {width, height, format, channels});
        std::vector<float> base_values(16 * channels);
        std::vector<unsigned char> integer_values(base_values.size());
        for(std::size_t i = 0; i < base_values.size(); ++i)
        {
            integer_values[i] = static_cast<unsigned char>((i * 37) % 256);
            base_values[i] = format == CU_AD_FORMAT_FLOAT
                ? static_cast<float>(i) - 5.0f : static_cast<float>(integer_values[i]);
        }
        if(format == CU_AD_FORMAT_UNSIGNED_INT8)
        {
            integer_values.back() = 255;
            base_values.back() = 255;
            cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()),
                std::as_bytes(std::span{integer_values}), cdw::row_size_bytes(image.descriptor()));
        }
        else
        {
            cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()),
                std::as_bytes(std::span{base_values}), cdw::row_size_bytes(image.descriptor()));
        }
        for(const bool normalized : {false, true})
        for(const auto address : {CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_WRAP,
                                  CU_TR_ADDRESS_MODE_MIRROR, CU_TR_ADDRESS_MODE_BORDER})
        {
            if(!normalized && address != CU_TR_ADDRESS_MODE_CLAMP) { continue; }
            for(const bool integer_read : {false, true})
            {
                if(integer_read && format != CU_AD_FORMAT_UNSIGNED_INT8) { continue; }
                for(const auto filter : {CU_TR_FILTER_MODE_POINT, CU_TR_FILTER_MODE_LINEAR})
                {
                    if(integer_read && filter == CU_TR_FILTER_MODE_LINEAR) { continue; }
                    cdw::Texture2DDescriptor settings;
                    settings.normalized_coordinates = normalized;
                    settings.address_mode_u = address;
                    settings.address_mode_v = address;
                    settings.filter_mode = filter;
                    settings.read_as_integer = integer_read;
                    // const owner から読み取り用 view を取得する経路も使う。
                    const cdw::Array2D& constant_image = image;
                    cdw::TextureObject texture(constant_image.view(), settings);
                    verify_descriptor(texture, image.view(), settings);
                    std::vector<float> coordinates = texel_coordinates;
                    if(normalized) { for(float& value : coordinates) { value /= 4.0f; } }
                    std::vector<float> reference_values = base_values;
                    if(format == CU_AD_FORMAT_UNSIGNED_INT8 && !integer_read)
                    {
                        for(float& value : reference_values) { value /= 255.0f; }
                    }
                    std::vector<float> result;
                    if(integer_read)
                    {
                        const auto raw = sample<unsigned int>(module, texture, stream, coordinates, channels);
                        result.assign(raw.begin(), raw.end());
                    }
                    else { result = sample<float>(module, texture, stream, coordinates, channels); }
                    for(std::size_t i = 0; i < coordinates.size() / 2; ++i)
                    for(unsigned int c = 0; c < channels; ++c)
                    {
                        const float expected = reference_sample(reference_values, width, height, channels, c,
                            texel_coordinates[2 * i], texel_coordinates[2 * i + 1], settings);
                        const std::string label = "format=" + std::to_string(static_cast<int>(format)) +
                            " channels=" + std::to_string(channels) + " mode=" +
                            std::to_string(static_cast<int>(address)) + " filter=" +
                            std::to_string(static_cast<int>(filter)) + " sample=" + std::to_string(i);
                        if(integer_read) { require(result[i * channels + c] == expected, label); }
                        else { require_near(result[i * channels + c], expected, label); }
                    }
                }
            }
        }
    }
}

void test_invalid_configurations(const CUcontext context)
{
    cdw::Array2D image(context, {4, 4, CU_AD_FORMAT_UNSIGNED_INT8, 4});
    cdw::Array2D float_image(context, {4, 4, CU_AD_FORMAT_FLOAT, 1});
    require_invalid_argument([] { cdw::TextureObject bad(cdw::ConstArray2DView{}); });
    auto invalid = static_cast<cdw::ConstArray2DView>(image.view());
    invalid.context = nullptr;
    require_invalid_argument([&] { cdw::TextureObject bad(invalid); });
    invalid = image.view(); invalid.descriptor.width = 0;
    require_invalid_argument([&] { cdw::TextureObject bad(invalid); });
    invalid = image.view(); invalid.descriptor.num_channels = 3;
    require_invalid_argument([&] { cdw::TextureObject bad(invalid); });

    cdw::Texture2DDescriptor settings;
    settings.read_as_integer = true;
    require_invalid_argument([&] { cdw::TextureObject bad(float_image.view(), settings); });
    settings.filter_mode = CU_TR_FILTER_MODE_LINEAR;
    require_invalid_argument([&] { cdw::TextureObject bad(image.view(), settings); });
    for(const auto mode : {CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_MIRROR, CU_TR_ADDRESS_MODE_BORDER})
    {
        settings = {};
        settings.normalized_coordinates = false;
        settings.address_mode_u = mode;
        require_invalid_argument([&] { cdw::TextureObject bad(image.view(), settings); });
        settings.address_mode_u = CU_TR_ADDRESS_MODE_CLAMP;
        settings.address_mode_v = mode;
        require_invalid_argument([&] { cdw::TextureObject bad(image.view(), settings); });
    }
}

void test_ownership(const CUcontext context, const cdw::Module& module)
{
    const CUcontext before = cdw::get_current_context();
    cdw::TextureObject empty;
    require(!empty.is_valid() && empty.context() == nullptr && empty.handle() == 0, "Default texture is not empty");
    empty.close(); empty.close();
    cdw::Array2D image(context, {2, 2, CU_AD_FORMAT_FLOAT, 1});
    const std::array<float, 4> values{1, 3, 5, 7};
    cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()),
        std::as_bytes(std::span{values}), 2 * sizeof(float));
    cdw::Stream stream(context);
    {
        cdw::TextureObject original(image.view());
        const auto handle = original.handle();
        cdw::TextureObject moved(std::move(original));
        require(!original.is_valid() && original.context() == nullptr, "Move did not empty source");
        require(moved.handle() == handle, "Move recreated the texture");
        original.close();
        cdw::TextureObject target(image.view());
        target = std::move(moved);
        require(!moved.is_valid() && target.handle() == handle, "Move assignment failed");
        auto* alias = &target;
        target = std::move(*alias);
        verify_descriptor(target, image.view(), {});
        const auto result = sample<float>(module, target, stream, {0.25f, 0.25f, 0.75f, 0.75f}, 1);
        require_near(result[0], 1, "Sampling after move");
        require_near(result[1], 7, "Sampling after move");
        target.close(); target.close();
        require(!target.is_valid() && target.context() == nullptr, "Close did not clear texture");
    }
    { cdw::TextureObject temporary(image.view()); }
    std::array<float, 4> output{};
    cdw::copy_array_to_host(std::as_writable_bytes(std::span{output}), 2 * sizeof(float),
        image.view(), cdw::make_full_array_region(image.view()));
    require(values == output, "Texture destruction must not destroy or change the borrowed array");
    require_current(before);
}

void test_handle_table(const CUcontext context, const cdw::Module& module)
{
    cdw::Array2D image(context, {4, 4, CU_AD_FORMAT_FLOAT, 1});
    cdw::Array2D other(context, {4, 4, CU_AD_FORMAT_FLOAT, 1});
    std::array<float, 16> a{}, b{};
    for(std::size_t i = 0; i < a.size(); ++i) { a[i] = static_cast<float>(i); b[i] = a[i] + 100; }
    cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()), std::as_bytes(std::span{a}), 16);
    cdw::copy_host_to_array(other.view(), cdw::make_full_array_region(other.view()), std::as_bytes(std::span{b}), 16);
    cdw::Texture2DDescriptor settings;
    settings.normalized_coordinates = false;
    cdw::TextureObject point(image.view(), settings);
    cdw::TextureObject other_point(other.view(), settings);
    settings.filter_mode = CU_TR_FILTER_MODE_LINEAR;
    cdw::TextureObject linear(image.view(), settings);
    const std::array<CUtexObject, 3> handles{point.handle(), linear.handle(), other_point.handle()};
    const std::array<unsigned int, 6> indices{0, 1, 2, 1, 0, 2};
    std::array<float, 12> coordinates{};
    coordinates.fill(1); // 非正規化の (1,1)。point は 5、linear は 2.5。
    std::array<float, 6> output{};
    cdw::DeviceBuffer h(context, sizeof(handles)), ids(context, sizeof(indices));
    cdw::DeviceBuffer xy(context, sizeof(coordinates)), out(context, sizeof(output));
    cdw::Stream stream(context);
    FinishStream finish(stream);
    cdw::copy_host_to_device(h.view(), std::as_bytes(std::span{handles}));
    cdw::copy_host_to_device(ids.view(), std::as_bytes(std::span{indices}));
    cdw::copy_host_to_device(xy.view(), std::as_bytes(std::span{coordinates}));
    CUdeviceptr hp = h.device_ptr(), ip = ids.device_ptr(), xp = xy.device_ptr(), op = out.device_ptr();
    unsigned int count = static_cast<unsigned int>(indices.size());
    void* args[]{&hp, &ip, &xp, &op, &count};
    cdw::LaunchConfig launch;
    launch.block_x = 32;
    cdw::launch_kernel(cdw::get_kernel(module, "cdw_sample_texture_table"), launch, stream, args);
    cdw::synchronize_stream(stream);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{output}), out.view());
    const std::array<float, 3> expected{5, 2.5f, 105};
    for(std::size_t i = 0; i < output.size(); ++i) { require_near(output[i], expected[indices[i]], "Handle table"); }
}


void test_dimensions_and_axis_modes(const CUcontext context, const cdw::Module& module)
{
    cdw::Stream stream(context);
    cdw::Array2D image(context, {4, 2, CU_AD_FORMAT_FLOAT, 1});
    const std::array<float, 8> pixels{0, 1, 2, 3, 4, 5, 6, 7};
    cdw::copy_host_to_array(image.view(), cdw::make_full_array_region(image.view()),
        std::as_bytes(std::span{pixels}), 4 * sizeof(float));
    cdw::Texture2DDescriptor settings;
    settings.address_mode_u = CU_TR_ADDRESS_MODE_WRAP;
    settings.address_mode_v = CU_TR_ADDRESS_MODE_CLAMP;
    cdw::TextureObject texture(image.view(), settings);
    verify_descriptor(texture, image.view(), settings);
    const auto values = sample<float>(module, texture, stream,
        {-0.125f, 1.25f, 1.125f, -0.25f, 0.375f, 0.75f}, 1);
    require_near(values[0], 7, "u=wrap, v=clamp");
    require_near(values[1], 0, "u=wrap, v=clamp");
    require_near(values[2], 5, "Non-square normalized coordinates");

    cdw::Array2D single(context, {1, 1, CU_AD_FORMAT_FLOAT, 1});
    const std::array<float, 1> one{-3.5f};
    cdw::copy_host_to_array(single.view(), cdw::make_full_array_region(single.view()),
        std::as_bytes(std::span{one}), sizeof(float));
    settings = {};
    settings.filter_mode = CU_TR_FILTER_MODE_LINEAR;
    cdw::TextureObject single_texture(single.view(), settings);
    for(const float value : sample<float>(module, single_texture, stream,
        {0.5f, 0.5f, -1.0f, 2.0f, 1.0f, 1.0f}, 1))
    {
        require_near(value, -3.5f, "1x1 linear clamp");
    }
}

void test_async_update(const CUcontext context, const cdw::Module& module)
{
    cdw::Array2D image(context, {4, 4, CU_AD_FORMAT_FLOAT, 1});
    cdw::PinnedHostBuffer input(context, 16 * sizeof(float));
    cdw::TextureObject texture(image.view()); // 画素転送は構築の責務ではない。
    cdw::Stream upload(context), render(context);
    cdw::Event ready(context);
    FinishStream finish_upload(upload), finish_render(render);
    const auto original_handle = texture.handle();
    for(const float value : {2.0f, 9.0f})
    {
        std::array<float, 16> values{};
        values.fill(value);
        std::memcpy(input.bytes().data(), values.data(), sizeof(values));
        cdw::copy_host_to_array_async(image.view(), cdw::make_full_array_region(image.view()),
            input.view(), 4 * sizeof(float), upload);
        cdw::record_event(ready, upload);
        cdw::stream_wait_event(render, ready);
        const auto result = sample<float>(module, texture, render, {0.125f, 0.125f, 0.875f, 0.875f}, 1);
        for(const float actual : result) { require_near(actual, value, "Updated array"); }
        require(texture.handle() == original_handle, "Pixel update should not require rebuilding a texture");
        // sample() 内で render の完了を確認済み。次の書き換えとアップロードはその後。
    }
}

} // namespace

int main(int argc, char** argv)
{
    static_assert(!std::is_copy_constructible_v<cdw::TextureObject>);
    static_assert(!std::is_copy_assignable_v<cdw::TextureObject>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::TextureObject>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::TextureObject>);
    static_assert(std::is_nothrow_destructible_v<cdw::TextureObject>);
    static_assert(sizeof(float) == 4 && sizeof(unsigned int) == 4 && sizeof(CUtexObject) == 8);
    try
    {
        require(argc == 2, "Expected the texture test fatbin path as the only argument");
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
        std::cout << "Texture tests on: " << cdw::get_device_info(device).name << '\n';
        const CUcontext before = cdw::get_current_context();
        {
            cdw::RetainPrimaryContext primary(device);
            const CUcontext context = primary.handle();
            cdw::Module module(context, std::filesystem::path(argv[1]));
            test_invalid_configurations(context);
            test_ownership(context, module);
            test_sampling(context, module);
            test_handle_table(context, module);
            test_dimensions_and_axis_modes(context, module);
            test_async_update(context, module);
            {
                ExternalContext other(device);
                cdw::ScopedCurrentContext current(other.handle());
                test_ownership(context, module);
                require_current(other.handle());
                current.restore();
            }
            require_current(before);
        }
        require_current(before);
        std::cout << "All texture object tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
