#include <cuda_driver_wrapper/mipmapped_array_2d.hpp>
#include <cuda_driver_wrapper/texture_object.hpp>
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

void require(bool condition, std::string_view message)
{
    if(!condition) { throw std::runtime_error(std::string(message)); }
}

template<class Exception, class Function>
void require_throws(Function&& function)
{
    try { std::forward<Function>(function)(); }
    catch(const Exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

void require_current(CUcontext expected)
{
    require(cdw::get_current_context() == expected, "Current context was not restored");
}

void require_near(float actual, float expected, const std::string& label)
{
    // 座標とLODには二進で正確な値を使う。UINT8->float変換等の丸めは許容。
    if(!std::isfinite(actual) || std::abs(actual - expected) > 0.0002f)
    {
        throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected));
    }
}

// 必ず全使用資源より後、非同期投入より前に構築する。
class FinishStream final
{
public:
    explicit FinishStream(const cdw::Stream& stream) noexcept : stream_(stream) {}
    ~FinishStream() noexcept
    {
        try { cdw::synchronize_stream(stream_); }
        catch(...) { std::terminate(); }
    }
    FinishStream(const FinishStream&) = delete;
    FinishStream& operator=(const FinishStream&) = delete;
private:
    const cdw::Stream& stream_;
};

class ExternalContext final
{
public:
    explicit ExternalContext(CUdevice device)
    {
#if CUDA_VERSION >= 13000
        CUctxCreateParams parameters{};
        cdw::check_cuda(cuCtxCreate(&context_, &parameters, 0, device), "cuCtxCreate (mipmap test)");
#else
        cdw::check_cuda(cuCtxCreate(&context_, 0, device), "cuCtxCreate (mipmap test)");
#endif
        try
        {
            CUcontext popped = nullptr;
            cdw::check_cuda(cuCtxPopCurrent(&popped), "cuCtxPopCurrent (mipmap test setup)");
            require(popped == context_, "Unexpected context after cuCtxCreate");
        }
        catch(...)
        {
            static_cast<void>(cdw::report_cuda_cleanup_result(cuCtxDestroy(context_), "cuCtxDestroy (rollback)"));
            throw;
        }
    }
    ~ExternalContext() noexcept
    {
        static_cast<void>(cdw::report_cuda_cleanup_result(cuCtxDestroy(context_), "cuCtxDestroy (mipmap test)"));
    }
    ExternalContext(const ExternalContext&) = delete;
    ExternalContext& operator=(const ExternalContext&) = delete;
    CUcontext handle() const noexcept { return context_; }
private:
    CUcontext context_ = nullptr;
};

void test_metadata()
{
    require(cdw::full_mip_level_count(1, 1) == 1, "1x1 level count");
    require(cdw::full_mip_level_count(8, 2) == 4, "8x2 level count");
    require(cdw::full_mip_level_count(7, 5) == 3, "Odd extent level count");
    require(cdw::full_mip_level_count(1, 9) == 4, "1x9 level count");
    require(cdw::full_mip_level_count((std::numeric_limits<std::size_t>::max)(), 1) ==
            static_cast<unsigned int>(std::numeric_limits<std::size_t>::digits), "Large extent level count");
    require_throws<std::invalid_argument>([] { static_cast<void>(cdw::full_mip_level_count(0, 1)); });
    require_throws<std::invalid_argument>([] { static_cast<void>(cdw::full_mip_level_count(1, 0)); });
    cdw::MipmappedArray2D empty;
    empty.close(); empty.close();
    require(!empty.is_valid() && empty.level_count() == 0, "Default state must be empty");
    require_throws<std::invalid_argument>([&] { static_cast<void>(cdw::get_mip_level(empty.view(), 0)); });
    require_throws<std::invalid_argument>([] { cdw::MipmappedArray2D bad(nullptr, {1, 1}, 1); });
    require_throws<std::invalid_argument>([]
    {
        cdw::TextureObject bad(cdw::ConstMipmappedArray2DView{}, cdw::MipmappedTexture2DDescriptor{});
    });
}

void test_storage(CUcontext context)
{
    const auto before = cdw::get_current_context();
    for(const auto format : {CU_AD_FORMAT_UNSIGNED_INT8, CU_AD_FORMAT_FLOAT})
    for(unsigned int channels : {1u, 2u, 4u})
    for(const auto shape : {std::array<std::size_t, 2>{7, 5}, {1, 9}, {9, 1}, {1, 1}})
    {
        const auto levels = cdw::full_mip_level_count(shape[0], shape[1]);
        cdw::MipmappedArray2D image(context, {shape[0], shape[1], format, channels}, levels);
        std::size_t w = shape[0], h = shape[1];
        for(unsigned int level = 0; level < levels; ++level)
        {
            const auto target = cdw::get_mip_level(image.view(), level);
            const auto source = cdw::get_mip_level(std::as_const(image).view(), level);
            require(source.array == target.array, "Const level view changed the handle");
            require(target.descriptor.width == w && target.descriptor.height == h &&
                    target.descriptor.format == format && target.descriptor.num_channels == channels,
                    "Driver level dimensions/format differ from expected values");
            const auto row = cdw::row_size_bytes(target.descriptor);
            // FLOATであってもここはバイト保存の検証。サンプリングは後の既知のfloat値で行う。
            const auto pitch = row + 8;
            const auto bytes = cdw::required_2d_bytes(row, h, pitch);
            std::vector<std::byte> input(bytes, std::byte{0xCD}), output(bytes, std::byte{0xCD});
            for(std::size_t y = 0; y < h; ++y)
            for(std::size_t x = 0; x < row; ++x)
            {
                input[y * pitch + x] = static_cast<std::byte>((level * 31 + y * 7 + x) % 256);
            }
            cdw::copy_host_to_array(target, cdw::make_full_array_region(target), input, pitch);
            cdw::copy_array_to_host(output, pitch, source, cdw::make_full_array_region(source));
            require(input == output, "Mip level roundtrip changed data/padding");
            require_current(before);
            w = (std::max)(std::size_t{1}, w / 2);
            h = (std::max)(std::size_t{1}, h / 2);
        }
        require_throws<std::out_of_range>([&] { static_cast<void>(cdw::get_mip_level(image.view(), levels)); });
    }
    // 完全なchainより短い確保も有効。要求した個数を保持し、黙ってclampしない。
    cdw::MipmappedArray2D partial(context, {8, 4, CU_AD_FORMAT_FLOAT, 1}, 2);
    require(partial.level_count() == 2, "Partial chain count");
    require_throws<std::out_of_range>([&] { static_cast<void>(cdw::get_mip_level(partial.view(), 2)); });
    require_throws<std::invalid_argument>([&] { cdw::MipmappedArray2D bad(context, {8, 4}, 0); });
    require_throws<std::invalid_argument>([&] { cdw::MipmappedArray2D bad(context, {8, 4}, 5); });
    require_throws<std::invalid_argument>([&] { cdw::MipmappedArray2D bad(context, {0, 4}, 1); });
    require_throws<std::invalid_argument>([&] { cdw::MipmappedArray2D bad(context, {4, 0}, 1); });
    require_throws<std::invalid_argument>([&] { cdw::MipmappedArray2D bad(context, {4, 4, CU_AD_FORMAT_FLOAT, 3}, 1); });
    require_current(before);
}

void test_ownership(CUcontext context)
{
    const auto before = cdw::get_current_context();
    cdw::MipmappedArray2D original(context, {4, 2, CU_AD_FORMAT_FLOAT, 1}, 3);
    const auto borrowed = original.view();
    const auto level = cdw::get_mip_level(borrowed, 2);
    const std::array<float, 1> value{17.0f};
    cdw::copy_host_to_array(level, {0, 0, 1, 1}, std::as_bytes(std::span{value}), sizeof(float));
    cdw::MipmappedArray2D moved(std::move(original));
    require(!original.is_valid() && original.level_count() == 0 && original.context() == nullptr,
            "Move did not clear source");
    cdw::MipmappedArray2D destination(context, {2, 2}, 1);
    destination = std::move(moved);
    require(!moved.is_valid() && destination.handle() == borrowed.array, "Move assignment changed resource");
    auto* alias = &destination;
    destination = std::move(*alias);
    std::array<float, 1> output{};
    cdw::copy_array_to_host(std::as_writable_bytes(std::span{output}), sizeof(float), level, {0, 0, 1, 1});
    require(output == value, "Move invalidated an existing level view");
    {
        cdw::TextureObject first(destination.view(), cdw::MipmappedTexture2DDescriptor{});
        const CUtexObject handle = first.handle();
        cdw::TextureObject second(std::move(first));
        require(!first.is_valid() && second.handle() == handle, "Texture move changed handle");
        cdw::TextureObject third(destination.view(), cdw::MipmappedTexture2DDescriptor{});
        third = std::move(second);
        require(!second.is_valid(), "Texture move assignment did not clear source");
        third.close(); third.close();
    }
    // texture の破棄は親のmipmapped arrayを破棄しない。
    require(cdw::get_mip_level(destination.view(), 2).array == level.array, "Texture destroyed parent array");
    destination.close(); destination.close();
    require(!destination.is_valid() && destination.level_count() == 0, "Close did not clear state");
    require_current(before);
}

// 各レベルはテスト側で独立に作る入力画像。level 0 を縮小する実装ではない。
void upload_constants(cdw::MipmappedArray2D& image)
{
    const auto base = image.base_descriptor();
    for(unsigned int level = 0; level < image.level_count(); ++level)
    {
        const auto view = cdw::get_mip_level(image.view(), level);
        const auto count = view.descriptor.width * view.descriptor.height * base.num_channels;
        if(base.format == CU_AD_FORMAT_FLOAT)
        {
            std::vector<float> pixels(count);
            for(std::size_t i = 0; i < count; ++i)
            { pixels[i] = 0.5f * static_cast<float>(level) + 0.125f * static_cast<float>(i % base.num_channels); }
            cdw::copy_host_to_array(view, cdw::make_full_array_region(view),
                                   std::as_bytes(std::span{pixels}), cdw::row_size_bytes(view.descriptor));
        }
        else
        {
            std::vector<unsigned char> pixels(count);
            for(std::size_t i = 0; i < count; ++i)
            { pixels[i] = static_cast<unsigned char>(level * 64u + (i % base.num_channels) * 7u); }
            cdw::copy_host_to_array(view, cdw::make_full_array_region(view),
                                   std::as_bytes(std::span{pixels}), cdw::row_size_bytes(view.descriptor));
        }
    }
}

std::vector<float> sample(
    const cdw::Module& module, const cdw::TextureObject& texture, const cdw::Stream& stream,
    const std::string& kernel_name, const std::vector<float>& queries
)
{
    require(!queries.empty() && queries.size() % 3 == 0, "Invalid query fixture");
    const auto count = static_cast<unsigned int>(queries.size() / 3);
    cdw::DeviceBuffer device_queries(texture.context(), queries.size() * sizeof(float));
    cdw::DeviceBuffer device_result(texture.context(), static_cast<std::size_t>(count) * 4 * sizeof(float));
    FinishStream finish(stream); // 起動例外後も、バッファ解放より先に待つ。
    cdw::copy_host_to_device(device_queries.view(), std::as_bytes(std::span{queries}));
    const auto kernel = cdw::get_kernel(module, kernel_name);
    CUtexObject handle = texture.handle();
    CUdeviceptr query_pointer = device_queries.device_ptr(), result_pointer = device_result.device_ptr();
    unsigned int argument_count = count;
    void* arguments[]{&handle, &query_pointer, &result_pointer, &argument_count};
    cdw::LaunchConfig config;
    config.block_x = 64;
    config.grid_x = (count + 63u) / 64u;
    cdw::launch_kernel(kernel, config, stream, arguments);
    cdw::synchronize_stream(stream);
    std::vector<float> result(static_cast<std::size_t>(count) * 4);
    cdw::copy_device_to_host(std::as_writable_bytes(std::span{result}), device_result.view());
    return result;
}

void check_texture_description(
    const cdw::TextureObject& texture, const cdw::MipmappedArray2D& image,
    const cdw::MipmappedTexture2DDescriptor& settings
)
{
    cdw::ScopedCurrentContext current(texture.context());
    CUDA_RESOURCE_DESC resource{};
    CUDA_TEXTURE_DESC sampling{};
    cdw::check_cuda(cuTexObjectGetResourceDesc(&resource, texture.handle()), "cuTexObjectGetResourceDesc (mip test)");
    cdw::check_cuda(cuTexObjectGetTextureDesc(&sampling, texture.handle()), "cuTexObjectGetTextureDesc (mip test)");
    require(resource.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY &&
            resource.res.mipmap.hMipmappedArray == image.handle(), "Wrong mip resource descriptor");
    require((sampling.flags & CU_TRSF_NORMALIZED_COORDINATES) != 0, "Normalized coordinates not enabled");
    require(sampling.filterMode == settings.sampling.filter_mode &&
            sampling.mipmapFilterMode == settings.mipmap_filter_mode, "Wrong mip filtering configuration");
    require_near(sampling.minMipmapLevelClamp, settings.min_lod, "Minimum LOD descriptor");
    require_near(sampling.maxMipmapLevelClamp, settings.max_lod.value_or(static_cast<float>(image.level_count() - 1)),
                 "Maximum LOD descriptor");
}

void test_lod_sampling(CUcontext context, const cdw::Module& module)
{
    const auto before = cdw::get_current_context();
    for(const auto format : {CU_AD_FORMAT_UNSIGNED_INT8, CU_AD_FORMAT_FLOAT})
    for(unsigned int channels : {1u, 2u, 4u})
    for(unsigned int levels : {1u, 2u, 4u})
    {
        cdw::MipmappedArray2D image(context, {8, 4, format, channels}, levels);
        upload_constants(image);
        cdw::Stream stream(context);
        const std::vector<float> lods{-3.0f, 0.0f, 0.25f, 0.75f, 1.0f, 1.25f, 1.75f, 2.25f, 2.75f, 3.0f, 8.0f};
        std::vector<float> queries;
        for(float lod : lods) { queries.insert(queries.end(), {0.375f, 0.625f, lod}); }
        for(const auto mip_filter : {CU_TR_FILTER_MODE_POINT, CU_TR_FILTER_MODE_LINEAR})
        for(const auto spatial_filter : {CU_TR_FILTER_MODE_POINT, CU_TR_FILTER_MODE_LINEAR})
        {
            cdw::MipmappedTexture2DDescriptor settings;
            settings.sampling.filter_mode = spatial_filter;
            settings.mipmap_filter_mode = mip_filter;
            // case0:全確保レベル、case1:レベル0固定、case2:明示的な中間LOD範囲。
            for(int clamp_case = 0; clamp_case < 3; ++clamp_case)
            {
                settings.min_lod = 0.0f; settings.max_lod.reset();
                if(clamp_case == 1) { settings.max_lod = 0.0f; }
                if(clamp_case == 2)
                {
                    if(levels != 4) { continue; }
                    settings.min_lod = 1.25f; settings.max_lod = 2.25f;
                }
                cdw::TextureObject texture(image.view(), settings);
                check_texture_description(texture, image, settings);
                const auto output = sample(module, texture, stream, "cdw_mip_f" + std::to_string(channels), queries);
                for(std::size_t i = 0; i < lods.size(); ++i)
                {
                    float lod = std::clamp(lods[i], settings.min_lod,
                        settings.max_lod.value_or(static_cast<float>(levels - 1)));
                    // point の丸め境界ちょうどの LOD は選ばない。
                    if(mip_filter == CU_TR_FILTER_MODE_POINT) { lod = std::floor(lod + 0.5f); }
                    for(unsigned int channel = 0; channel < channels; ++channel)
                    {
                        const float expected = format == CU_AD_FORMAT_FLOAT
                            ? lod * 0.5f + static_cast<float>(channel) * 0.125f
                            : (lod * 64.0f + static_cast<float>(channel) * 7.0f) / 255.0f;
                        require_near(output[4 * i + channel], expected, "LOD sampling");
                    }
                }
                require_current(before);
            }
        }
        if(format == CU_AD_FORMAT_UNSIGNED_INT8)
        {
            cdw::MipmappedTexture2DDescriptor settings;
            settings.sampling.read_as_integer = true;
            cdw::TextureObject texture(image.view(), settings);
            const auto output = sample(module, texture, stream, "cdw_mip_u" + std::to_string(channels), queries);
            for(std::size_t i = 0; i < lods.size(); ++i)
            {
                const float lod = std::floor(std::clamp(lods[i], 0.0f, static_cast<float>(levels - 1)) + 0.5f);
                for(unsigned int channel = 0; channel < channels; ++channel)
                { require_near(output[4 * i + channel], lod * 64.0f + static_cast<float>(channel) * 7.0f, "Integer LOD"); }
            }
        }
    }
}

void test_spatial_and_mip_filter(CUcontext context, const cdw::Module& module)
{
    cdw::MipmappedArray2D image(context, {4, 4, CU_AD_FORMAT_FLOAT, 1}, 3);
    const std::array<std::vector<float>, 3> pixels{
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        std::vector<float>{10, 14, 18, 22}, std::vector<float>{30}
    };
    for(unsigned int level = 0; level < 3; ++level)
    {
        auto view = cdw::get_mip_level(image.view(), level);
        cdw::copy_host_to_array(view, cdw::make_full_array_region(view),
                               std::as_bytes(std::span{pixels[level]}), cdw::row_size_bytes(view.descriptor));
    }
    cdw::MipmappedTexture2DDescriptor settings;
    settings.sampling.filter_mode = CU_TR_FILTER_MODE_LINEAR;
    settings.mipmap_filter_mode = CU_TR_FILTER_MODE_LINEAR;
    cdw::TextureObject texture(image.view(), settings);
    cdw::Stream stream(context);
    const auto output = sample(module, texture, stream, "cdw_mip_f1",
                               {0.5f, 0.5f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 1.5f});
    require_near(output[0], 7.5f, "Spatial interpolation in level0");
    require_near(output[4], 11.75f, "Spatial and mip interpolation");
    require_near(output[8], 23.0f, "Level1 and level2 interpolation");
}

void test_invalid_sampling(CUcontext context)
{
    cdw::MipmappedArray2D image(context, {8, 4, CU_AD_FORMAT_FLOAT, 1}, 4);
    auto reject = [&](const cdw::MipmappedTexture2DDescriptor& settings)
    { require_throws<std::invalid_argument>([&] { cdw::TextureObject bad(image.view(), settings); }); };
    cdw::MipmappedTexture2DDescriptor s;
    s.sampling.normalized_coordinates = false; reject(s);
    s = {}; s.min_lod = -1; reject(s);
    s = {}; s.max_lod = 4; reject(s);
    s = {}; s.min_lod = 2; s.max_lod = 1; reject(s);
    s = {}; s.min_lod = (std::numeric_limits<float>::quiet_NaN)(); reject(s);
    s = {}; s.max_lod = (std::numeric_limits<float>::infinity)(); reject(s);
    s = {}; s.sampling.read_as_integer = true; reject(s);
    cdw::MipmappedArray2D integer_image(context, {8, 4}, 4);
    s = {}; s.sampling.read_as_integer = true; s.mipmap_filter_mode = CU_TR_FILTER_MODE_LINEAR;
    require_throws<std::invalid_argument>([&] { cdw::TextureObject bad(integer_image.view(), s); });
}

void test_async_upload(CUcontext context, const cdw::Module& module)
{
    cdw::MipmappedArray2D image(context, {8, 4, CU_AD_FORMAT_FLOAT, 1}, 4);
    std::vector<cdw::PinnedHostBuffer> staging;
    for(unsigned int level = 0; level < 4; ++level)
    {
        auto view = cdw::get_mip_level(image.view(), level);
        staging.emplace_back(context, view.descriptor.width * view.descriptor.height * sizeof(float));
    }
    cdw::TextureObject texture(image.view(), cdw::MipmappedTexture2DDescriptor{});
    cdw::Stream upload(context), consumer(context);
    cdw::Event ready(context);
    FinishStream finish_upload(upload), finish_consumer(consumer);
    for(unsigned int iteration = 0; iteration < 2; ++iteration)
    {
        for(unsigned int level = 0; level < 4; ++level)
        {
            const float value = static_cast<float>(iteration * 10 + level);
            std::vector<float> pixels(staging[level].size_bytes() / sizeof(float), value);
            std::memcpy(staging[level].bytes().data(), pixels.data(), staging[level].size_bytes());
            const auto view = cdw::get_mip_level(image.view(), level);
            cdw::copy_host_to_array_async(view, cdw::make_full_array_region(view), staging[level].view(),
                                         cdw::row_size_bytes(view.descriptor), upload);
        }
        cdw::record_event(ready, upload);
        cdw::stream_wait_event(consumer, ready);
        // sample() はconsumer完了まで待つので次の反復で参照中の配列を更新しない。
        const auto output = sample(module, texture, consumer, "cdw_mip_f1",
            {0.5f, 0.5f, 0, 0.5f, 0.5f, 1, 0.5f, 0.5f, 2, 0.5f, 0.5f, 3});
        cdw::synchronize_event(ready);
        for(unsigned int level = 0; level < 4; ++level)
        {
            require_near(output[4 * level], static_cast<float>(iteration * 10 + level), "Async mip upload");
            std::fill(staging[level].bytes().begin(), staging[level].bytes().end(), std::byte{0});
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    static_assert(!std::is_copy_constructible_v<cdw::MipmappedArray2D>);
    static_assert(std::is_nothrow_move_constructible_v<cdw::MipmappedArray2D>);
    static_assert(!std::is_nothrow_move_assignable_v<cdw::MipmappedArray2D>);
    static_assert(std::is_nothrow_destructible_v<cdw::MipmappedArray2D>);
    static_assert(std::is_convertible_v<cdw::MipmappedArray2DView, cdw::ConstMipmappedArray2DView>);
    static_assert(!std::is_convertible_v<cdw::ConstMipmappedArray2DView, cdw::MipmappedArray2DView>);
    static_assert(std::is_same_v<decltype(cdw::get_mip_level(std::declval<cdw::ConstMipmappedArray2DView>(), 0)),
                                 cdw::ConstArray2DView>);
    try
    {
        test_metadata();
        if(argc == 2 && std::string_view(argv[1]) == "--metadata")
        {
            std::cout << "All mipmap metadata tests passed.\n";
            return 0;
        }
        if(argc != 2) { throw std::invalid_argument("Usage: mipmapped_array_test <fatbin> | --metadata"); }
        int count = 0;
        try { cdw::initialize_driver(); count = cdw::get_device_count(); }
        catch(const cdw::CudaError& error)
        {
            if(error.result() != CUDA_ERROR_NO_DEVICE) { throw; }
            std::cout << "SKIP: no CUDA device.\n"; return 77;
        }
        if(count == 0) { std::cout << "SKIP: no visible CUDA device.\n"; return 77; }
        const CUdevice device = cdw::get_device(0);
        std::cout << "Mipmap tests on: " << cdw::get_device_info(device).name << '\n';
        const auto before = cdw::get_current_context();
        {
            cdw::RetainPrimaryContext primary(device);
            const auto context = primary.handle();
            cdw::Module module(context, std::filesystem::path(argv[1]));
            std::cout << "[mipmap] storage and ownership\n";
            test_storage(context); test_ownership(context); test_invalid_sampling(context);
            std::cout << "[mipmap] LOD and filtering\n";
            test_lod_sampling(context, module); test_spatial_and_mip_filter(context, module);
            std::cout << "[mipmap] asynchronous update\n";
            test_async_upload(context, module);
            {
                ExternalContext other(device);
                cdw::ScopedCurrentContext current(other.handle());
                test_ownership(context);
                test_spatial_and_mip_filter(context, module);
                require_current(other.handle());
                current.restore();
            }
            require_current(before);
        }
        require_current(before);
        std::cout << "All mipmapped array tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
