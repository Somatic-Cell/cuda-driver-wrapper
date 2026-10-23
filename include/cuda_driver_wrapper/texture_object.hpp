#pragma once

#include <cuda_driver_wrapper/array_2d_memory.hpp>

#include <optional>
#include <source_location>

namespace cuda_driver_wrapper
{
struct ConstMipmappedArray2DView;

// 2D array を読むためのディスクリプタ
struct Texture2DDescriptor
{
    CUaddress_mode address_mode_u = CU_TR_ADDRESS_MODE_CLAMP;
    CUaddress_mode address_mode_v = CU_TR_ADDRESS_MODE_CLAMP;
    CUfilter_mode filter_mode = CU_TR_FILTER_MODE_POINT;

    // true: [0, 1) が画像の全体
    // false: [0, width), [0, height) が画像の全体
    bool normalized_coordinates = true;

    // UINT8: false は [0, 1] の float, true は元の整数値を返す
    // Float: false のみ許可
    // 整数値で読み出すときは point のみ許可する
    bool read_as_integer = false;
};

struct MipmappedTexture2DDescriptor
{
    Texture2DDescriptor sampling{};
    CUfilter_mode mipmap_filter_mode = CU_TR_FILTER_MODE_POINT;
    float min_lod = 0.0f;

    // 未指定なら確保した最後のレベルとする
    std::optional<float> max_lod = std::nullopt;

    // 近似的な trilinear interpolation をデフォルトで無効にする
    bool disable_trilinear_optimization = true;
};

// CUtexObject を所有する
// CUarray, CUcontext は借用する
// 構築時にテクスチャの転送，複製， context の retain, GPU の同期を行わない
class TextureObject final
{
public:
    TextureObject() noexcept = default;
    explicit TextureObject(
        ConstArray2DView source,
        Texture2DDescriptor descriptor = {},
        std::source_location location = std::source_location::current()
    );

    TextureObject(
        ConstMipmappedArray2DView source,
        MipmappedTexture2DDescriptor descriptor = {},
        std::source_location location = std::source_location::current()
    );

    ~TextureObject() noexcept;

    // コピーの禁止
    TextureObject(const TextureObject&) = delete;
    TextureObject& operator=(const TextureObject&) = delete;

    // move は許可
    TextureObject(TextureObject&& other) noexcept;
    TextureObject& operator=(TextureObject&& other);

    void close(std::source_location location = std::source_location::current());

    // 借用のためのハンドル
    // 呼び出し側からは destroy してはいけない
    // tex2D<T> の T は array の成分数と read_as_integer との指定に合わせる
    [[nodiscard]] CUtexObject handle() const noexcept { return texture_; }
    [[nodiscard]] CUcontext context() const noexcept { return context_; }
    [[nodiscard]] bool is_valid() const noexcept { return texture_ != 0; }

private:
    void create(
        const CUDA_RESOURCE_DESC& resource,
        const CUDA_TEXTURE_DESC& sampling,
        std::source_location location
    );
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;   // 借用する
    CUtexObject texture_ = 0;       // 所有する．整数ハンドルであってポインタではない

};

} // namespace cuda_driver_wrapper