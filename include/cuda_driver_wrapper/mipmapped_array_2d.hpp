#pragma once

#include <cuda_driver_wrapper/array_2d_memory.hpp>

#include <cstddef>
#include <source_location>

namespace cuda_driver_wrapper{

// 2D mip chain のディスクリプタ
// 読み取り専用
struct ConstMipmappedArray2DView
{
    CUcontext context = nullptr;
    CUmipmappedArray array = nullptr;
    Array2DDescriptor base_descriptor {};
    unsigned int level_count = 0;
};

// 読み書き可能
struct MipmappedArray2DView
{
    CUcontext context = nullptr;
    CUmipmappedArray array = nullptr;
    Array2DDescriptor base_descriptor {};
    unsigned int level_count = 0;

    // 読み取り専用への変換は許す
    [[nodiscard]] operator ConstMipmappedArray2DView() const noexcept
    {
        return {context, array, base_descriptor, level_count};
    }
};

// 1x1 までの完全な chain のレベル数を計算する関数
// レベルゼロを含んだ個数を計算
[[nodiscard]] unsigned int full_mip_level_count(
    std::size_t width,
    std::size_t height
);

// レベルの番号は [0, level_count) とする
[[nodiscard]] Array2DView get_mip_level(
    MipmappedArray2DView source,
    unsigned int level,
    std::source_location = std::source_location::current()
);

[[nodiscard]] ConstArray2DView get_mip_level(
    ConstMipmappedArray2DView source,
    unsigned int level,
    std::source_location = std::source_location::current()
);

// cuMipmappedArrayCreate で作る chain を一つ所有する
// context は借用する
class MipmappedArray2D final
{
public:
    MipmappedArray2D() noexcept = default;
    
    explicit MipmappedArray2D(
        CUcontext context,
        Array2DDescriptor base_descriptor,
        unsigned int level_count,
        std::source_location location = std::source_location::current()
    );

    ~MipmappedArray2D() noexcept;

    // コピーの禁止
    MipmappedArray2D(const MipmappedArray2D&) = delete;
    MipmappedArray2D& operator=(const MipmappedArray2D&) = delete;

    // ムーブは OK
    MipmappedArray2D(MipmappedArray2D&& other) noexcept;
    MipmappedArray2D& operator=(MipmappedArray2D && other);

    // 成功後は，親と全レベルの借用ハンドルが失効する
    void close(std::source_location location = std::source_location::current());

    [[nodiscard]] CUcontext context() const noexcept { return context_; }
    [[nodiscard]] CUmipmappedArray handle() const noexcept {return array_; }
    [[nodiscard]] Array2DDescriptor base_descriptor() const noexcept { return base_descriptor_; }
    [[nodiscard]] unsigned int level_count() const noexcept { return level_count_; }
    [[nodiscard]] bool is_valid() const noexcept { return array_ != nullptr; }
    
    // 借用のみ
    [[nodiscard]] MipmappedArray2DView view() & noexcept 
    {
        return {context_, array_, base_descriptor_, level_count_}; 
    }
    
    [[nodiscard]] ConstMipmappedArray2DView view() const & noexcept 
    {
        return {context_, array_, base_descriptor_, level_count_}; 
    }
    
    MipmappedArray2DView view() && = delete;
    ConstMipmappedArray2DView view() const && = delete;

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;
    CUmipmappedArray array_ = nullptr;
    Array2DDescriptor base_descriptor_{};
    unsigned int level_count_ = 0;
};

} // namespace cuda_driver_wrapper