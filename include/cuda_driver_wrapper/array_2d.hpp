#pragma once

#include <cuda_driver_wrapper/array_2d_memory.hpp>

namespace cuda_driver_wrapper{

// cuArrayCreate で作る 2D CUDA Array を一つ所有する
// context は借用する
class Array2D final
{
public:
    Array2D() noexcept = default;
    
    explicit Array2D(
        CUcontext context,
        Array2DDescriptor descriptor,
        std::source_location location = std::source_location::current()
    );

    ~Array2D() noexcept;

    // コピーの禁止
    Array2D(const Array2D&) = delete;
    Array2D& operator=(const Array2D&) = delete;

    // ムーブは OK
    Array2D(Array2D&& other) noexcept;
    Array2D& operator=(Array2D && other);

    void close(std::source_location location = std::source_location::current());

    [[nodiscard]] CUcontext context() const noexcept { return context_; }
    [[nodiscard]] CUarray handle() const noexcept {return array_; }
    [[nodiscard]] bool is_valid() const noexcept { return array_ != nullptr; }
    [[nodiscard]] Array2DDescriptor descriptor() const noexcept { return descriptor_; }
    
    // 借用のみ
    [[nodiscard]] Array2DView view() & noexcept {return {context_, array_, descriptor_}; }
    [[nodiscard]] ConstArray2DView view() const & noexcept {return {context_, array_, descriptor_}; }
    Array2DView view() && = delete;
    ConstArray2DView view() const && = delete;

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;
    CUarray array_ = nullptr;
    Array2DDescriptor descriptor_{};
};

} // namespace cuda_driver_wrapper