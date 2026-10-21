#pragma once

#include <cuda_driver_wrapper/pinned_host_memory.hpp>

#include <cstddef>
#include <source_location>
#include <span>

namespace cuda_driver_wrapper
{

class PinnedHostBuffer final
{
public:
    PinnedHostBuffer() noexcept = default;

    explicit PinnedHostBuffer(
        CUcontext context,
        std::size_t size_bytes,
        std::source_location location = std::source_location::current()
    );

    ~PinnedHostBuffer() noexcept;

    // コピーの禁止
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    
    // move は許可
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    // move 先の既存の allocation の close が失敗する可能性があるので，noexcept としない
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other);

    void close(std::source_location location = std::source_location::current());

    [[nodiscard]] CUcontext context() const noexcept {return context_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept {return size_bytes_; }
    [[nodiscard]] bool empty() const noexcept {return size_bytes_ == 0; }

    // CPU でバイト列を読み書きする view 
    [[nodiscard]] std::span<std::byte> bytes() & noexcept
    {
        return {data_, size_bytes_};
    }

    [[nodiscard]] std::span<const std::byte> bytes() const & noexcept
    {
        return {data_, size_bytes_};
    }
    
    std::span<std::byte> bytes () && = delete;
    std::span<const std::byte> bytes() const && = delete;

    // GPU への転送時に渡す非所有の view
    // CPU の bytes() と区別する
    [[nodiscard]] PinnedHostMemoryView view() & noexcept
    {
        return {context_, data_, size_bytes_};
    }

    [[nodiscard]] ConstPinnedHostMemoryView view() const & noexcept
    {
        return {context_, data_, size_bytes_};
    }

    PinnedHostMemoryView view() && = delete;
    ConstPinnedHostMemoryView view() const && = delete;

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;
    std::byte* data_ = nullptr;
    std::size_t size_bytes_ = 0;
};

} // namespace cuda_driver_wrapper