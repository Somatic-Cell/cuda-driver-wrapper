#include <cuda_driver_wrapper/pinned_host_buffer.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

PinnedHostBuffer::PinnedHostBuffer(
    const CUcontext context,
    const std::size_t size_bytes,
    const std::source_location location
) : context_(context)
{
    if(context_ == nullptr)
    {
        throw std::invalid_argument("PinnedHostBuffer requires a non-null context");
    }
    if(size_bytes == 0)
    {
        return;
    }

    ScopedCurrentContext current(context_, location);
    void* created = nullptr;
    check_cuda(
        cuMemHostAlloc(&created, size_bytes, 0),
        "cuMemHostAlloc", location
    );
    data_ = static_cast<std::byte*>(created);
    size_bytes_ = size_bytes;

    try
    {
        current.restore(location);
    }
    catch(...)
    {
        // コンストラクタが失敗すると自分のデストラクタは呼ばれないため
        close_noexcept();
        throw;
    }
}

PinnedHostBuffer::~PinnedHostBuffer() noexcept
{
    close_noexcept();
}

// move
PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer && other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
        data_(std::exchange(other.data_, nullptr)),
        size_bytes_(std::exchange(other.size_bytes_, 0))
{
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other)
{
    if(this != &other)
    {
        // 自身の要素を解放する
        close();

        // other を解放しつつ，自身の要素とする
        context_ = std::exchange(other.context_, nullptr);
        data_ = std::exchange(other.data_, nullptr);
        size_bytes_ = std::exchange(other.size_bytes_, 0);
    }
    return *this;
}

void PinnedHostBuffer::close(const std::source_location location)
{
    if(data_ == nullptr)
    {
        context_ = nullptr;
        size_bytes_ = 0;
        return;
    }

    ScopedCurrentContext current(context_, location);
    check_cuda(
        cuMemFreeHost(data_), 
        "cuMemFreeHost", location
    );

    // Free の成功を先に反映し，復元失敗時の二重 free を防ぐ
    data_ = nullptr;
    context_ = nullptr;
    size_bytes_ = 0;
    current.restore(location);
}

void PinnedHostBuffer::close_noexcept() noexcept
{
    if(data_ == nullptr)
    {
        return;
    }

    const CUcontext context = std::exchange(context_, nullptr);
    std::byte* const data = std::exchange(data_, nullptr);
    size_bytes_ = 0;
    try
    {
        {
            ScopedCurrentContext current(context);
            static_cast<void>(report_cuda_cleanup_result(cuMemFreeHost(data), "cuMemFreeHost"));
            current.restore();
        }
    }
    catch(const std::exception& error)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper pinned host buffer cleanup] %s\n", error.what()
        ));
    }
    catch(...)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper pinned host buffer cleanup] unknown failure\n"
        ));
    }
    
}

} // namespace cuda_driver_wrapper