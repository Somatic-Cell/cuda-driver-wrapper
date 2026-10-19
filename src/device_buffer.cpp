#include <cuda_driver_wrapper/device_buffer.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

DeviceBuffer::DeviceBuffer(
    const CUcontext context,
    const std::size_t size_bytes,
    const std::source_location location
) : context_(context)
{
    // context が nullptr だった場合はサイズに関係なく拒否
    if(context_ == nullptr)
    {
        throw std::invalid_argument("DeviceBuffer requires a non-null context");
    }
    // 0 バイトだった場合は空のバッファとして扱い，cuMemAlloc　を呼ばない
    if(size_bytes == 0)
    {
        return;
    }

    ScopedCurrentContext current(context_, location);
    CUdeviceptr created = 0;
    check_cuda(
        cuMemAlloc(&created, size_bytes),
        "cuMemAlloc", location
    );

    data_ = created;
    size_bytes_ = size_bytes;

    try
    {
        current.restore(location);
    }
    catch(...)
    {
        // コンストラクタの失敗時にはデストラクタは呼ばれないので
        // 取得済みの allocation はここで後始末する
        close_noexcept();
        throw;
    }
}

DeviceBuffer::~DeviceBuffer() noexcept
{
    close_noexcept();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
        data_(std::exchange(other.data_, 0)),
        size_bytes_(std::exchange(other.size_bytes_, 0))
{
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other)
{
    if(this != &other)
    {
        close();
        context_ = std::exchange(other.context_, nullptr);
        data_ = std::exchange(other.data_, 0);
        size_bytes_ = std::exchange(other.size_bytes_, 0);
    }
    return *this;
}

void DeviceBuffer::close(const std::source_location location)
{
    if(data_ == 0)
    {
        context_ = nullptr;
        size_bytes_ = 0;
        return;
    }

    ScopedCurrentContext current(context_, location);
    check_cuda(cuMemFree(data_), "cuMemFree", location);

    // Free の成功を先に反映し，もし restore で例外が出ても二重開放しない
    data_ = 0;
    context_ = nullptr;
    size_bytes_ = 0;
    current.restore(location);
}

void DeviceBuffer::close_noexcept() noexcept
{
    if(data_ == 0)
    {
        return;
    }

    const CUcontext context = std::exchange(context_, nullptr);
    const CUdeviceptr data = std::exchange(data_, 0);
    size_bytes_ = 0;
    try 
    {
        ScopedCurrentContext current(context);
        static_cast<void>(report_cuda_cleanup_result(cuMemFree(data), "cuMemFree"));
        current.restore();
    }
    catch(const std::exception& error)
    {
        static_cast<void>(
            std::fprintf(
                stderr, "[cuda_driver_wrapper device buffer cleanup] %s\n", error.what()
            )
        );
    }
    catch(...)
    {
        static_cast<void>(
            std::fprintf(
                stderr, "[cuda_driver_wrapper device buffer cleanup] unknown failure\n"
            )
        );
    }
}

} // namespace cuda_driver_wrapper