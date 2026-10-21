#include <cuda_driver_wrapper/pinned_host_memory.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

namespace {

void require_host_view(const ConstPinnedHostMemoryView memory)
{
    // 
    if(memory.size_bytes != 0 && memory.data  == nullptr)
    {
        throw std::invalid_argument("A non-empty pinned host view requires a non-null address");
    }
    
    // 
    if(memory.data != nullptr && memory.context == nullptr)
    {
        throw std::invalid_argument("A pinned host address requires a non-null context");
    }

    // 終端アドレスの整数オーバフローを検出
    if(memory.data != nullptr)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(memory.data);
        const auto maximum = (std::numeric_limits<std::uintptr_t>::max)();
        if(std::cmp_greater(memory.size_bytes, maximum - address))
        {
            throw std::overflow_error("Pinned host view address range overflows uintptr_t");
        }
    }
}


// CPU <-> GPU のコピー時に不正がないかどうかをチェック
void require_copy(
    const ConstDeviceMemoryView device_memory,
    const ConstPinnedHostMemoryView host_memory,
    const Stream& stream
)
{
    static_cast<void>(subview(device_memory, 0, device_memory.size_bytes));
    require_host_view(host_memory);

    if(device_memory.size_bytes != host_memory.size_bytes)
    {
        throw std::invalid_argument("Async host copy requires equal source and destination byte sizes");
    }
    if(!stream.is_valid())
    {
        throw std::logic_error("Stream is close or moved from");
    }
    if((device_memory.context != nullptr && device_memory.context != stream.context()) ||
        (host_memory.context != nullptr && host_memory.context != stream.context()))
    {
        throw std::invalid_argument("host memory, device memory and stream must use the same context");
    }
}

} // namespace 

ConstPinnedHostMemoryView subview(
    const ConstPinnedHostMemoryView memory,
    const std::size_t offset_bytes,
    const std::size_t count_bytes
)
{
    require_host_view(memory);
    if(offset_bytes > memory.size_bytes || count_bytes > memory.size_bytes - offset_bytes)
    {
        throw std::out_of_range("Pinned host subview is outside ther supplied byte range");
    }

    // 空の view に対する nullptr + 0 を避ける
    const std::byte* data = memory.data;
    if(offset_bytes != 0)
    {
        data += offset_bytes;
    }
    return {memory.context, data, count_bytes};
}

PinnedHostMemoryView subview(
    const PinnedHostMemoryView memory,
    const std::size_t offset_bytes,
    const std::size_t count_bytes
)
{
    // const 側で検査を済ませ，mutable な元のアドレスから範囲を作る
    static_cast<void>(subview(
        static_cast<ConstPinnedHostMemoryView>(memory), offset_bytes, count_bytes
    ));
    std::byte* data = memory.data;
    if(offset_bytes != 0)
    {
        data += offset_bytes;
    }
    return {memory.context, data, count_bytes};
}

// CPU -> GPU
void copy_host_to_device_async(
    const DeviceMemoryView destination,
    const ConstPinnedHostMemoryView source,
    const Stream& stream,
    const std::source_location location
)
{
    require_copy(destination, source, stream);
    if(source.size_bytes == 0)
    {
        return;
    }

    ScopedCurrentContext current(stream.context(), location);
    check_cuda(
        cuMemcpyHtoDAsync(destination.data, source.data, source.size_bytes, stream.handle()),
        "cuMemcpyHtoDAsync", location
    );
    // 完了まで待機せず，stream に投入したらコンテキストを復元して即座に修了
    current.restore(location);
}

// GPU -> CPU
void copy_device_to_host_async(
    const PinnedHostMemoryView destination,
    const ConstDeviceMemoryView source,
    const Stream& stream,
    const std::source_location location
)
{
    require_copy(source, destination, stream);
    if(source.size_bytes == 0)
    {
        return;
    }

    ScopedCurrentContext current(stream.context(), location);
    check_cuda(
        cuMemcpyDtoHAsync(destination.data, source.data, source.size_bytes, stream.handle()),
        "cuMemcpyDtoHAsync", location
    );
    current.restore(location);
}


} // namespace cuda_driver_wrapper