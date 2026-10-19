#include <cuda_driver_wrapper/device_memory.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

namespace
{

void require_view(const ConstDeviceMemoryView memory)
{
    if(memory.size_bytes != 0 && memory.data == 0){
        throw std::invalid_argument("A non-empty device view requires a non-zero address");
    }
    if(memory.data != 0 && memory.context == nullptr)
    {
        throw std::invalid_argument("A device address requires a non-null context");
    }

    // view の終端アドレスまで CUdeviceptr で表現できる範囲に限定する
    const CUdeviceptr maximum = (std::numeric_limits<CUdeviceptr>::max)();
    if(std::cmp_greater(memory.size_bytes, maximum - memory.data))
    {
        throw std::overflow_error("Device view address range overflows CUdeviceptr");
    }
}

void require_equal_sizes(const std::size_t destination, const std::size_t source)
{
    if(destination != source)
    {
        throw std::invalid_argument("Memory copy requires equal source and destination byte sizes");
    }
}

void require_stream(const Stream& stream)
{
    if(!stream.is_valid())
    {
        throw std::logic_error("Stream is closed or move-from");
    }
}

void require_stream_context(const ConstDeviceMemoryView memory, const Stream& stream)
{
    if(memory.context != nullptr && memory.context != stream.context())
    {
        throw std::invalid_argument("Device memory and stream must belong to the same context");
    }
}

} // namespace

// 要素数からバイト数を計算する
// もしも扱える上限を超えていればエラーを送出
std::size_t checked_byte_size(const std::size_t count, const std::size_t element_size)
{
    if(element_size == 0)
    {
        throw std::invalid_argument("element_size must be greater than zero");
    }
    if(count > (std::numeric_limits<std::size_t>::max)() / element_size)
    {
        throw std::length_error("Element cont times element size overflows size_t");
    }
    return count * element_size;
}

ConstDeviceMemoryView subview(
    const ConstDeviceMemoryView memory,
    const std::size_t offset_bytes,
    const std::size_t count_bytes
)
{
    require_view(memory);
    if(offset_bytes > memory.size_bytes || count_bytes > memory.size_bytes - offset_bytes)
    {
        throw std::out_of_range("Device subview is outside the supplied byte range");
    }
    return {memory.context, memory.data + static_cast<CUdeviceptr>(offset_bytes), count_bytes};
}

DeviceMemoryView subview(
    const DeviceMemoryView memory,
    const std::size_t offset_bytes,
    const std::size_t count_bytes
)
{
    const auto result = subview(static_cast<ConstDeviceMemoryView>(memory), offset_bytes, count_bytes);
    return {result.context, result.data, result.size_bytes};
}

void copy_host_to_device(
    const DeviceMemoryView destination,
    const std::span<const std::byte> source,
    const std::source_location location
)
{
    require_view(destination);
    require_equal_sizes(destination.size_bytes, source.size_bytes());
    if(source.empty())
    {
        return;
    }

    ScopedCurrentContext current(destination.context, location);
    check_cuda(
        cuMemcpyHtoD(destination.data, source.data(), source.size_bytes()),
        "cuMemcpyHtoD",
        location
    );

    // pageable host memory では， cuMemcpyHtoD が DMA 完了前に戻る場合がある
    // 同じ翻訳単位の default stream をもち，正常終了時の転送完了を保証する
    // 別の non-blocking stream の先行アクセスを待つ処理ではない
    check_cuda(
        cuStreamSynchronize(nullptr),
        "cuStreamSynchronize (upload completion)",
        location
    );
    current.restore(location);
}

void copy_device_to_host(
    const std::span<std::byte> destination,
    const ConstDeviceMemoryView source,
    const std::source_location location
)
{
    require_view(source);
    require_equal_sizes(destination.size_bytes(), source.size_bytes);
    if(destination.empty())
    {
        return;
    }
    ScopedCurrentContext current(source.context, location);
    check_cuda(
        cuMemcpyDtoH(destination.data(), source.data, source.size_bytes),
        "cuMemcpyDtoH",
        location
    );
    current.restore(location);
}

void copy_device_to_device_async(
    const DeviceMemoryView destination,
    const ConstDeviceMemoryView source,
    const Stream& stream,
    const std::source_location location
)
{
    require_view(destination);
    require_view(source);
    require_equal_sizes(destination.size_bytes, source.size_bytes);
    require_stream(stream);
    require_stream_context(destination, stream);
    require_stream_context(source, stream);
    if(source.size_bytes == 0 || destination.data == source.data)
    {
        return;
    }

    const CUdeviceptr distance = destination.data < source.data 
        ? source.data - destination.data : destination.data - source.data;
    if(std::cmp_less(distance, source.size_bytes))
    {
        throw std::invalid_argument("Device copy ranges partially overlap");
    }

    ScopedCurrentContext current(stream.context(), location);
    check_cuda(
        cuMemcpyDtoDAsync(destination.data, source.data, source.size_bytes, stream.handle()),
        "cuMemcpyDtoDAsync", location
    );

    current.restore(location);
}

void fill_device_bytes_async(
    const DeviceMemoryView destination,
    const unsigned char value,
    const Stream& stream,
    const std::source_location location
)
{
    require_view(destination);
    require_stream(stream);
    require_stream_context(destination, stream);
    if(destination.size_bytes == 0)
    {
        return;
    }

    ScopedCurrentContext current(stream.context(), location);
    check_cuda(
        cuMemsetD8Async(destination.data, value, destination.size_bytes, stream.handle()),
        "cuMemsetD8Async", location
    );
    current.restore(location);
}

} // namespace cuda_driver_wrapper
