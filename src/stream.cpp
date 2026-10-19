#include <cuda_driver_wrapper/stream.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

namespace
{

void require_stream(const Stream& stream)
{
    if(!stream.is_valid())
    {
        throw std::logic_error("Stream is closed or moved-from");
    }
}

} // namespace

Stream::Stream(
    const CUcontext context,
    const unsigned int flags,
    const std::source_location location
) : context_(context)
{
    ScopedCurrentContext current(context_, location);
    CUstream created = nullptr;
    check_cuda(cuStreamCreate(&created, flags), "cuStreamCreate", location);
    stream_ = created;

    try
    {
        current.restore(location);
    }
    catch(...)
    {
        close_noexcept();
        throw;
    }
}

Stream::~Stream() noexcept
{
    close_noexcept();
}

// ムーブコンストラクタ
Stream::Stream(Stream&& other) noexcept
    :   context_(std::exchange(other.context_, nullptr)),   // ムーブされた other.context_ は空にする
        stream_(std::exchange(other.stream_, nullptr))      
{
}

// ムーブ代入
Stream& Stream::operator=(Stream&& other)
{
    if(this != &other)
    {
        close(); // 自分のリソースを解放．失敗したら other の所有権は自分に移さない
        context_ = std::exchange(other.context_, nullptr);
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

void Stream::close(const std::source_location location)
{
    if(!is_valid())
    {
        return;
    }

    ScopedCurrentContext current(context_, location);
    check_cuda(cuStreamDestroy(stream_), "cuStreamDestroy", location);

    stream_ = nullptr;
    context_ = nullptr;
    current.restore(location);
}

void Stream::close_noexcept() noexcept
{
    if(!is_valid())
    {
        return;
    }

    const CUcontext context = std::exchange(context_, nullptr);
    const CUstream stream = std::exchange(stream_, nullptr);
    try
    {
        {
            ScopedCurrentContext current(context);
            static_cast<void>(report_cuda_cleanup_result(
                cuStreamDestroy(stream),
                "cuStreamDestroy"
            ));
            current.restore();
        }
    }
    catch(const std::exception& e)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper stream cleanup] %s \n", e.what()
        ));
    }
    catch(...)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper stream cleanup] unknown failure\n"
        ));
    }
}

void synchronize_stream(
    const Stream& stream,
    const std::source_location location
)
{
    require_stream(stream);
    ScopedCurrentContext current(stream.context(), location);
    check_cuda(cuStreamSynchronize(stream.handle()), "cuStreamSynchronize", location);
    current.restore(location);
}

bool is_stream_ready(
    const Stream& stream,
    const std::source_location location
)
{
    require_stream(stream);
    ScopedCurrentContext current(stream.context(), location);
    const CUresult result = cuStreamQuery(stream.handle());
    if(result != CUDA_ERROR_NOT_READY)
    {
        check_cuda(result, "cuStreamQuery", location);
    }
    current.restore(location);
    return result == CUDA_SUCCESS;
}


} // namespace cuda_driver_wrapper