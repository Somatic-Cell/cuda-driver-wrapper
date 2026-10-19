#include <cuda_driver_wrapper/event.hpp>
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

void require_event(const Event& event)
{
    if(!event.is_valid())
    {
        throw std::logic_error("Event is closed or moved-from");
    }
}

void require_stream(const Stream& stream)
{
    if(!stream.is_valid())
    {
        throw std::logic_error("Stream is closed or moved-from");
    }
}

} // namespace

Event::Event(
    const CUcontext context,
    const unsigned int flags,
    const std::source_location location
) : context_(context)
{
    ScopedCurrentContext current(context_, location);
    CUevent created = nullptr;
    check_cuda(cuEventCreate(&created, flags), "cuEventCreate", location);
    event_ = created;

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

Event::~Event() noexcept
{
    close_noexcept();
}

// ムーブ
Event::Event(Event&& other) noexcept
    :   context_(std::exchange(other.context_, nullptr)),
        event_(std::exchange(other.event_, nullptr))
{
}

Event& Event::operator=(Event&& other)
{
    if(this != &other)
    {
        close();
        context_ = std::exchange(other.context_, nullptr);
        event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
}

void Event::close(std::source_location location)
{
    if(!is_valid())
    {
        return;
    }

    ScopedCurrentContext current(context_, location);
    check_cuda(cuEventDestroy(event_), "cuEventDestroy", location);
    event_ = nullptr;
    context_ = nullptr;
    current.restore(location);
}

void Event::close_noexcept() noexcept
{
    if(!is_valid())
    {
        return;
    }

    const CUcontext context = std::exchange(context_, nullptr);
    const CUevent event = std::exchange(event_, nullptr);
    try
    {
        ScopedCurrentContext current(context);
        static_cast<void>(report_cuda_cleanup_result(
            cuEventDestroy(event), "cuEventDestroy"
        ));
        current.restore();
    }
    catch(const std::exception& e)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper event cleanup] %s\n", e.what()
        ));
    }
    catch(...)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper event cleanup] unknown failure\n"
        ));
    }
}

void record_event(
    Event& event,
    const Stream& stream,
    const std::source_location location
)
{
    require_event(event);
    require_stream(stream);
    if(event.context() != stream.context())
    {
        throw std::invalid_argument("record_event requires the same context");
    }

    ScopedCurrentContext current(stream.context(), location);
    check_cuda(cuEventRecord(event.handle(), stream.handle()), "cuEventRecord", location);
    current.restore(location);
}

void synchronize_event(
    const Event& event,
    const std::source_location location
)
{
    require_event(event);
    ScopedCurrentContext current(event.context(), location);
    check_cuda(cuEventSynchronize(event.handle()), "cuEventSynchronize", location);
    current.restore(location);
}

bool is_event_ready(
    const Event& event,
    const std::source_location location
)
{
    require_event(event);
    ScopedCurrentContext current(event.context(), location);
    const CUresult result = cuEventQuery(event.handle());
    if(result != CUDA_ERROR_NOT_READY)
    {
        check_cuda(result, "cuEventQuery", location);
    }
    current.restore(location);
    return result == CUDA_SUCCESS;
}

void stream_wait_event(
    const Stream& stream,
    const Event& event,
    const std::source_location location
)
{
    require_stream(stream);
    require_event(event);
    ScopedCurrentContext current(stream.context(), location);

    // Wait は Record と異なり，別 context の event にも対応する
    check_cuda(cuStreamWaitEvent(stream.handle(), event.handle(), 0), "cuStreamWaitEvent", location);
    current.restore(location);
}

float elapsed_time_ms(
    const Event& start,
    const Event& end,
    const std::source_location location
)
{
    require_event(start);
    require_event(end);
    if(start.context() != end.context())
    {
        throw std::invalid_argument("elapsed_time_ms requires the same context");
    }

    ScopedCurrentContext current(start.context(), location);
    float milliseconds = 0.0f;
    check_cuda(
        cuEventElapsedTime(&milliseconds, start.handle(), end.handle()), 
        "cuEventElapsedTime",
        location
    );

    current.restore(location);
    return milliseconds;
}

} // namespace cuda_driver_wrapper