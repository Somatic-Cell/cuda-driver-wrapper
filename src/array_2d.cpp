#include <cuda_driver_wrapper/array_2d.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

Array2D::Array2D(
    const CUcontext context,
    const Array2DDescriptor descriptor,
    const std::source_location location
) : context_(context), descriptor_(descriptor)
{
    if(context_ == nullptr)
    {
        throw std::invalid_argument("Array2D requires a non-null context");
    }

    if(descriptor_.height == 0)
    {
        throw std::invalid_argument("Array2D height must be non-zero");
    }

    static_cast<void>(row_size_bytes(descriptor_));

    CUDA_ARRAY_DESCRIPTOR allocation{};
    allocation.Width = descriptor_.width;
    allocation.Height = descriptor_.height;
    allocation.Format = descriptor_.format;
    allocation.NumChannels = descriptor_.num_channels;

    ScopedCurrentContext current(context_, location);
    CUarray created = nullptr;
    check_cuda(
        cuArrayCreate(&created, &allocation),
        "cuArrayCreate",
        location
    );
    array_ = created;
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

Array2D::~Array2D() noexcept
{
    close_noexcept();
}

Array2D::Array2D(Array2D&& other) noexcept
    :   context_(std::exchange(other.context_, nullptr)),
        array_(std::exchange(other.array_, nullptr)),
        descriptor_(std::exchange(other.descriptor_, Array2DDescriptor{}))
{
}

Array2D& Array2D::operator=(Array2D&& other)
{
    if(this != & other)
    {
        close();
        context_ = std::exchange(other.context_, nullptr);
        array_ = std::exchange(other.array_, nullptr);
        descriptor_ = std::exchange(other.descriptor_, Array2DDescriptor{});
    }
    return *this;
}

void Array2D::close(const std::source_location location)
{
    if(!is_valid()) { return; }
    ScopedCurrentContext current(context_, location);
    check_cuda(
        cuArrayDestroy(array_), 
        "cuArrayDestroy", location
    );
    array_ = nullptr;
    context_ = nullptr;
    descriptor_ = {};
    current.restore(location);
}

void Array2D::close_noexcept() noexcept
{
    if(!is_valid()){ return; }
    
    const CUcontext context = std::exchange(context_, nullptr);
    const CUarray array = std::exchange(array_, nullptr);
    descriptor_ = {};

    try
    {
        ScopedCurrentContext current(context);
        static_cast<void>(report_cuda_cleanup_result(
            cuArrayDestroy(array), 
            "cuArrayDestroy"
        ));
        current.restore();
    } 
    catch(const std::exception& error)
    {
        static_cast<void>(std::fprintf(stderr, "[cuda_driver_wrapper array cleanup] %s\n", error.what()));
    }
    catch(...)
    {
        static_cast<void>(std::fprintf(stderr, "[cuda_driver_wrapper array cleanup] unknown failure\n"));
    }
}


} // namespace cuda_driver_wrapper