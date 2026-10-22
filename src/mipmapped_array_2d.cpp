#include <cuda_driver_wrapper/mipmapped_array_2d.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{


unsigned int full_mip_level_count(
    const std::size_t width,
    const std::size_t height
)
{
    if(width == 0 || height == 0)
    {
        throw std::invalid_argument("MipmappedArray2D width and height must be non-zero");
    }
    std::size_t extent = (std::max)(width, height);
    unsigned int count = 1;
    while (extent > 1)
    {
        extent = (extent >> 1);
        ++count;
    }
    return count;
}

namespace{

void validate_descriptor(
    const Array2DDescriptor& descriptor, 
    const unsigned int levels   // 所有したい mip level
)
{
    // 理論上最大の mip level を計算
    const auto maximum = full_mip_level_count(descriptor.width, descriptor.height);
    static_cast<void>(texel_size_bytes(descriptor.format, descriptor.num_channels));

    // 所有しようとしている mip level が不正でないかを確認
    if(levels == 0 || levels > maximum)
    {
        throw std::invalid_argument("Mip level count must be in [1, full_mip_level_count]");
    }
}

void require_view(const ConstMipmappedArray2DView source)
{
    if(source.context == nullptr || source.array == nullptr)
    {
        throw std::invalid_argument("Mipmapped array view requires a non-null context and array");
    }
    validate_descriptor(source.base_descriptor, source.level_count);
}

} // namespace 

ConstArray2DView get_mip_level(
    const ConstMipmappedArray2DView source,
    const unsigned int level,
    const std::source_location location
)
{
    require_view(source);
    if(level >= source.level_count)
    {
        throw std::out_of_range("Mip level index is outside the allocated chain");
    }

    ScopedCurrentContext current(source.context, location);
    CUarray array = nullptr;
    check_cuda(
        cuMipmappedArrayGetLevel(&array, source.array, level),
        "cuMipmappedArrayGetLevel", location
    );

    // level の実寸を Driver から取得
    CUDA_ARRAY3D_DESCRIPTOR actual{};
    check_cuda(
        cuArray3DGetDescriptor(&actual, array),
        "cuArray3DGetDescriptor (mip level)", location
    );

    if(actual.Depth != 0 || actual.Width == 0 || actual.Height == 0)
    {
        throw std::invalid_argument("Mip level must be an ordinary non-layered 2D array");
    }

    static_cast<void>(texel_size_bytes(actual.Format, actual.NumChannels));
    const ConstArray2DView result{
        source.context, array, {actual.Width, actual.Height, actual.Format, actual.NumChannels}
    };
    current.restore(location);
    return result;

}

Array2DView get_mip_level(
    const MipmappedArray2DView source,
    const unsigned int level,
    const std::source_location location
)
{
    const auto result = get_mip_level(static_cast<ConstMipmappedArray2DView>(source), level, location);
    return {result.context, result.array, result.descriptor};
}

MipmappedArray2D::MipmappedArray2D(
    const CUcontext context,
    const Array2DDescriptor base_descriptor,
    const unsigned int level_count,
    const std::source_location location
) : context_(context), base_descriptor_(base_descriptor), level_count_(level_count)
{
    if(context_ == nullptr)
    {
        throw std::invalid_argument("MipmappedArray2D requires a non-null context");
    }
    validate_descriptor(base_descriptor_, level_count_);

    CUDA_ARRAY3D_DESCRIPTOR allocation{};
    allocation.Width = base_descriptor_.width;
    allocation.Height = base_descriptor_.height;
    allocation.Depth = 0; // 2D 
    allocation.Format = base_descriptor_.format;
    allocation.NumChannels = base_descriptor_.num_channels;

    ScopedCurrentContext current(context_, location);
    CUmipmappedArray created = nullptr;
    check_cuda(
        cuMipmappedArrayCreate(&created, &allocation, level_count_),
        "cuMipmappedArrayCreate",
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

MipmappedArray2D::~MipmappedArray2D() noexcept
{
    close_noexcept();
}

MipmappedArray2D::MipmappedArray2D(MipmappedArray2D&& other) noexcept
    :   context_(std::exchange(other.context_, nullptr)),
        array_(std::exchange(other.array_, nullptr)),
        base_descriptor_(std::exchange(other.base_descriptor_, Array2DDescriptor{})),
        level_count_(std::exchange(other.level_count_, 0))
{
}

MipmappedArray2D& MipmappedArray2D::operator=(MipmappedArray2D&& other)
{
    if(this != & other)
    {
        close();
        context_ = std::exchange(other.context_, nullptr);
        array_ = std::exchange(other.array_, nullptr);
        base_descriptor_ = std::exchange(other.base_descriptor_, Array2DDescriptor{});
        level_count_ = std::exchange(other.level_count_, 0);
    }
    return *this;
}

void MipmappedArray2D::close(const std::source_location location)
{
    if(!is_valid()) { return; }
    ScopedCurrentContext current(context_, location);
    check_cuda(
        cuMipmappedArrayDestroy(array_), 
        "cuMipmappedArrayDestroy", location
    );
    array_ = nullptr;
    context_ = nullptr;
    base_descriptor_ = {};
    level_count_ = 0;
    current.restore(location);
}

void MipmappedArray2D::close_noexcept() noexcept
{
    if(!is_valid()){ return; }
    
    const CUcontext context = std::exchange(context_, nullptr);
    const CUmipmappedArray array = std::exchange(array_, nullptr);
    base_descriptor_ = {};
    level_count_ = 0;

    try
    {
        ScopedCurrentContext current(context);
        static_cast<void>(report_cuda_cleanup_result(
            cuMipmappedArrayDestroy(array), 
            "cuMipmappedArrayDestroy"
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