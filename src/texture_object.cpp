#include <cuda_driver_wrapper/texture_object.hpp>
#include <cuda_driver_wrapper/mipmapped_array_2d.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace cuda_driver_wrapper
{

namespace
{

void validate_address_mode(const CUaddress_mode mode)
{
    switch(mode)
    {
        case CU_TR_ADDRESS_MODE_WRAP:
        case CU_TR_ADDRESS_MODE_CLAMP:
        case CU_TR_ADDRESS_MODE_MIRROR:
        case CU_TR_ADDRESS_MODE_BORDER:
            return;
        default:
            throw std::invalid_argument("Unsupported 2D texture address mode");
    }
}

void validate_sampling(
    const Array2DDescriptor& array,
    const Texture2DDescriptor& descriptor
)
{
    if(array.width == 0 || array.height == 0)
    {
        throw std::invalid_argument("TextureObject requires a non-zero 2D array extent");
    }
    static_cast<void>(texel_size_bytes(array.format, array.num_channels));
    validate_address_mode(descriptor.address_mode_u);
    validate_address_mode(descriptor.address_mode_v);
    if(descriptor.filter_mode != CU_TR_FILTER_MODE_POINT && 
    descriptor.filter_mode != CU_TR_FILTER_MODE_LINEAR)
    {
        throw std::invalid_argument("2D texture filtering must be point or linear");
    }


    // ディスクリプタの組み合わせが有効かどうかを確認
    // 非正規化テクスチャとして使う場合，クランプを許さない
    if(
        !descriptor.normalized_coordinates &&
        (   descriptor.address_mode_u != CU_TR_ADDRESS_MODE_CLAMP ||
            descriptor.address_mode_v != CU_TR_ADDRESS_MODE_CLAMP )
    )
    {
        throw std::invalid_argument("This API requires clamp addressing for unnormalized coordinates");
    }

    // テクスチャを整数として読みだす場合
    if(descriptor.read_as_integer)
    {
        // UINT8 以外の整数読み出しを許さない
        if(array.format != CU_AD_FORMAT_UNSIGNED_INT8)
        {
            throw std::invalid_argument("Integer reads require a UINT8 array");
        }

        // 補間を許さない
        if(descriptor.filter_mode != CU_TR_FILTER_MODE_POINT)
        {
            throw std::invalid_argument("Integer texture reads require point filtering");
        }
    }

}

CUDA_TEXTURE_DESC make_sampling_descriptor(const Texture2DDescriptor& descriptor)
{
    CUDA_TEXTURE_DESC sampling{};
    sampling.addressMode[0] = descriptor.address_mode_u;
    sampling.addressMode[1] = descriptor.address_mode_v;
    sampling.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    sampling.filterMode = descriptor.filter_mode;

    if(descriptor.normalized_coordinates)
    {
        sampling.flags |= CU_TRSF_NORMALIZED_COORDINATES;
    }
    if(descriptor.read_as_integer)
    {
        sampling.flags |= CU_TRSF_READ_AS_INTEGER;
    }
    sampling.maxAnisotropy = 1;
    sampling.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    
    return sampling;
}

} // namespace 


TextureObject::TextureObject(
    const ConstArray2DView source,
    const Texture2DDescriptor descriptor,
    const std::source_location location
) : context_(source.context)
{
    if(source.context == nullptr || source.array == nullptr)
    {
        throw std::invalid_argument("TextureObject requires a non-null context and array");
    }

    validate_sampling(source.descriptor, descriptor);
    CUDA_RESOURCE_DESC resource{};

    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = source.array;
    
    create(resource, make_sampling_descriptor(descriptor), location);
}

TextureObject::TextureObject(
    const ConstMipmappedArray2DView source,
    const MipmappedTexture2DDescriptor descriptor,
    const std::source_location location
) : context_(source.context)
{
    if(source.context == nullptr || source.array == nullptr)
    {
        throw std::invalid_argument("TextureObject requires a non-null context and mipmapped array");
    }

    validate_sampling(source.base_descriptor, descriptor.sampling);
    const auto maximum = full_mip_level_count(
        source.base_descriptor.width, source.base_descriptor.height
    );

    if(source.level_count == 0 || source.level_count > maximum)
    {
        throw std::invalid_argument("Invalid mip level count in texture source view");
    }

    if(!descriptor.sampling.normalized_coordinates)
    {
        throw std::invalid_argument("Mipmapped textures require normalized contexts");
    }

    if(descriptor.mipmap_filter_mode != CU_TR_FILTER_MODE_POINT && 
    descriptor.mipmap_filter_mode != CU_TR_FILTER_MODE_LINEAR)
    {
        throw std::invalid_argument("Mipmap flitering must be point or linear");
    }

    if(descriptor.sampling.read_as_integer && descriptor.mipmap_filter_mode != CU_TR_FILTER_MODE_POINT)
    {
        throw std::invalid_argument("Integer mipmapped texture reads require  point mipmap filtering");
    }

    const float last_lod = static_cast<float>(source.level_count - 1);
    const float min_lod = descriptor.min_lod;
    const float max_lod = descriptor.max_lod.value_or(last_lod);

    if(!std::isfinite(min_lod) || !std::isfinite(max_lod) || min_lod < 0.0f || max_lod < min_lod || max_lod > last_lod)
    {
        throw std::invalid_argument("LOD clamps be finite and satisfy 0 <= min <= max <= last level");
    }
    
    CUDA_RESOURCE_DESC resource{};

    resource.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resource.res.mipmap.hMipmappedArray = source.array;
    auto sampling = make_sampling_descriptor(descriptor.sampling);
    
    sampling.mipmapFilterMode = descriptor.mipmap_filter_mode;
    sampling.minMipmapLevelClamp = min_lod;
    sampling.maxMipmapLevelClamp = max_lod;
    if(descriptor.disable_trilinear_optimization)
    {
        sampling.flags |= CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION;
    }

    create(resource, sampling, location);
}

void TextureObject::create(
    const CUDA_RESOURCE_DESC& resource,
    const CUDA_TEXTURE_DESC& sampling,
    const std::source_location location
)
{
    ScopedCurrentContext current(context_, location);
    CUtexObject created = 0;

    check_cuda(
        cuTexObjectCreate(&created, &resource, &sampling, nullptr),
        "cuTexOjbectCreate", location
    );
    texture_ = created;

    try
    {
        current.restore(location);
    }
    catch(...)
    {
        // コンストラクタ失敗時には自分のデストラクタが呼ばれない
        close_noexcept();
        throw;
    }
}

TextureObject::~TextureObject() noexcept
{
    close_noexcept();
}

// move
TextureObject::TextureObject(TextureObject&& other) noexcept
    :   context_(std::exchange(other.context_, nullptr)),
        texture_(std::exchange(other.texture_, 0))
{
}

TextureObject& TextureObject::operator=(TextureObject&& other)
{
    if(this != &other)
    {
        close();
        context_ = std::exchange(other.context_, nullptr);
        texture_ = std::exchange(other.texture_, 0);
    }
    return *this;
}

void TextureObject::close(const std::source_location location)
{
    if(!is_valid()) { return; }
    ScopedCurrentContext current(context_, location);
    check_cuda(
        cuTexObjectDestroy(texture_),
        "cuTexObjectDestroy", location
    );

    context_ = nullptr;
    texture_ = 0;

    current.restore(location);
    
}

void TextureObject::close_noexcept() noexcept
{
    if(!is_valid()) { return; }

    const CUcontext context = std::exchange(context_, nullptr);
    const CUtexObject texture = std::exchange(texture_, 0);

    try
    {
        ScopedCurrentContext current(context);
        static_cast<void>(report_cuda_cleanup_result(
            cuTexObjectDestroy(texture), 
            "cuTexOjbectDestroy"
        ));
        current.restore();
    }
    catch(const std::exception& error)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper texture cleanup] %s\n", error.what()
        ));
    }
    catch(...)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper texture cleanup] unknown failure\n"
        ));
    }
}


} // namespace cuda_driver_wrapper