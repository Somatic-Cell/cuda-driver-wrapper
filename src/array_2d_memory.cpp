#include <cuda_driver_wrapper/array_2d_memory.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/stream.hpp>

#include <limits>
#include <stdexcept>

namespace cuda_driver_wrapper
{

// 1 テクセルが何バイトに相当するかを計算する
std::size_t texel_size_bytes(
    const CUarray_format format,
    const unsigned int num_channels
)
{
    // チャネル数が不正でないかどうかをチェック
    if(num_channels != 1 && num_channels != 2 && num_channels != 4)
    {
        throw std::invalid_argument("Array2D supports 1, 2, 4 channels");
    }

    switch(format)
    {
        case CU_AD_FORMAT_UNSIGNED_INT8: return num_channels;   // uint8 は 1 チャネル 1 バイト
        case CU_AD_FORMAT_FLOAT: return 4u * num_channels;      // float は 1 チャネル 4 バイト
        default: throw std::invalid_argument("This API supports only UINT8 and FLOAT array components");
    }
}

// テクスチャの横 1 行分の画素データは何バイト分かを計算する
std::size_t row_size_bytes(
    const Array2DDescriptor& descriptor
)
{
    // テクセル数の不正を検出
    if(descriptor.width == 0 || descriptor.height == 0)
    {
        throw std::invalid_argument("Array2D width and height must both be non-zero");
    }

    // 1 テクセル当たりのバイト数
    const auto texel_bytes = texel_size_bytes(descriptor.format, descriptor.num_channels);  
    // 1 行当たりのバイト数
    const auto row_bytes = checked_byte_size(descriptor.width, texel_bytes);

    return row_bytes;
}

// 転送時に触る linear メモリの範囲を求める
// テクスチャのデータを linear メモリに置いたとき，
// 先頭から最低バイト必要かを計算する
// 要するに，最後の行の，メモリアラインのための余白を除いたバイト数を計算する
std::size_t required_2d_bytes(
    const std::size_t row_bytes,    // テクスチャ一行当たりのバイト数
    const std::size_t height,       // テクスチャの高さ方向のテクセル数
    const std::size_t pitch_bytes   // ある行の先頭から次の行の先頭までの間隔 (必ずしも row_bytes と一致しない)
)
{
    // コピーする画素がない場合
    if(row_bytes == 0 || height == 0) { return 0; }
    
    // pitch_bytes = row_bytes + (メモリアラインメントのための余白) なので
    // pitch_bytes は row_bytes 以上になるはず
    if(pitch_bytes < row_bytes)
    {
        throw std::invalid_argument("Row pitch is smaller than the copied row");
    }

    // 戻り値がオーバフローしないかどうかの検査
    const auto maximum = (std::numeric_limits<std::size_t>::max)();
    if(height - 1 > (maximum - row_bytes) / pitch_bytes)
    {
        throw std::length_error("2D memory footprint overflows size_t");
    }

    return (height - 1) * pitch_bytes + row_bytes;
}

namespace
{

// 無効な CUDA array でないかどうかを検査する
void require_array(const ConstArray2DView view)
{
    if(view.context == nullptr || view.array == nullptr)
    {
        throw std::invalid_argument("Array2D view requires a valid context and array");
    }
    static_cast<void>(row_size_bytes(view.descriptor));
}

// コピーしたい領域が有効であるかどうかを確認
std::size_t validate_array_region(
    const ConstArray2DView view,    // コピー元
    const ArrayRegion2D region      // 上記のコピー元のうち，コピー対象としたい領域
)
{
    // 無効な CUDA array でないかどうかを検査する
    require_array(view);
    
    if(region.x > view.descriptor.width || region.width > view.descriptor.width - region.x ||
        region.y > view.descriptor.height || region.height > view.descriptor.height - region.y)
    {
        throw std::out_of_range("Array2D region is outside the supplied array extent");
    }

    // 一行当たりのバイト数を計算して返す
    return checked_byte_size(
        region.width, texel_size_bytes(view.descriptor.format, view.descriptor.num_channels)
    );
}

// コピーが要求するバイト数を，利用可能なバイト数が満たしているかどうかをチェックする
void require_capacity(
    const std::size_t available,    // 利用可能なバイト数 
    const std::size_t required      // （コピーが）要求するバイト数
)
{
    if(available < required)
    {
        throw std::out_of_range("Linear memory is too small for the requested 2D transfer");
    }
}

// コンテキストが，stream の所属するコンテキストと一致するかどうかを確認
void require_stream_context(
    const Stream& stream, 
    const CUcontext context
)
{
    // ストリームが空でないかどうか確認
    if(!stream.is_valid())
    { 
        throw std::logic_error("Stream is close or moved-from"); 
    }

    // コンテキストが，stream の所属するコンテキストと一致するかどうかを確認
    if(context != nullptr && context != stream.context())
    {
        throw std::invalid_argument("2D transfer resources and stream must use the same context");
    }

}

// コピー元の配列を指定する
void set_source_array(
    CUDA_MEMCPY2D& copy, 
    const ConstArray2DView view, 
    const ArrayRegion2D region
)
{
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;   // コピー元は CUDA Array
    copy.srcArray = view.array;                 // コピー元がどの CUDA Array かを指定
    
    // コピー元の開始位置
    copy.srcXInBytes = checked_byte_size(
        region.x, texel_size_bytes(view.descriptor.format, view.descriptor.num_channels)
    );
    copy.srcY = region.y;   
}

// コピー先の配列を指定する
void set_destination_array(
    CUDA_MEMCPY2D& copy, 
    const Array2DView view, 
    const ArrayRegion2D region
)
{
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;   // コピー先は CUDA Array
    copy.dstArray = view.array;                 // コピー先はどの CUDA Array かを指定
    
    // コピー先の開始位置
    copy.dstXInBytes = checked_byte_size(
        region.x, texel_size_bytes(view.descriptor.format, view.descriptor.num_channels)
    );
    copy.dstY = region.y;
}

// CUDA_MEMCPY2D で指定されている指示を元にコピーを実行し
// コピーの完了まで確認する
// host -> array または array -> host 
void copy_host_and_wait(
    const CUDA_MEMCPY2D& copy,      // コピーの指示
    const CUcontext context,        
    std::source_location location
)
{
    ScopedCurrentContext current(context, location);
    check_cuda(cuMemcpy2D(&copy), "cuMemcpy2D", location);
    check_cuda(
        cuStreamSynchronize(nullptr), // デフォルトストリームに対する同期
        "cuStreamSynchronize (2D copy completion)", location
    );
    current.restore(location);
}

// CUDA_MEMCPY2D で指定されている指示を元にコピーを指定したストリームに投入する
// コピーの完了まで待たない
// host -> array または array -> host 
void enqueue_host_copy(
    const CUDA_MEMCPY2D& copy,
    const Stream& stream,
    const std::source_location location
)
{
    ScopedCurrentContext current(stream.context(), location);
    check_cuda(cuMemcpy2DAsync(&copy, stream.handle()), "cuMemcpy2DAsync", location);
    current.restore(location);
}

// linear memory -> array 
// array -> linear memory
// array -> array
void enqueue_device_copy(
    const CUDA_MEMCPY2D& copy, 
    const Stream& stream, 
    const std::source_location location
)
{
    CUDA_MEMCPY3D transfer{};
    transfer.srcXInBytes = copy.srcXInBytes;
    transfer.srcY = copy.srcY;
    transfer.srcMemoryType = copy.srcMemoryType;
    transfer.srcDevice = copy.srcDevice;
    transfer.srcArray = copy.srcArray;
    transfer.srcPitch = copy.srcPitch;
    transfer.srcHeight = copy.Height;
    transfer.dstXInBytes = copy.dstXInBytes;
    transfer.dstY = copy.dstY;
    transfer.dstMemoryType = copy.dstMemoryType;
    transfer.dstDevice = copy.dstDevice;
    transfer.dstArray = copy.dstArray;
    transfer.dstPitch = copy.dstPitch;
    transfer.dstHeight = copy.Height;
    transfer.WidthInBytes = copy.WidthInBytes;
    transfer.Height = copy.Height;
    transfer.Depth = 1;

    ScopedCurrentContext current(stream.context(), location);
    check_cuda(
        cuMemcpy3DAsync(&transfer, stream.handle()),
        "cuMemcpy3DAsync (2D Device copy)", location
    );
    current.restore(location);
}

} // namespace

ArrayRegion2D make_full_array_region(const ConstArray2DView array)
{
    require_array(array);
    return {0, 0, array.descriptor.width, array.descriptor.height};
}

// Pageable memory または pinned memory (CPU) -> 2D array (GPU)
void copy_host_to_array(
    const Array2DView destination,          // 格納先の CUDA Array の情報
    const ArrayRegion2D region,             // 格納したい領域
    const std::span<const std::byte> source,      // コピー元となる CPU 側のメモリ領域
    const std::size_t source_pitch_bytes,   // 行当たりのバイト数
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(destination, region);
    const auto bytes = required_2d_bytes(row_bytes, region.height, source_pitch_bytes);
    
    // 
    require_capacity(source.size_bytes(), bytes);
    if(bytes == 0){ return; }

    // ディスクリプタ
    CUDA_MEMCPY2D copy{};
    
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;    // 転送元は HOST (0x01)
    copy.srcHost = source.data();               // 転送元となる配列の開始アドレス
    copy.srcPitch = source_pitch_bytes;         // 行当たりのバイト数
    
    set_destination_array(copy, destination, region);
    
    // コピーする矩形領域の大きさ
    copy.WidthInBytes = row_bytes;
    copy.Height = region.height;

    copy_host_and_wait(copy, destination.context, location);
}

// array (GPU) -> host (CPU)
void copy_array_to_host(
    const std::span<std::byte> destination,
    const std::size_t destination_pitch_bytes,
    const ConstArray2DView source,
    const ArrayRegion2D region,
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(source, region);
    const auto bytes = required_2d_bytes(row_bytes, region.height, destination_pitch_bytes);
    require_capacity(destination.size_bytes(), bytes);
    if(bytes == 0) { return; }

    CUDA_MEMCPY2D copy{};
    
    set_source_array(copy, source, region);
    
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = destination.data();
    copy.dstPitch = destination_pitch_bytes;
    
    // コピーする矩形領域の大きさ
    copy.WidthInBytes = row_bytes;
    copy.Height = region.height;
    
    copy_host_and_wait(copy, source.context, location);
}

void copy_host_to_array_async(
    const Array2DView destination,
    const ArrayRegion2D region,
    const ConstPinnedHostMemoryView source,
    const std::size_t source_pitch_bytes,
    const Stream& stream,
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(destination, region);
    static_cast<void>(subview(source, 0, source.size_bytes));
    require_stream_context(stream, destination.context);
    require_stream_context(stream, source.context);
    const auto bytes = required_2d_bytes(row_bytes, region.height, source_pitch_bytes);
    require_capacity(source.size_bytes, bytes);
    if(bytes == 0) { return; }

    CUDA_MEMCPY2D copy{};
    
    // コピー元: host
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = source.data;
    copy.srcPitch = source_pitch_bytes;
    
    // コピー先: CUDA array
    set_destination_array(copy, destination, region);
    
    // コピーする矩形領域の大きさ
    copy.WidthInBytes = row_bytes;
    copy.Height = region.height;
    
    enqueue_host_copy(copy, stream, location);
}

void copy_array_to_host_async(
    const PinnedHostMemoryView destination,
    const std::size_t destination_pitch_bytes,
    const ConstArray2DView source,
    const ArrayRegion2D region,
    const Stream& stream,
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(source, region);
    static_cast<void>(subview(destination, 0, destination.size_bytes));
    require_stream_context(stream, source.context);
    require_stream_context(stream, destination.context);
    const auto bytes = required_2d_bytes(row_bytes, region.height, destination_pitch_bytes);
    require_capacity(destination.size_bytes, bytes);
    if(bytes == 0){ return; }

    CUDA_MEMCPY2D copy{};
    
    // コピー元: CUDA array
    set_source_array(copy, source, region);

    // コピー先: pinned host memory
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = destination.data;
    copy.dstPitch = destination_pitch_bytes;
    
    // コピーする矩形領域の大きさ
    copy.WidthInBytes = row_bytes;
    copy.Height = region.height;

    enqueue_host_copy(copy, stream, location);
}

void copy_device_to_array_async(
    const Array2DView destination,
    const ArrayRegion2D region,
    const ConstDeviceMemoryView source,
    const std::size_t source_pitch_bytes,
    const Stream& stream,
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(destination, region);
    static_cast<void>(subview(source, 0, source.size_bytes));
    require_stream_context(stream, destination.context);
    require_stream_context(stream, source.context);
    const auto bytes = required_2d_bytes(row_bytes, region.height, source_pitch_bytes);
    require_capacity(source.size_bytes, bytes);
    if(bytes == 0){ return; }

    CUDA_MEMCPY2D copy{};

    // コピー元: linear memory
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = source.data;
    copy.srcPitch = source_pitch_bytes;
    
    // コピー先: CUDA array
    set_destination_array(copy, destination, region);
    
    // コピーする矩形領域の大きさ
    copy.WidthInBytes = row_bytes;
    copy.Height = region.height;
    
    enqueue_device_copy(copy, stream, location);
}

void copy_array_to_device_async(
    const DeviceMemoryView destination,
    const std::size_t destination_pitch_bytes,
    const ConstArray2DView source,
    const ArrayRegion2D region,
    const Stream& stream,
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(source, region);
    static_cast<void>(subview(destination, 0, destination.size_bytes));
    require_stream_context(stream, source.context);
    require_stream_context(stream, destination.context);
    const auto bytes = required_2d_bytes(row_bytes, region.height, destination_pitch_bytes);
    require_capacity(destination.size_bytes, bytes);
    if(bytes == 0){ return; }

    CUDA_MEMCPY2D copy{};
    
    set_source_array(copy, source, region);
    
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = destination.data;
    copy.dstPitch = destination_pitch_bytes;
    
    copy.WidthInBytes = row_bytes;
    copy.Height = region.height;
    
    enqueue_device_copy(copy, stream, location);
}

void copy_array_to_array_async(
    const Array2DView destination,
    const ArrayRegion2D destination_region,
    const ConstArray2DView source,
    const ArrayRegion2D source_region,
    const Stream& stream,
    const std::source_location location
)
{
    const auto row_bytes = validate_array_region(destination, destination_region);
    static_cast<void>(validate_array_region(source, source_region));
    require_stream_context(stream, destination.context);
    require_stream_context(stream, source.context);

    if(
        destination.descriptor.format != source.descriptor.format ||
        destination.descriptor.num_channels != source.descriptor.num_channels ||
        destination_region.width != source_region.width ||
        destination_region.height != source_region.height
    )
    {
        throw std::invalid_argument("Array copies require matching formats and region extents");
    }

    if(row_bytes == 0 || source_region.height == 0){ return; }

    if(destination.array == source.array)
    {
        if(destination_region.x == source_region.x && destination_region.y == source_region.y)
        {
            return;
        }
        
        const bool overlap_x =  destination_region.x < source_region.x + source_region.width &&
                                source_region.x < destination_region.x + destination_region.width;
        
        const bool overlap_y =  destination_region.y < source_region.y + source_region.height &&
                                source_region.y < destination_region.y + destination_region.height;
        
                                if(overlap_x && overlap_y) 
        {
            throw std::invalid_argument("Array copy region overlap");
        }
    }
    CUDA_MEMCPY2D copy{};
    set_source_array(copy, source, source_region);
    set_destination_array(copy, destination, destination_region);
    copy.WidthInBytes = row_bytes;
    copy.Height = source_region.height;
    enqueue_device_copy(copy, stream, location);
}

} // namespace cuda_driver_wrapper