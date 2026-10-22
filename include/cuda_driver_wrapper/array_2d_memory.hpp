// GPU 上の 2D メモリ (CUDA Array) へテクスチャなどを入れる，取り出す，コピーするためにつかう
// 2D テクスチャをサンプルする用途で GPU 上のメモリを用いる場合，
// 通常の Linear memory に格納するよりも，CUDA Array に格納する方が，通常はキャッシュ効率が良くなる
#pragma once

#include <cuda_driver_wrapper/pinned_host_memory.hpp>

#include <cstddef>
#include <source_location>
#include <span>

namespace cuda_driver_wrapper
{

class Stream;

// 2D Array のディスクリプタ
// UINT8 or FLOAT の 1, 2, 4 成分に限定
struct Array2DDescriptor
{
    std::size_t width = 0;  // テクセル数
    std::size_t height = 0; // テクセル数
    CUarray_format format = CU_AD_FORMAT_UNSIGNED_INT8;
    unsigned int num_channels = 1;  // チャネル数
};

// 配列内のコピー対象を指定する
// ホスト画像の上下左右の意味は考えず，行をその並びのままコピーする
struct ArrayRegion2D
{
    // 以下，単位は全てテクセル
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t width = 0;
    std::size_t height = 0;
};

// 2D Array の情報を，その所在と共に管理するためのディスクリプタ
// この構造体自体は配列を所有しない
// 読み取り専用
struct ConstArray2DView
{
    CUcontext context = nullptr;
    CUarray array = nullptr;
    Array2DDescriptor descriptor{};
};

// 読み書きに使える
struct Array2DView
{
    CUcontext context = nullptr;
    CUarray array = nullptr;
    Array2DDescriptor descriptor{};

    // 読み書き用から読み取り専用への変換は許す
    [[nodiscard]] operator ConstArray2DView() const noexcept
    {
        return {context, array, descriptor};
    }
};

// 1 テクセルが何バイトに相当するかを計算する
[[nodiscard]] std::size_t texel_size_bytes(
    CUarray_format format,      // 各チャネルの格納方式
    unsigned int num_channels   // チャネル数
);

// テクスチャの横一行分の画素データは何バイト分かを計算する
[[nodiscard]] std::size_t row_size_bytes(const Array2DDescriptor& descriptor);
// 配列全体を表す矩形情報を計算する
[[nodiscard]] ArrayRegion2D make_full_array_region(ConstArray2DView array);

// 先頭行の先頭から，最後の有効画素までに必要なバイト数を計算
// テクスチャのデータを linear メモリに置いたとき，
// 先頭から最低バイトひつようかを計算する
[[nodiscard]] std::size_t required_2d_bytes(
    std::size_t row_bytes,
    std::size_t height,
    std::size_t pitch_bytes
);

// 通常の pageable memory または pinned memory (CPU) -> 2D array (GPU)
// 正常終了時に転送完了
// default stream の完了を確認する
void copy_host_to_array(
    Array2DView destination,            // 格納先の CUDA Array の情報
    ArrayRegion2D region,               // 格納したい領域
    std::span<const std::byte> source,  // コピー元となる CPU 側のメモリ領域
    std::size_t source_pitch_bytes,     
    std::source_location location = std::source_location::current()
);

// array (GPU) -> host (CPU)
void copy_array_to_host(
    std::span<std::byte> destination,
    std::size_t destination_pitch_bytes,
    ConstArray2DView source,
    ArrayRegion2D region,
    std::source_location location = std::source_location::current()
);

// Async: 指定した stream に投入
// 完了まで入力の内容を保ち，出力を CPU で参照せず，双方の所有者を生存させる
// host (CPU) -> array (GPU)
void copy_host_to_array_async(
    Array2DView destination,
    ArrayRegion2D region,
    ConstPinnedHostMemoryView source,
    std::size_t source_pitch_bytes,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// array (GPU) -> host (CPU)
void copy_array_to_host_async(
    PinnedHostMemoryView destination,
    std::size_t destination_pitch_bytes,
    ConstArray2DView source,
    ArrayRegion2D region,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// linear memory (GPU) -> array (GPU)
void copy_device_to_array_async(
    Array2DView destinaion,
    ArrayRegion2D region,
    ConstDeviceMemoryView source,
    std::size_t source_pitch_bytes,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// array (GPU) -> linear memory (GPU)
void copy_array_to_device_async(
    DeviceMemoryView destinaion,
    std::size_t destination_pitch_bytes,
    ConstArray2DView source,
    ArrayRegion2D region,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// array (GPU) -> array (GPU)
void copy_array_to_array_async(
    Array2DView destination,
    ArrayRegion2D destination_region,
    ConstArray2DView source,
    ArrayRegion2D source_region,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

} // namespace cuda_driver_wrapper