#pragma once 

#include <cuda_driver_wrapper/device_memory.hpp>

#include <cstddef>
#include <source_location>

namespace cuda_driver_wrapper
{

class Stream;

// ホスト側の pinned memory を借用する．単位は byte
// 通常の pageable memory を入れても pinned memory にはならない
// data の範囲全体が pinned であり，処理完了まで有効であることは呼び出し側が保証する
// メタデータの検査だけを行う
struct ConstPinnedHostMemoryView
{
    CUcontext context = nullptr;
    const std::byte* data = nullptr;    // 先頭アドレス
    std::size_t size_bytes = 0;         // データのバイト数
};

struct PinnedHostMemoryView
{
    CUcontext context = nullptr;  
    std::byte* data = nullptr;      // 先頭アドレス
    std::size_t size_bytes = 0;     // データのバイト数

    [[nodiscard]] operator ConstPinnedHostMemoryView() const noexcept
    {
        return {context, data, size_bytes};
    }
};

// 元の範囲内だけを借用する．確保，コピー，ページの固定や同期は行わない
[[nodiscard]] PinnedHostMemoryView subview(
    PinnedHostMemoryView memory,
    std::size_t offset_bytes,
    std::size_t count_bytes
);

[[nodiscard]] ConstPinnedHostMemoryView subview(
    ConstPinnedHostMemoryView memory,
    std::size_t offset_bytes,
    std::size_t count_bytes
);

// CPU -> GPU
// サイズは厳密に一致させる．部分転送には subview を使う
// source の内容変更，開放，および destination の競合アクセス，開放は
// この転送の完了確認より前に行う
void copy_host_to_device_async(
    DeviceMemoryView destination,       // 転送先 (GPU)
    ConstPinnedHostMemoryView source,   // 転送元 (CPU)
    const Stream& stream,               // 処理を投入するストリーム
    std::source_location location = std::source_location::current()
);

// GPU -> CPU
void copy_device_to_host_async(
    PinnedHostMemoryView destination,   // 転送先 (CPU)
    ConstDeviceMemoryView source,       // 転送元 (GPU)
    const Stream& stream,               // 処理を投入するストリーム
    std::source_location location = std::source_location::current()
);

} // namespace cuda_driver_wrapper