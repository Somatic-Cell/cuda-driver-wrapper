#pragma once

#include <cuda.h>

#include <cstddef>
#include <source_location>
#include <span>

namespace cuda_driver_wrapper
{

class Stream;

// ホスト側で使う，デバイスメモリの記述子
// ホスト側の処理が，GPU 上のどの範囲を使うかを伝えるための記述子
// 単位は全て Byte
// context とメモリの寿命は延長しない． data をホスト側のポインタとして間接参照してはいけない
// 外部のメモリも記述できる
// context, アドレス，範囲が実際の allocation と整合することは，呼び出し側が保証する
// この API で扱うのは同一の context 内の通常の linear デバイスメモリ
// Managed memory, peer copy, stream-ordered allocation の管理は対象外
struct ConstDeviceMemoryView
{
    CUcontext context = nullptr;    // 所属コンテキストの借用ハンドル
    CUdeviceptr data = 0;           // デバイスアドレス
    std::size_t size_bytes = 0;     // この view が表す範囲のバイト数
};

struct DeviceMemoryView
{
    CUcontext context = nullptr;    // 所属コンテキストの借用ハンドル
    CUdeviceptr data = 0;           // デバイスアドレス
    std::size_t size_bytes = 0;     // この view が表す範囲のバイト数

    // 書き込み可能な view は，読み取り元としても使える．逆方向に暗黙の変換はしない
    // CUdevicePtr 自体に const 性はないため，生の API まで保護する仕組みではない
    [[nodiscard]] operator ConstDeviceMemoryView() const noexcept
    {
        return {context, data, size_bytes};
    }
};

// count * element_size のオーバフローを検出する
// 失敗は length_error (オーバフローした場合), invalid_argument(要素サイズが 0 の場合)
[[nodiscard]] std::size_t checked_byte_size(
    std::size_t count,
    std::size_t element_size
);

// 一部分だけを渡す場合
// 元の view の範囲内だけを借用する
// メタデータだけの検証であり，実際の allocation の存在までは問合せない
[[nodiscard]] DeviceMemoryView subview(
    DeviceMemoryView memory,
    std::size_t offset_bytes,
    std::size_t count_bytes
);

[[nodiscard]] ConstDeviceMemoryView subview(
    ConstDeviceMemoryView memory,
    std::size_t offset_bytes,
    std::size_t count_bytes
);



// CPU -> GPU にコピーの完了を確認して戻る
// ホストの転送は，この段階では完了待ちを含む同期インターフェースに限定
// サイズは厳密に一致させ，一部分のコピーには subview / span::subspan を使う
void copy_host_to_device(
    DeviceMemoryView destination,
    std::span<const std::byte> source,
    std::source_location location = std::source_location::current()
);

// GPU -> CPU にコピーの完了を確認して戻る
void copy_device_to_host(
    std::span<std::byte> destination,
    ConstDeviceMemoryView source,
    std::source_location location = std::source_location::current()
);


// GPU -> GPU
// 指定した stream にコピー処理を投入．呼び出し内では完了待ちしない
// 転送元と転送先のバイト数は一致させる
// 範囲外のコピーは何もしない．それ以外の重なりは拒否する
// 両方の view と stream は同じ context
void copy_device_to_device_async(
    DeviceMemoryView destination,
    ConstDeviceMemoryView source,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// GPU メモリの各バイトを value にする
// 型付きの要素の値を代入する操作ではない
// 指定した stream に処理を投入．呼び出し内では完了待ちしない
void fill_device_bytes_async(
    DeviceMemoryView destination,
    unsigned char value,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);


} // namespace cuda_driver_wrapper