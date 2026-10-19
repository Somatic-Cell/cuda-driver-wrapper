// ----------------------------
// 利用する CUDA デバイスを管理する
// ----------------------------
#pragma once

#include <cuda.h>

#include <cstddef>
#include <source_location>
#include <string>

namespace cuda_driver_wrapper
{

// クエリした時点でのデバイスの情報を保有する構造体
struct DeviceInfo
{
    std::string name;
    std::size_t total_memory_bytes = 0; // 総メモリ量 (空き容量ではない)
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multiprocessor_count = 0;
    int max_threads_per_block = 0;
};


// cuInit(0) を呼ぶ関数．失敗は CudaError として報告．
// このラッパー自身は初期化済みのフラグを保持しない
// コンテキストの作成，保持，current の変更は行わない
void initialize_driver(
    std::source_location location = std::source_location::current()
);

// ----------------------
//　このプロセスで CUDA driver の初期化が成功していることが前提の関数
// ----------------------

// このプロセスから利用可能なデバイスの個数を返す．成功時の 0 はそのまま返す
[[nodiscard]]
int get_device_count(
    std::source_location location = std::source_location::current()
);

// GPU の番号を指定して CUDA デバイスを取得
[[nodiscard]]
CUdevice get_device(
    int ordinal, // 利用したい GPU の番号. [0, get_device_count()) 
    std::source_location location = std::source_location::current()
);

// Device attribute (デバイス属性): GPU の Compute Capability 等を取得する関数
[[nodiscard]]
int get_device_attribute(
    CUdevice device,
    CUdevice_attribute attribute,
    std::source_location location = std::source_location::current()
);

// 名前，総メモリ量，基本的な計算能力をまとめて問い合わせる関数
// クエリが一部失敗した場合，不完全な DeviceInfo は返さない
[[nodiscard]]
DeviceInfo get_device_info(
    CUdevice device,
    std::source_location location = std::source_location::current()
);

} // namespace cuda_driver_wrapper