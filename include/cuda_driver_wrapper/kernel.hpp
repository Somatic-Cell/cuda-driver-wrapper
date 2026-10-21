#pragma once

#include <cuda.h>

#include <source_location>
#include <span>
#include <string_view>

namespace cuda_driver_wrapper
{
    
class Module;
class Stream;


// 非所有の記述子
// コピーしても module / context の寿命は延びない
// Module::close() 後の失効は自動検出しない
// これは CUmodule から得る CUfunction であって，CUlibrary / CUkernel 用ではない
struct Kernel
{
    CUcontext context = nullptr;
    CUfunction function = nullptr;
};

struct LaunchConfig
{
    // ブロック数
    unsigned int grid_x = 1;
    unsigned int grid_y = 1;
    unsigned int grid_z = 1;

    // 各ブロック内のスレッド数
    unsigned int block_x = 1;
    unsigned int block_y = 1;
    unsigned int block_z = 1;

    // ブロックごとの動的な shared memory (Bytes)
    unsigned int shared_memory_bytes = 0;
};

// module 内に宣言された name という名前の kernel を取得する
// name は空文字と埋め込みの NUL を拒否．失敗は CUDA Error も含めて例外で報告
[[nodiscard]] Kernel get_kernel(
    const Module& module,   // kernel を取り出したいモジュール
    std::string_view name,  // モジュール内に宣言された kernel 名
    std::source_location location = std::source_location::current()
);

// 一次所有者から寿命の短い借用記述子を取り出す間違いを防ぐ
Kernel get_kernel(
    const Module&&,
    std::string_view,
    std::source_location = std::source_location::current()
) = delete;

// 指定した stream に kernel を投入する
// kernel / stream の context は一致させる． 
// zero dimension は拒否し，デバイス固有の上限や resource の制約は cuLaunchKernel にゆだねる

// kernel_arguments[i] は 第 i 番目の引数の値を置いたホスト領域のアドレス
// カーネルの定義と引数の数，順序，サイズ，ABI を一致させるのは呼び出し側の責任
// 本当に引数なしの kernel に限り，空の span を渡してよい
// 配列とホスト上の引数値は，この呼び出し中に生存させる
// 引数で参照する GPU メモリ，module, context は GPU 処理完了まで生存させる
// 例外になっても，復元段階の失敗であればすでに投入済みの場合がある
void launch_kernel(
    Kernel kernel,
    const LaunchConfig& config,
    const Stream& stream,
    std::span<void*> kernel_arguments,
    std::source_location location = std::source_location::current()
);

}