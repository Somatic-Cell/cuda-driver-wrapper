// ------------------------------------
// CUDA Stream: GPU に投入する処理の実行順序を決める処理列
// 同じ stream に投入した処理は，投入順に実行される
// 別の stream との間で順序付けをする際は，event 等で依存関係を指定する
// ------------------------------------
#pragma once

#include <cuda.h>
#include <source_location>

namespace cuda_driver_wrapper
{

// CUstream を一つ所有し，所属する CUcontext は借用する
// conext は，この Stream の使用・破棄が終わるまで有効に保つこと
// context の retain / 破棄時の GPU 処理の完了待ちは自動では行わない
// 同じオブジェクトの使用と close / move / 破棄は，呼び出し側で直列化する
class Stream final
{
public:
    explicit Stream(
        CUcontext context,
        unsigned int flags = CU_STREAM_NON_BLOCKING,    // legacy default stream との暗黙の同期を行わない
        std::source_location location = std::source_location::current()
    );

    ~Stream() noexcept;

    // コピーの禁止
    Stream(const Stream&) = delete;             // コピーコンストラクタの禁止
    Stream& operator=(const Stream&) = delete;  // コピー代入の禁止

    // ムーブ
    // このクラスは CUDA のストリームを作り直したり，
    // 別の GPU はコンテキストに移動したりしないため，
    // ムーブを許す
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other);      // 古い stream の close が失敗する可能性を考え，ムーブ代入は noexcept にしない

    // 破棄を要求するが，ストリームに投入済みの処理に対して完了待ちはしない
    // 成功すると空になり，もともと空のオブジェクトで呼んでも何もしない
    void close(std::source_location location = std::source_location::current());

    // 借用ハンドル．呼び出し側で destroy しない．API の current の設定は呼び出し側の責任
    [[nodiscard]] CUstream handle() const noexcept {return stream_; }
    [[nodiscard]] CUcontext context() const noexcept {return context_; }
    [[nodiscard]] bool is_valid() const noexcept {return stream_ != nullptr; }

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;
    CUstream stream_ = nullptr;
};

// ホスト側でこのストリームに投入した処理の完了を待つ
void synchronize_stream(
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// CUDA_SUCCESS なら true, CUDA_ERROR_NOT_READY なら false
// その他の CUA Error はCudaError として送出
// close / move 後の空の stream は拒否
[[nodiscard]] bool is_stream_ready(
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

} // namespace cuda_driver_pointer