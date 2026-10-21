#pragma once 

#include <cuda.h>

#include <filesystem>
#include <source_location>

namespace cuda_driver_wrapper
{

// cuModuleLoadDataEx で取得した CUmodule を一つ所有し，context は借用する
// Driver の初期化済で，有効な context を渡す
// CUDA の ptx /cubin / fatbin が対象．OptiX-IR は対象外
// close / move / 破棄と，同じ module の使用は呼び出し側で直列化
class Module final
{
public:
    Module() noexcept = default;
    Module(
        CUcontext context,
        const std::filesystem::path& path,
        std::source_location location = std::source_location::current()
    );

    ~Module() noexcept;

    // コピーの禁止
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // move は OK
    Module(Module&& other) noexcept;
    Module& operator=(Module&& other);

    // この module を使う全処理の完了を先に保証しておくこと
    // 独自の同期は追加しないが，Driver 自身が待機しないとは保証しない
    // 成功後，借用していた CUfunction / module 内の global 等は使用不可
    // 空なら何もせず，明示的な close の失敗は例外で報告
    void close(std::source_location location = std::source_location::current());

    [[nodiscard]] CUmodule handle() const noexcept { return module_; }
    [[nodiscard]] CUcontext context() const noexcept {return context_; }
    [[nodiscard]] bool is_valid() const noexcept { return module_ != nullptr; }

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;   // 借用
    CUmodule module_ = nullptr;     // 所有
    
};


} // namespace cuda_driver_wrapper