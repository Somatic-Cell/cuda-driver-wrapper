// ------------------------------------
// CUDA ドライバが管理するコンテキストを，C++ から安全に利用するためのラッパ
// コンテキスト: GPU を利用するための実行環境と，その資源の寿命を管理する単位
// CPU でいうプロセスと似た概念
// ------------------------------------

#pragma once

#include <cuda.h>

#include <source_location>
#include <thread>

namespace cuda_driver_wrapper
{

// 現在のコンテキストを取得
[[nodiscard]]
CUcontext get_current_context(
    std::source_location location = std::source_location::current()
);


// Primary context への参照を保持するクラス
// 指定したデバイスの primary context に対する retain を行い，その参照を1つ保持する
// context 全体を独占所有するわけではない
class RetainPrimaryContext final
{
public:
    explicit RetainPrimaryContext(
        CUdevice device,
        std::source_location location = std::source_location::current()
    );

    // release の失敗は報告するが，例外は送出しない
    ~RetainPrimaryContext() noexcept;

    // コピーの禁止
    RetainPrimaryContext(const RetainPrimaryContext&) = delete;                 // コピーコンストラクタの禁止
    RetainPrimaryContext& operator=(const RetainPrimaryContext&) = delete;      // コピー代入の禁止
    
    // ムーブの禁止
    // 禁止しなければならない技術的な理由はないが，
    // コンテキストの所有者を外側のスコープに固定して，その内側で
    // ストリームやメモリを管理したいため
    RetainPrimaryContext(RetainPrimaryContext&&) = delete;                      // ムーブコンストラクタの禁止
    RetainPrimaryContext& operator=(RetainPrimaryContext&&) = delete;           // ムーブ代入の禁止

    [[nodiscard]]
    CUdevice device() const noexcept {return device_;}

    // この関数を実行しただけでは current にならない
    [[nodiscard]]
    CUcontext handle() const noexcept {return context_;}

private:
    CUdevice device_;
    CUcontext context_ = nullptr;
};

// コンテキストを現在の CPU スレッドから操作できる状態に一時的に設定するクラス
// 有効な CUcontext を借用し，構築時に current context のスタックへ一度 push, 復元時に一度 pop する
// push, pop の間で例外が発生すると，pop が実行されない可能性があるため，RAII を利用する
// 操作終了後に元の状態に戻す
class [[nodiscard]] ScopedCurrentContext final
{
public:
    // nullptr は std::invalid_argument として拒否
    explicit ScopedCurrentContext(
        CUcontext context,
        std::source_location location = std::source_location::current()
    );

    ~ScopedCurrentContext() noexcept;

    // コピーの禁止
    ScopedCurrentContext(const ScopedCurrentContext&) = delete;
    ScopedCurrentContext& operator=(const ScopedCurrentContext&) = delete;

    // ムーブの禁止
    // ホストのスレッドで push したものを，対応する順序で pop する必要があるため
    ScopedCurrentContext(ScopedCurrentContext&&) = delete;
    ScopedCurrentContext& operator=(ScopedCurrentContext&&) = delete;

    void restore(
        std::source_location location = std::source_location::current()
    );

    [[nodiscard]]
    bool is_active() const noexcept {return active_; }

private:
    CUcontext context_;             // current context
    std::thread::id thread_id_;     // current context は CPU スレッドごとの状態なので，構築したスレッド ID を記録しておく
    bool active_ = false;
};


} // namespace cuda_driver_wrapper