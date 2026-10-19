#pragma once

#include <cuda_driver_wrapper/device_memory.hpp>

#include <cstddef>
#include <source_location>

namespace cuda_driver_wrapper
{

// cuMemAlloc で得た一つの linear allocation を所有する
// context は借用する
// 未初期化のバイト列

class DeviceBuffer final 
{
public:
    DeviceBuffer() noexcept = default;

    // nullptr context はサイズに関係なく拒否
    // 0 バイトだった場合は空のバッファとして扱い，cuMemAlloc を呼ばない
    explicit DeviceBuffer(
        CUcontext context,
        std::size_t size_bytes,
        std::source_location location = std::source_location::current()
    );

    ~DeviceBuffer() noexcept;

    // コピーの禁止
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // ムーブ
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer && other);

    // 明示的な開放は失敗を例外で報告
    // 成功時は未確保状態
    void close(std::source_location location = std::source_location::current());

    [[nodiscard]] CUcontext context() const noexcept {return context_; }
    [[nodiscard]] CUdeviceptr device_ptr() const noexcept {return data_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept {return size_bytes_; }
    [[nodiscard]] bool empty() const noexcept {return size_bytes_ == 0; }

    
    // const の所有者からは読み取り用の view を返す
    [[nodiscard]] DeviceMemoryView view() & noexcept
    {
        return {context_, data_, size_bytes_};
    }

    [[nodiscard]] ConstDeviceMemoryView view() const & noexcept
    {
        return {context_, data_, size_bytes_};
    }

    // 一次的な DeviceBuffer から view を取り出す操作は禁止
    DeviceMemoryView view() && = delete;
    ConstDeviceMemoryView view() const && = delete;

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;   // 所属しているコンテキストの借用ハンドル
    CUdeviceptr data_ = 0;          // デバイスのアドレス
    std::size_t size_bytes_ = 0;    // 確保したバイト数
};


} // namespace cuda_driver_wrapper