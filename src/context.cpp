#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <cstdio>
#include <stdexcept>

namespace cuda_driver_wrapper
{

namespace 
{

void report_context_contract_failure(const char* message)noexcept
{
    static_cast<void>(std::fprintf(
        stderr, "[cuda_driver_wrapper context cleanup] %s \n", message
    ));
};

} // namespace

CUcontext get_current_context(const std::source_location location)
{
    CUcontext context = nullptr;
    check_cuda(
        cuCtxGetCurrent(&context), 
        "cuCtxGetCurrent",
        location
    );
    return context;
}

RetainPrimaryContext::RetainPrimaryContext(
    const CUdevice device,
    const std::source_location location
) : device_(device)
{
    check_cuda(
        cuDevicePrimaryCtxRetain(&context_, device_),
        "cuDevicePrimaryCtxRetain",
        location
    );
}

RetainPrimaryContext::~RetainPrimaryContext() noexcept
{
    static_cast<void>(report_cuda_cleanup_result(
        cuDevicePrimaryCtxRelease(device_), 
        "cuDevicePrimaryCtxRelease"
    ));
}

// current context を context に変更する
ScopedCurrentContext::ScopedCurrentContext(
    const CUcontext context,
    const std::source_location location
) : context_(context), thread_id_(std::this_thread::get_id())
{
    if(context_ == nullptr)
    {
        throw std::invalid_argument("ScopedCurrentContext requires a non-null context");
    }

    check_cuda(
        cuCtxPushCurrent(context_), 
        "cuCtxPushCurrent",
        location
    );
    active_ = true;

}

// current context のスタックから一度 pop し，元の current context に戻す
void ScopedCurrentContext::restore(
    const std::source_location location
)
{
    // 非アクティブなら何もしない
    if(!active_)
    {
        return;
    }

    // この関数を実行している CPU スレッドが，このクラスを作ったスレッドと一致しているかどうかを確認
    if(std::this_thread::get_id() != thread_id_)
    {
        throw std::logic_error("ScopedCurrentContext must be restored on its creating thread");
    }

    // restore() を実行している CPU スレッドの current context が，
    // このクラスを作ったときの context と一致しているかどうかを確認
    if(get_current_context(location) != context_)
    {
        throw std::logic_error("ScopedCurrentContext must be restored in LIFO order");
    }

    // CPU スレッドの current context のスタックを一つ pop
    CUcontext popped = nullptr;
    check_cuda(
        cuCtxPopCurrent(&popped),
        "cuCtxPopCurrent",
        location
    );

    // pop したので current context はもう context_ ではない
    active_ = false;

    // pop した context と context_ は一致するはずなので，チェック
    if(popped != context_)
    {
        throw std::logic_error("cuCtxPopCurrent returned an unexpected context");
    }
}

ScopedCurrentContext::~ScopedCurrentContext() noexcept
{
    // 既に使い終わっているなら何もしない
    if(!active_)
    {
        return;
    }

    // この関数を実行している CPU スレッドが，このクラスを作ったスレッドと一致しているかどうかを確認
    if(std::this_thread::get_id() != thread_id_)
    {
        report_context_contract_failure("scope destroyed on a different thread; context was not popped");
        return;
    }

    CUcontext current = nullptr;
    if(!report_cuda_cleanup_result(
        cuCtxGetCurrent(&current),
        "cuCtxGetCurrent (Scope cleanup)"
    ))
    {
        return;
    }

    if(current != context_)
    {
        report_context_contract_failure("scope destroyed out of LIFO order; context was not popped");
        return;
    }

    CUcontext popped = nullptr;
    if(!report_cuda_cleanup_result(
        cuCtxPopCurrent(&popped),
        "cuCtxPopCurrent (Scope cleanup)"
    ))
    {
        return;
    }

    active_ = false;
    if(popped != context_)
    {
        report_context_contract_failure("cuCtxPopCurrent returned an unexpected context");
    }
}


} // namespace cuda_driver_wrapper