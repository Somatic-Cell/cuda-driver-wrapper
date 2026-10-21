#include <cuda_driver_wrapper/kernel.hpp>
#include <cuda_driver_wrapper/module.hpp>
#include <cuda_driver_wrapper/stream.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>

#include <stdexcept>
#include <string>

namespace cuda_driver_wrapper
{

Kernel get_kernel
(
    const Module& module,           // kernel を取り出したいモジュール
    const std::string_view name,    // モジュール内に宣言された kernel (関数) 名
    std::source_location location
)
{
    if(!module.is_valid())
    {
        throw std::logic_error("Module is empty, close or moved-from");
    }
    if(name.empty() || name.find('\0') != std::string_view::npos)
    {
        throw std::invalid_argument("Kernel name must be non-empty and contain no NUL");
    }

    // string_view の末尾が NUL であるとは限らないので，string に変換
    const std::string terminated_name(name);
    const std::string operation = "cuModuleGetFunction [" + terminated_name + "]";
    ScopedCurrentContext current(module.context(), location);
    CUfunction function = nullptr;
    check_cuda(
        cuModuleGetFunction(&function, module.handle(), terminated_name.c_str()),
        operation.c_str(),
        location
    );
    current.restore(location);
    return {module.context(), function};
}

void launch_kernel(
    const Kernel kernel,
    const LaunchConfig& config,
    const Stream& stream,
    const std::span<void*> kernel_arguments,
    const std::source_location location
)
{
    // Kernel が有効かどうか確認
    if(kernel.context == nullptr || kernel.function == nullptr)
    {
        throw std::invalid_argument("Kernel must contain a context and a function");
    }
    
    // Stream が有効かどうか確認
    if(!stream.is_valid())
    {
        throw std::logic_error("Stream is closed or moved-from");
    }
    
    // context が有効かどうか確認
    if(kernel.context != stream.context())
    {
        throw std::invalid_argument("Kernel and stream must belong to the same context");
    }
    
    // グリッド・ブロックのサイズが適切かどうかを確認
    if(config.grid_x == 0 || config.grid_y == 0 || config.grid_z == 0 
    || config.block_x == 0 || config.block_y == 0 || config.block_z == 0)
    {
        throw std::invalid_argument("Grid and block dimensions must be non-zero");
    }
    
    // 引数が不正 (null) でないかどうか確認
    for(void* argument : kernel_arguments)
    {
        if(argument == nullptr)
        {
            // GPU pointer の値が 0 と，ホスト上の引数領域のアドレスが nullptr は別
            throw std::invalid_argument("Kernel argument storage must not be null");
        }
    }

    ScopedCurrentContext current(kernel.context, location);
    check_cuda(
        cuLaunchKernel(
            kernel.function,
            config.grid_x, config.grid_y, config.grid_z,
            config.block_x, config.block_y, config.block_z,
            config.shared_memory_bytes,
            stream.handle(),
            kernel_arguments.empty() ? nullptr : kernel_arguments.data(),   // 引数が必要ない場合にも対応
            nullptr
        ),
        "cuLaunchKernel",
        location
    );

    current.restore(location);
}

} // namespace cuda_driver_wrapper