#include <cuda_driver_wrapper/module.hpp>
#include <cuda_driver_wrapper/context.hpp>
#include <cuda_driver_wrapper/error.hpp>
#include <cuda_driver_wrapper/read_binary_file.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuda_driver_wrapper
{

Module::Module(
    const CUcontext context,
    const std::filesystem::path& path,
    const std::source_location location
) : context_(context)
{
    // context が nullptr なら拒否
    if(context_ == nullptr)
    {
        throw std::invalid_argument("Module requires a non-null context");
    }

    // ptx 等のファイルを読む
    auto image = read_binary_file(path);
    if(image.empty())
    {
        throw std::invalid_argument("CUDA module file is empty");
    }

    // PTX は NUL 終端が必要．binary の有効なイメージに続く 1 byte 分は読まれない
    // 汎用のファイル読み込み関数ではなく，CUDA の入力を準備する場所で追加する
    image.push_back('\0');

    const auto utf8 = path.u8string();
    const std::string operation = "cuModuleLoadDataEx [" + std::string(utf8.begin(), utf8.end()) + "]";

    // JIT コンパイル失敗時の説明を追加（固定長なので長いログは切り詰められる可能性あり）
    std::array<char, 8192> error_log{};
    CUjit_option options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    };
    void* values[] = {
        error_log.data(),
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(error_log.size()))
    };

    ScopedCurrentContext current(context_, location);
    CUmodule created = nullptr;
    const CUresult result = cuModuleLoadDataEx(&created, image.data(), 2, options, values);
    if(result != CUDA_SUCCESS)
    {
        const auto end = std::find(error_log.begin(), error_log.end(), '\0');
        const std::string log(error_log.begin(), end);
        const std::string diagnostic = operation + (log.empty() ? "" : "\nJIT error log (at most 8192 bytes):\n" + log);
        check_cuda(result, diagnostic.c_str(), location);
    }

    module_ = created;

    try
    {
        current.restore(location);
    }
    catch(...)
    {
        // 失敗したコンストラクタには ~Module() が実行されない
        close_noexcept();
        throw;
    }
}

Module::~Module() noexcept
{
    close_noexcept();
}

Module::Module(Module&& other) noexcept
    :   context_(std::exchange(other.context_, nullptr)),
        module_(std::exchange(other.module_, nullptr))
{
}

Module& Module::operator=(Module&& other)
{
    if(this != &other)
    {
        close();
        context_  = std::exchange(other.context_, nullptr);
        module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
}

void Module::close(const std::source_location location)
{
    if(!is_valid()) { return; }

    ScopedCurrentContext current(context_, location);
    check_cuda(cuModuleUnload(module_), "cuModuleUnload", location);
    // Unload の成功を先に反映し，restore だけが失敗しても二重に Unload しない
    module_ = nullptr;
    context_ = nullptr;
    current.restore(location);
}

void Module::close_noexcept() noexcept
{
    if(!is_valid()) { return; }

    const CUcontext context = std::exchange(context_, nullptr);
    const CUmodule module = std::exchange(module_, nullptr);

    try
    {
        ScopedCurrentContext current(context);
        static_cast<void>(report_cuda_cleanup_result(cuModuleUnload(module), "cuModuleUnload"));
        current.restore();
    }
    catch(const std::exception& error)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper module cleanup] %s\n", error.what()
        ));
    }
    catch(...)
    {
        static_cast<void>(std::fprintf(
            stderr, "[cuda_driver_wrapper module cleanup] unknown failure\n"
        ));
    }
    
}

} // namespace cuda_driver_wrapper
