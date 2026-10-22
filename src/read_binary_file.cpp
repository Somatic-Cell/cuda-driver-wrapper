#include <cuda_driver_wrapper/read_binary_file.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace cuda_driver_wrapper
{
std::vector<char> read_binary_file(const std::filesystem::path& path)
{
    if(
        path.empty() 
        ||  path.native().find(std::filesystem::path::value_type{}) 
            != std::filesystem::path::string_type::npos
    )
    {
        throw::std::invalid_argument("File path must be non-empty and contain no NUL");
    }

    // 開く操作には，native の path を直接渡す
    // UTF-8 への変換はエラーを投げるためだけに使う
    const auto utf8 = path.u8string();
    const std::string name(utf8.begin(), utf8.end());
    std::ifstream input(path, std::ios::binary | std::ios::ate); // バイナリで開き，末尾に移動 (at end)
    if(!input) // ファイルが開けなった
    {
        throw std::runtime_error("Failed to open file: " + name);
    }

    const std::streampos end = input.tellg(); // 現在位置を取得
    if(end == std::streampos(-1))
    {
        throw std::runtime_error("Failed to determine file size: " + name);
    }
    const auto length = static_cast<std::streamoff>(end);
    if(length < 0)
    {
        throw std::runtime_error("Negative file size: " + name);
    }

    const auto size = static_cast<std::uintmax_t>(length);
    if( size > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
        size > static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw std::length_error("File is too large to read into memory: " + name);
    }

    std::vector<char> bytes(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);  // ファイルの先頭 (beginning) から 0 バイトの位置にファイルポインタを移動
    if(!input)
    {
        throw std::runtime_error("Failed to seek to file begining: " + name);
    }
    if(!bytes.empty() && !input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    {
        throw std::runtime_error("Failed to read the complete file: " + name);
    }
    return bytes;
}

} // namespace cuda_driver_wrapper