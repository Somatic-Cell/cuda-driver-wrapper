#include <cuda_driver_wrapper/read_binary_file.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
namespace cdw = cuda_driver_wrapper;

void require(bool condition, std::string_view message)
{
    if(!condition) { throw std::runtime_error(std::string(message)); }
}

template<class Exception, class Function>
void require_throws(Function&& function, std::string_view message)
{
    bool caught = false;
    try { function(); }
    catch(const Exception&) { caught = true; }
    require(caught, message);
}

// ファイルの内容を試すテストだけが所有する一時ディレクトリ。
class TempDirectory final
{
public:
    TempDirectory()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for(int attempt = 0; attempt < 100; ++attempt)
        {
            path_ = std::filesystem::temp_directory_path() /
                ("cdw_m0_6_" + std::to_string(stamp) + "_" + std::to_string(attempt));
            if(std::filesystem::create_directory(path_)) { return; }
        }
        throw std::runtime_error("Failed to create a unique temporary directory");
    }
    ~TempDirectory() noexcept
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    const std::filesystem::path& path() const noexcept { return path_; }
private:
    std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, const std::vector<char>& bytes)
{
    std::ofstream output(path, std::ios::binary);
    require(bool(output), "failed to open a fixture for writing");
    if(!bytes.empty())
    {
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    output.close();
    require(bool(output), "failed to write a fixture");
}

} // namespace

int main()
{
    try
    {
        TempDirectory directory;
        const std::vector<char> expected{'A', '\0', 'B', '\r', '\n', static_cast<char>(0xff)};
        const auto binary = directory.path() / "bytes.bin";
        write_file(binary, expected);
        require(cdw::read_binary_file(binary) == expected, "all binary bytes must be preserved");

        const auto empty = directory.path() / "empty.bin";
        write_file(empty, {});
        require(cdw::read_binary_file(empty).empty(), "an empty file must be accepted by the reader");

        const auto unicode = directory.path() / std::filesystem::path(u8"読み込み確認.bin");
        write_file(unicode, expected);
        require(cdw::read_binary_file(unicode) == expected, "native filesystem paths must work");

        require_throws<std::runtime_error>([&] {
            static_cast<void>(cdw::read_binary_file(directory.path() / "missing.bin"));
        }, "a missing file must fail");
        require_throws<std::invalid_argument>([] {
            static_cast<void>(cdw::read_binary_file({}));
        }, "an empty path must fail");

        auto native = binary.native();
        native.push_back(std::filesystem::path::value_type{});
        native += std::filesystem::path("hidden_suffix").native();
        require_throws<std::invalid_argument>([&] {
            static_cast<void>(cdw::read_binary_file(std::filesystem::path(native)));
        }, "an embedded NUL must not silently truncate a path");

        std::cout << "All binary file tests passed.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
