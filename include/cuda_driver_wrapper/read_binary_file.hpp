#pragma once

#include <filesystem>
#include <vector>

namespace cuda_driver_wrapper
{

// ファイルを加工せずに読み込む
// 空のファイルに対しては空の vector を返す
// CUDA の初期化，module のロード，末尾への '\0' の追加は行わない
[[nodiscard]] std::vector<char> read_binary_file(
    const std::filesystem::path & path
);

} // namespace cuda_driver_wrapper