// Driver API からロードするテスト用 module。
// ホスト側 CUDA Runtime API、<<<...>>>、OptiX は使用しない。
// extern "C" により、Driver で問い合わせるエントリ名を固定する。

extern "C" __global__ void cdw_affine_u32(
    unsigned int* values,
    unsigned int count,
    unsigned int multiplier,
    unsigned int increment
)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < count)
    {
        values[index] = values[index] * multiplier + increment;
    }
}

// 真に引数を持たないカーネルの起動を検証する。
extern "C" __global__ void cdw_noop()
{
}

// 1D、blockDim.x が 2 のべき乗、shared bytes >= blockDim.x*sizeof(unsigned int)。
// これらはこのテスト用カーネルの仕様であり、ラッパー全体の制限ではない。
extern "C" __global__ void cdw_block_sum_u32(
    const unsigned int* values,
    unsigned int* block_sums,
    unsigned int count
)
{
    extern __shared__ unsigned int scratch[];
    const unsigned int local = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + local;
    scratch[local] = index < count ? values[index] : 0u;
    __syncthreads();

    for(unsigned int offset = blockDim.x / 2; offset != 0; offset /= 2)
    {
        if(local < offset) { scratch[local] += scratch[local + offset]; }
        __syncthreads();
    }
    if(local == 0) { block_sums[blockIdx.x] = scratch[0]; }
}
