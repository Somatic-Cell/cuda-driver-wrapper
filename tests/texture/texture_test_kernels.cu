// cuda_runtime.h は tex2D / float2 / float4 等のデバイス側宣言のため。
// ホスト側 Runtime API の呼び出しや <<< >>> 起動は使用しない。
#include <cuda.h>
#include <cuda_runtime.h>

static_assert(sizeof(CUtexObject) == sizeof(cudaTextureObject_t));

// coordinates は x0,y0,x1,y1,... の float 列。
// 出力の成分数を、配列の成分数と tex2D<T> の T に正確に合わせる。
extern "C" __global__ void cdw_sample_float_1(
    cudaTextureObject_t texture, const float* coordinates,
    float* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    output[i] = tex2D<float>(texture, coordinates[2 * i], coordinates[2 * i + 1]);
}

extern "C" __global__ void cdw_sample_float_2(
    cudaTextureObject_t texture, const float* coordinates,
    float* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    const float2 value = tex2D<float2>(texture, coordinates[2 * i], coordinates[2 * i + 1]);
    output[2 * i] = value.x;
    output[2 * i + 1] = value.y;
}

extern "C" __global__ void cdw_sample_float_4(
    cudaTextureObject_t texture, const float* coordinates,
    float* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    const float4 value = tex2D<float4>(texture, coordinates[2 * i], coordinates[2 * i + 1]);
    output[4 * i] = value.x;
    output[4 * i + 1] = value.y;
    output[4 * i + 2] = value.z;
    output[4 * i + 3] = value.w;
}

extern "C" __global__ void cdw_sample_u8_1(
    cudaTextureObject_t texture, const float* coordinates,
    unsigned int* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    output[i] = tex2D<unsigned char>(texture, coordinates[2 * i], coordinates[2 * i + 1]);
}

extern "C" __global__ void cdw_sample_u8_2(
    cudaTextureObject_t texture, const float* coordinates,
    unsigned int* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    const uchar2 value = tex2D<uchar2>(texture, coordinates[2 * i], coordinates[2 * i + 1]);
    output[2 * i] = value.x;
    output[2 * i + 1] = value.y;
}

extern "C" __global__ void cdw_sample_u8_4(
    cudaTextureObject_t texture, const float* coordinates,
    unsigned int* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    const uchar4 value = tex2D<uchar4>(texture, coordinates[2 * i], coordinates[2 * i + 1]);
    output[4 * i] = value.x;
    output[4 * i + 1] = value.y;
    output[4 * i + 2] = value.z;
    output[4 * i + 3] = value.w;
}

// 材質ごとに異なる texture handle を選ぶ状況の最小テスト。
extern "C" __global__ void cdw_sample_texture_table(
    const cudaTextureObject_t* textures, const unsigned int* indices,
    const float* coordinates, float* output, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    output[i] = tex2D<float>(textures[indices[i]], coordinates[2 * i], coordinates[2 * i + 1]);
}
