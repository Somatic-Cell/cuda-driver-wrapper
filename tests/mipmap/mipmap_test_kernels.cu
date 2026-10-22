// CUDAのデバイス側宣言を使用する。ホスト側Runtime APIや<<< >>>起動は使わない。
#include <cuda_runtime.h>

// queries は [u, v, lod] の3個ずつ。出力は一問い合わせにつき4個のfloat。
// 出力の未使用成分は0。テクスチャの形式・読み出し設定に対応するエントリを選ぶ。
template<class T>
__device__ void store_result(float* destination, const T value)
{
    destination[0] = static_cast<float>(value);
    destination[1] = 0.0f; destination[2] = 0.0f; destination[3] = 0.0f;
}

template<> __device__ void store_result(float* d, const float2 v)
{
    d[0] = v.x; d[1] = v.y; d[2] = 0.0f; d[3] = 0.0f;
}
template<> __device__ void store_result(float* d, const float4 v)
{
    d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
}
template<> __device__ void store_result(float* d, const uchar2 v)
{
    d[0] = static_cast<float>(v.x); d[1] = static_cast<float>(v.y); d[2] = 0.0f; d[3] = 0.0f;
}
template<> __device__ void store_result(float* d, const uchar4 v)
{
    d[0] = static_cast<float>(v.x); d[1] = static_cast<float>(v.y);
    d[2] = static_cast<float>(v.z); d[3] = static_cast<float>(v.w);
}

template<class T>
__device__ void sample_mip(
    cudaTextureObject_t texture, const float* queries, float* result, unsigned int count
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) { return; }
    const T value = tex2DLod<T>(texture, queries[3 * i], queries[3 * i + 1], queries[3 * i + 2]);
    store_result(result + 4 * i, value);
}

extern "C" __global__ void cdw_mip_f1(cudaTextureObject_t t, const float* q, float* r, unsigned int n)
{ sample_mip<float>(t, q, r, n); }
extern "C" __global__ void cdw_mip_f2(cudaTextureObject_t t, const float* q, float* r, unsigned int n)
{ sample_mip<float2>(t, q, r, n); }
extern "C" __global__ void cdw_mip_f4(cudaTextureObject_t t, const float* q, float* r, unsigned int n)
{ sample_mip<float4>(t, q, r, n); }
extern "C" __global__ void cdw_mip_u1(cudaTextureObject_t t, const float* q, float* r, unsigned int n)
{ sample_mip<unsigned char>(t, q, r, n); }
extern "C" __global__ void cdw_mip_u2(cudaTextureObject_t t, const float* q, float* r, unsigned int n)
{ sample_mip<uchar2>(t, q, r, n); }
extern "C" __global__ void cdw_mip_u4(cudaTextureObject_t t, const float* q, float* r, unsigned int n)
{ sample_mip<uchar4>(t, q, r, n); }
