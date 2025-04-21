#include <cfloat>
#include <cmath>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vectorTypes.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp
{
namespace cuda
{
template <typename T> __global__ void test_typecast_kernel(char *aDataOut, T *aDataIn)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 || y > 0)
    {
        return;
    }

    Vector4<int> *i1   = reinterpret_cast<Vector4<int> *>(aDataOut + 0);
    Vector4<int> *i2   = reinterpret_cast<Vector4<int> *>(aDataOut + 16);
    Vector4<uint> *ui1 = reinterpret_cast<Vector4<uint> *>(aDataOut + 32);
    Vector4<uint> *ui2 = reinterpret_cast<Vector4<uint> *>(aDataOut + 48);

    Vector4<short> *s1   = reinterpret_cast<Vector4<short> *>(aDataOut + 64);
    Vector4<short> *s2   = reinterpret_cast<Vector4<short> *>(aDataOut + 64 + 8);
    Vector4<ushort> *us1 = reinterpret_cast<Vector4<ushort> *>(aDataOut + 64 + 16);
    Vector4<ushort> *us2 = reinterpret_cast<Vector4<ushort> *>(aDataOut + 64 + 24);

    Vector4<sbyte> *sb1 = reinterpret_cast<Vector4<sbyte> *>(aDataOut + 96);
    Vector4<sbyte> *sb2 = reinterpret_cast<Vector4<sbyte> *>(aDataOut + 96 + 4);
    Vector4<byte> *b1   = reinterpret_cast<Vector4<byte> *>(aDataOut + 96 + 8);
    Vector4<byte> *b2   = reinterpret_cast<Vector4<byte> *>(aDataOut + 96 + 12);

    Vector4<BFloat16> *bf1 = reinterpret_cast<Vector4<BFloat16> *>(aDataOut + 112);
    Vector4<BFloat16> *bf2 = reinterpret_cast<Vector4<BFloat16> *>(aDataOut + 112 + 8);
    Vector4<HalfFp16> *hf1 = reinterpret_cast<Vector4<HalfFp16> *>(aDataOut + 112 + 16);
    Vector4<HalfFp16> *hf2 = reinterpret_cast<Vector4<HalfFp16> *>(aDataOut + 112 + 24);

    Vector4<float> *f1 = reinterpret_cast<Vector4<float> *>(aDataOut + 144);
    Vector4<float> *f2 = reinterpret_cast<Vector4<float> *>(aDataOut + 144 + 16);

    Vector4<BFloat16> *bf12 = reinterpret_cast<Vector4<BFloat16> *>(aDataOut + 176);
    Vector4<BFloat16> *bf22 = reinterpret_cast<Vector4<BFloat16> *>(aDataOut + 176 + 8);
    Vector4<HalfFp16> *hf12 = reinterpret_cast<Vector4<HalfFp16> *>(aDataOut + 176 + 16);
    Vector4<HalfFp16> *hf22 = reinterpret_cast<Vector4<HalfFp16> *>(aDataOut + 176 + 24);
    Vector4<float> *f12     = reinterpret_cast<Vector4<float> *>(aDataOut + 176 + 32);
    Vector4<float> *f22     = reinterpret_cast<Vector4<float> *>(aDataOut + 176 + 48);

    const T f1in = aDataIn[0];
    const T f2in = aDataIn[1];

    *i1  = Vector4<int>(f1in);
    *i2  = Vector4<int>(f2in);
    *ui1 = Vector4<uint>(f1in);
    *ui2 = Vector4<uint>(f2in);

    *s1  = Vector4<short>(f1in);
    *s2  = Vector4<short>(f2in);
    *us1 = Vector4<ushort>(f1in);
    *us2 = Vector4<ushort>(f2in);

    *sb1 = Vector4<sbyte>(f1in);
    *sb2 = Vector4<sbyte>(f2in);
    *b1  = Vector4<byte>(f1in);
    *b2  = Vector4<byte>(f2in);

    *bf1 = Vector4<BFloat16>(f1in);
    *bf2 = Vector4<BFloat16>(f2in);

    *hf1 = Vector4<HalfFp16>(f1in);
    *hf2 = Vector4<HalfFp16>(f2in);

    *f1 = Vector4<float>(f1in);
    *f2 = Vector4<float>(f2in);

    // bypass the SIMD float2BFloat converters in Vector4:
    bf12->x = BFloat16(static_cast<float>(f1in.x));
    bf12->y = BFloat16(static_cast<float>(f1in.y));
    bf12->z = BFloat16(static_cast<float>(f1in.z));
    bf12->w = BFloat16(static_cast<float>(f1in.w));
    bf22->x = BFloat16(static_cast<float>(f2in.x));
    bf22->y = BFloat16(static_cast<float>(f2in.y));
    bf22->z = BFloat16(static_cast<float>(f2in.z));
    bf22->w = BFloat16(static_cast<float>(f2in.w));

    // bypass the SIMD float2Half converters in Vector4:
    hf12->x = HalfFp16(static_cast<float>(f1in.x));
    hf12->y = HalfFp16(static_cast<float>(f1in.y));
    hf12->z = HalfFp16(static_cast<float>(f1in.z));
    hf12->w = HalfFp16(static_cast<float>(f1in.w));
    hf22->x = HalfFp16(static_cast<float>(f2in.x));
    hf22->y = HalfFp16(static_cast<float>(f2in.y));
    hf22->z = HalfFp16(static_cast<float>(f2in.z));
    hf22->w = HalfFp16(static_cast<float>(f2in.w));

    // bypass the SIMD Half2float or Bfloat2float, depending on T, converters in Vector4:
    f12->x = static_cast<float>(f1in.x);
    f12->y = static_cast<float>(f1in.y);
    f12->z = static_cast<float>(f1in.z);
    f12->w = static_cast<float>(f1in.w);
    f22->x = static_cast<float>(f2in.x);
    f22->y = static_cast<float>(f2in.y);
    f22->z = static_cast<float>(f2in.z);
    f22->w = static_cast<float>(f2in.w);
}

template <typename T> void runtest_typecast_kernel(char *aDataOut, T *aDataIn)
{
    test_typecast_kernel<<<1, 1>>>(aDataOut, aDataIn);
}

template void runtest_typecast_kernel<Vector4<float>>(char *, Vector4<float> *aDataIn);
template void runtest_typecast_kernel<Vector4<BFloat16>>(char *, Vector4<BFloat16> *aDataIn);
template void runtest_typecast_kernel<Vector4<HalfFp16>>(char *, Vector4<HalfFp16> *aDataIn);
} // namespace cuda
} // namespace opp