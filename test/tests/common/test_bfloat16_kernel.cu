#include <cfloat>
#include <cmath>
#include <common/bfloat16.h>
#include <common/bfloat16_impl.h>
#include <common/defines.h>
#include <device_launch_parameters.h>

#ifndef IS_HOST_COMPILER
#endif // !IS_HOST_COMPILER

namespace opp
{
namespace cuda
{

// avoid compiler warning "division by zero" by making it non-constant...
__device__ float getZero(float aVal)
{
    if (aVal == 0.0f)
    {
        return 0.0f;
    }
    return 0.0f;
}

__global__ void test_bfloat16_kernel(BFloat16 *aDataOut, bool *aBoolOut)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 || y > 0)
    {
        return;
    }

    BFloat16 dataIn[] = {BFloat16(0.4f),  BFloat16(0.5f),  BFloat16(0.6f),   BFloat16(1.5f), BFloat16(1.9f),
                         BFloat16(-1.5f), BFloat16(-2.5f), BFloat16(-10.3f), BFloat16(0.4f)};

    size_t offsetOut = 0;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp               = dataIn[i];
        aDataOut[offsetOut + 0 + i] = BFloat16::Round(dataIn[i]);
        temp.Round();
        aDataOut[offsetOut + 8 + i] = temp;
    }

    offsetOut = 2 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp               = dataIn[i];
        aDataOut[offsetOut + 0 + i] = BFloat16::Floor(dataIn[i]);
        temp.Floor();
        aDataOut[offsetOut + 8 + i] = temp;
    }

    offsetOut = 4 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp               = dataIn[i];
        aDataOut[offsetOut + 0 + i] = BFloat16::Ceil(dataIn[i]);
        temp.Ceil();
        aDataOut[offsetOut + 8 + i] = temp;
    }

    offsetOut = 6 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp               = dataIn[i];
        aDataOut[offsetOut + 0 + i] = BFloat16::RoundNearest(dataIn[i]);
        temp.RoundNearest();
        aDataOut[offsetOut + 8 + i] = temp;
    }

    offsetOut = 8 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp               = dataIn[i];
        aDataOut[offsetOut + 0 + i] = BFloat16::RoundZero(dataIn[i]);
        temp.RoundZero();
        aDataOut[offsetOut + 8 + i] = temp;
    }

    offsetOut = 10 * 8;

    aBoolOut[0] = dataIn[0] > dataIn[1];
    aBoolOut[1] = dataIn[1] > dataIn[0];

    aBoolOut[2] = dataIn[0] < dataIn[1];
    aBoolOut[3] = dataIn[1] < dataIn[0];

    aBoolOut[4] = dataIn[0] <= dataIn[1];
    aBoolOut[5] = dataIn[1] <= dataIn[0];
    aBoolOut[6] = dataIn[8] <= dataIn[0];

    aBoolOut[7] = dataIn[0] >= dataIn[1];
    aBoolOut[8] = dataIn[1] >= dataIn[0];
    aBoolOut[9] = dataIn[8] >= dataIn[0];

    aBoolOut[10] = dataIn[0] == dataIn[1];
    aBoolOut[11] = dataIn[8] == dataIn[0];

    aBoolOut[12] = dataIn[0] != dataIn[1];
    aBoolOut[13] = dataIn[8] != dataIn[0];

    // negate
    aDataOut[offsetOut + 0] = -dataIn[4];
    aDataOut[offsetOut + 1] = -dataIn[5];

    // +
    aDataOut[offsetOut + 2] = dataIn[0] + dataIn[1];
    // -
    aDataOut[offsetOut + 3] = dataIn[0] - dataIn[1];
    // *
    aDataOut[offsetOut + 4] = dataIn[0] * dataIn[1];
    // /
    aDataOut[offsetOut + 5] = dataIn[0] / dataIn[1];

    // +=
    BFloat16 temp = dataIn[0];
    temp += dataIn[1];
    aDataOut[offsetOut + 6] = temp;
    // -=
    temp = dataIn[0];
    temp -= dataIn[1];
    aDataOut[offsetOut + 7] = temp;
    // *=
    temp = dataIn[0];
    temp *= dataIn[1];
    aDataOut[offsetOut + 8] = temp;
    // /=
    temp = dataIn[0];
    temp /= dataIn[1];
    aDataOut[offsetOut + 9] = temp;

    offsetOut               = 90;
    aDataOut[offsetOut + 0] = BFloat16::Exp(dataIn[0]);
    aDataOut[offsetOut + 1] = BFloat16::Ln(dataIn[0]);
    aDataOut[offsetOut + 2] = BFloat16::Sqrt(dataIn[0]);
    aDataOut[offsetOut + 3] = BFloat16::Abs(dataIn[6]);

    temp = dataIn[0];
    temp.Exp();
    aDataOut[offsetOut + 4] = temp;

    temp = dataIn[0];
    temp.Ln();
    aDataOut[offsetOut + 5] = temp;

    temp = dataIn[0];
    temp.Sqrt();
    aDataOut[offsetOut + 6] = temp;

    temp = dataIn[6];
    temp.Abs();
    aDataOut[offsetOut + 7] = temp;

    temp                    = dataIn[3];
    temp                    = temp.GetSign();
    aDataOut[offsetOut + 8] = temp;

    temp                    = dataIn[5];
    temp                    = temp.GetSign();
    aDataOut[offsetOut + 9] = temp;

    offsetOut = 100;

    aDataOut[offsetOut + 0]  = BFloat16(0.0f);
    aDataOut[offsetOut + 1]  = BFloat16(-10.0f);
    aDataOut[offsetOut + 2]  = BFloat16(10.0f);
    aDataOut[offsetOut + 3]  = BFloat16(FLT_MAX);
    aDataOut[offsetOut + 4]  = BFloat16(-FLT_MAX);
    aDataOut[offsetOut + 5]  = BFloat16(1.0f / getZero(0.0f));
    aDataOut[offsetOut + 6]  = BFloat16(sqrtf(-1.0f));
    aDataOut[offsetOut + 7]  = BFloat16(INFINITY);
    aDataOut[offsetOut + 8]  = BFloat16(-INFINITY);
    aDataOut[offsetOut + 9]  = BFloat16(FLT_MIN);
    aDataOut[offsetOut + 10] = BFloat16(-0.0f);

    offsetOut = 110;
    for (size_t i = 0; i < 8; i++)
    {
        aDataOut[offsetOut + i] = dataIn[i];
    }
}

void runtest_bfloat16_kernel(BFloat16 *aDataOut, bool *aBoolOut)
{
    test_bfloat16_kernel<<<1, 1>>>(aDataOut, aBoolOut);
}
} // namespace cuda
} // namespace opp