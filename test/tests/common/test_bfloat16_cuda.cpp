#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVar.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <numeric>
#include <vector>

using namespace opp;
using namespace opp::cuda;
using namespace Catch;

namespace opp::cuda
{
void runtest_bfloat16_kernel(BFloat16 *aDataOut, bool *aBoolOut);
}

// avoid compiler error "division by zero" by making it non-constant...
float getZero(float aVal)
{
    if (aVal == 0.0f)
    {
        return 0.0f;
    }
    return 0.0f;
}

TEST_CASE("BFloat16 on Cuda", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<BFloat16> h_dataIn = {BFloat16(0.4f),  BFloat16(0.5f),   BFloat16(0.6f),
                                      BFloat16(1.5f),  BFloat16(1.9f),   BFloat16(-1.5f),
                                      BFloat16(-2.5f), BFloat16(-10.3f), BFloat16(0.4f)};
    std::vector<BFloat16> h_dataOut(116);
    std::vector<BFloat16> h_dataOutRef(116);
    bool h_dataBool[14];
    bool h_dataBoolRef[14];

    DevVar<BFloat16> d_dataOut(116);
    DevVar<bool> d_dataBool(14);

    runtest_bfloat16_kernel(d_dataOut.Pointer(), d_dataBool.Pointer());
    cudaSafeCall(cudaDeviceSynchronize());

    d_dataOut >> h_dataOut;
    d_dataBool >> h_dataBool;

    size_t offsetOut = 0;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp                   = h_dataIn[i];
        h_dataOutRef[offsetOut + 0 + i] = BFloat16::Round(h_dataIn[i]);
        temp.Round();
        h_dataOutRef[offsetOut + 8 + i] = temp;
    }

    offsetOut = 2 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp                   = h_dataIn[i];
        h_dataOutRef[offsetOut + 0 + i] = BFloat16::Floor(h_dataIn[i]);
        temp.Floor();
        h_dataOutRef[offsetOut + 8 + i] = temp;
    }

    offsetOut = 4 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp                   = h_dataIn[i];
        h_dataOutRef[offsetOut + 0 + i] = BFloat16::Ceil(h_dataIn[i]);
        temp.Ceil();
        h_dataOutRef[offsetOut + 8 + i] = temp;
    }

    offsetOut = 6 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp                   = h_dataIn[i];
        h_dataOutRef[offsetOut + 0 + i] = BFloat16::RoundNearest(h_dataIn[i]);
        temp.RoundNearest();
        h_dataOutRef[offsetOut + 8 + i] = temp;
    }

    offsetOut = 8 * 8;
    for (size_t i = 0; i < 8; i++)
    {
        BFloat16 temp                   = h_dataIn[i];
        h_dataOutRef[offsetOut + 0 + i] = BFloat16::RoundZero(h_dataIn[i]);
        temp.RoundZero();
        h_dataOutRef[offsetOut + 8 + i] = temp;
    }

    offsetOut = 10 * 8;

    h_dataBoolRef[0] = h_dataIn[0] > h_dataIn[1];
    h_dataBoolRef[1] = h_dataIn[1] > h_dataIn[0];

    h_dataBoolRef[2] = h_dataIn[0] < h_dataIn[1];
    h_dataBoolRef[3] = h_dataIn[1] < h_dataIn[0];

    h_dataBoolRef[4] = h_dataIn[0] <= h_dataIn[1];
    h_dataBoolRef[5] = h_dataIn[1] <= h_dataIn[0];
    h_dataBoolRef[6] = h_dataIn[8] <= h_dataIn[0];

    h_dataBoolRef[7] = h_dataIn[0] >= h_dataIn[1];
    h_dataBoolRef[8] = h_dataIn[1] >= h_dataIn[0];
    h_dataBoolRef[9] = h_dataIn[8] >= h_dataIn[0];

    h_dataBoolRef[10] = h_dataIn[0] == h_dataIn[1];
    h_dataBoolRef[11] = h_dataIn[8] == h_dataIn[0];

    h_dataBoolRef[12] = h_dataIn[0] != h_dataIn[1];
    h_dataBoolRef[13] = h_dataIn[8] != h_dataIn[0];

    // negate
    h_dataOutRef[offsetOut + 0] = -h_dataIn[4];
    h_dataOutRef[offsetOut + 1] = -h_dataIn[5];

    // +
    h_dataOutRef[offsetOut + 2] = h_dataIn[0] + h_dataIn[1];
    // -
    h_dataOutRef[offsetOut + 3] = h_dataIn[0] - h_dataIn[1];
    // *
    h_dataOutRef[offsetOut + 4] = h_dataIn[0] * h_dataIn[1];
    // /
    h_dataOutRef[offsetOut + 5] = h_dataIn[0] / h_dataIn[1];

    // +=
    BFloat16 temp = h_dataIn[0];
    temp += h_dataIn[1];
    h_dataOutRef[offsetOut + 6] = temp;
    // -=
    temp = h_dataIn[0];
    temp -= h_dataIn[1];
    h_dataOutRef[offsetOut + 7] = temp;
    // *=
    temp = h_dataIn[0];
    temp *= h_dataIn[1];
    h_dataOutRef[offsetOut + 8] = temp;
    // /=
    temp = h_dataIn[0];
    temp /= h_dataIn[1];
    h_dataOutRef[offsetOut + 9] = temp;

    offsetOut                   = 90;
    h_dataOutRef[offsetOut + 0] = BFloat16::Exp(h_dataIn[0]);
    h_dataOutRef[offsetOut + 1] = BFloat16::Ln(h_dataIn[0]);
    h_dataOutRef[offsetOut + 2] = BFloat16::Sqrt(h_dataIn[0]);
    h_dataOutRef[offsetOut + 3] = BFloat16::Abs(h_dataIn[6]);

    temp = h_dataIn[0];
    temp.Exp();
    h_dataOutRef[offsetOut + 4] = temp;

    temp = h_dataIn[0];
    temp.Ln();
    h_dataOutRef[offsetOut + 5] = temp;

    temp = h_dataIn[0];
    temp.Sqrt();
    h_dataOutRef[offsetOut + 6] = temp;

    temp = h_dataIn[6];
    temp.Abs();
    h_dataOutRef[offsetOut + 7] = temp;

    offsetOut = 98;

    h_dataOutRef[offsetOut + 0]  = BFloat16(0.0f);
    h_dataOutRef[offsetOut + 1]  = BFloat16(-10.0f);
    h_dataOutRef[offsetOut + 2]  = BFloat16(10.0f);
    h_dataOutRef[offsetOut + 3]  = BFloat16(FLT_MAX);
    h_dataOutRef[offsetOut + 4]  = BFloat16(-FLT_MAX);
    h_dataOutRef[offsetOut + 5]  = BFloat16(1.0f / getZero(0.0f));
    h_dataOutRef[offsetOut + 6]  = BFloat16(-sqrtf(-1.0f)); // cuda results in positive NAN
    h_dataOutRef[offsetOut + 7]  = BFloat16(INFINITY);
    h_dataOutRef[offsetOut + 8]  = BFloat16(-INFINITY);
    h_dataOutRef[offsetOut + 9]  = BFloat16(FLT_MIN);
    h_dataOutRef[offsetOut + 10] = BFloat16(-0.0f);

    offsetOut = 108;
    for (size_t i = 0; i < 8; i++)
    {
        h_dataOutRef[offsetOut + i] = h_dataIn[i];
    }

    ////////////////////////////////////////////////

    for (size_t i = 0; i < sizeof(h_dataBool); i++)
    {
        CHECK(h_dataBoolRef[i] == h_dataBool[i]);
    }

    for (size_t i = 0; i < h_dataOut.size(); i++)
    {
        if (isnan(h_dataOutRef[i]) || isnan(h_dataOut[i]))
        {
            if (isnan(h_dataOutRef[i]) && isnan(h_dataOut[i]))
            {
                // If both are NAN we consider the test as passed.
            }
            else
            {
                // if only one of the elements is NAN the check will fail and the test won't pass:
                CHECK(h_dataOutRef[i] == h_dataOut[i]);
            }
        }
        else
        {
            CHECK(h_dataOutRef[i] == h_dataOut[i]);
        }
    }
}
