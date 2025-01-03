#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVar.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <random>
#include <vector>

using namespace opp;
using namespace opp::cuda;
using namespace opp::image;
using namespace Catch;

namespace opp::cuda
{
template <typename T> Vector4<T> GetRandomValue(std::default_random_engine & /*aEngine*/)
{
    return Vector4<T>();
}
template <typename T>
Vector4<T> GetRandomValue(std::default_random_engine &aEngine)
    requires NativeFloatingPoint<T>
{
    std::uniform_real_distribution<T> uniform_dist(numeric_limits<T>::lowest() / static_cast<T>(100000),
                                                   numeric_limits<T>::max() / static_cast<T>(100000));
    return Vector4<T>(uniform_dist(aEngine), uniform_dist(aEngine), uniform_dist(aEngine), uniform_dist(aEngine));
}
template <typename T>
Vector4<T> GetRandomValue(std::default_random_engine &aEngine)
    requires NativeIntegral<T> && (!ByteSizeType<T>)
{
    std::uniform_int_distribution<T> uniform_dist(numeric_limits<T>::lowest(), numeric_limits<T>::max());
    return Vector4<T>(uniform_dist(aEngine), uniform_dist(aEngine), uniform_dist(aEngine), uniform_dist(aEngine));
}
template <typename T>
Vector4<T> GetRandomValue(std::default_random_engine &aEngine)
    requires NativeIntegral<T> && ByteSizeType<T>
{
    std::uniform_int_distribution<int> uniform_dist(static_cast<int>(numeric_limits<T>::lowest()),
                                                    static_cast<int>(numeric_limits<T>::max()));
    return Vector4<T>(static_cast<T>(uniform_dist(aEngine)), static_cast<T>(uniform_dist(aEngine)),
                      static_cast<T>(uniform_dist(aEngine)), static_cast<T>(uniform_dist(aEngine)));
}
template <> Vector4<BFloat16> GetRandomValue<BFloat16>(std::default_random_engine &aEngine)
{
    std::uniform_real_distribution<float> uniform_dist(numeric_limits<BFloat16>::lowest() / 10000.0_bf,
                                                       numeric_limits<BFloat16>::max() / 10000.0_bf);
    return Vector4<BFloat16>(BFloat16(uniform_dist(aEngine)), BFloat16(uniform_dist(aEngine)),
                             BFloat16(uniform_dist(aEngine)), BFloat16(uniform_dist(aEngine)));
}
template <> Vector4<HalfFp16> GetRandomValue<HalfFp16>(std::default_random_engine &aEngine)
{
    std::uniform_real_distribution<float> uniform_dist(-10, 10);
    return Vector4<HalfFp16>(HalfFp16(uniform_dist(aEngine)), HalfFp16(uniform_dist(aEngine)),
                             HalfFp16(uniform_dist(aEngine)), HalfFp16(uniform_dist(aEngine)));
}
template <typename T>
Vector4<T> GetRandomValue(std::default_random_engine &aEngine)
    requires ComplexFloatingPoint<T>
{
    std::uniform_real_distribution<complex_basetype_t<T>> uniform_dist(-100.0f, 100.0f);
    return Vector4<T>(T(uniform_dist(aEngine), uniform_dist(aEngine)), T(uniform_dist(aEngine), uniform_dist(aEngine)),
                      T(uniform_dist(aEngine), uniform_dist(aEngine)), T(uniform_dist(aEngine), uniform_dist(aEngine)));
}
template <> Pixel16bfcC4 GetRandomValue<c_BFloat16>(std::default_random_engine &aEngine)
{
    std::uniform_real_distribution<float> uniform_dist(-10000.0_bf, 10000.0_bf);
    return Pixel16bfcC4(
        c_BFloat16(static_cast<BFloat16>(uniform_dist(aEngine)), static_cast<BFloat16>(uniform_dist(aEngine))),
        c_BFloat16(static_cast<BFloat16>(uniform_dist(aEngine)), static_cast<BFloat16>(uniform_dist(aEngine))),
        c_BFloat16(static_cast<BFloat16>(uniform_dist(aEngine)), static_cast<BFloat16>(uniform_dist(aEngine))),
        c_BFloat16(static_cast<BFloat16>(uniform_dist(aEngine)), static_cast<BFloat16>(uniform_dist(aEngine))));
}
template <> Pixel16fcC4 GetRandomValue<c_HalfFp16>(std::default_random_engine &aEngine)
{
    std::uniform_real_distribution<float> uniform_dist(-10, 10);
    return Pixel16fcC4(
        c_HalfFp16(static_cast<HalfFp16>(uniform_dist(aEngine)), static_cast<HalfFp16>(uniform_dist(aEngine))),
        c_HalfFp16(static_cast<HalfFp16>(uniform_dist(aEngine)), static_cast<HalfFp16>(uniform_dist(aEngine))),
        c_HalfFp16(static_cast<HalfFp16>(uniform_dist(aEngine)), static_cast<HalfFp16>(uniform_dist(aEngine))),
        c_HalfFp16(static_cast<HalfFp16>(uniform_dist(aEngine)), static_cast<HalfFp16>(uniform_dist(aEngine))));
}

template <typename T> void runtest_vector4_kernel(Vector4<T> *aDataIn, Vector4<T> *aDataOut, Pixel8uC4 *aComp);

} // namespace opp::cuda

template <typename T> complex_basetype_t<T> smallEps()
{
    if constexpr (std::same_as<complex_basetype_t<T>, double>)
    {
        return 0.0000000001;
    }
    else if constexpr (std::same_as<complex_basetype_t<T>, float>)
    {
        return 0.00000001f;
    }
    else if constexpr (std::same_as<complex_basetype_t<T>, BFloat16>)
    {
        return 0.0_bf; // should always be identical on GPU and CPU
    }
    else if constexpr (std::same_as<complex_basetype_t<T>, HalfFp16>)
    {
        return 0.0001_hf;
    }
    else
    {
        return static_cast<complex_basetype_t<T>>(0);
    }
}
template <typename T> complex_basetype_t<T> largeEps()
{
    if constexpr (std::same_as<complex_basetype_t<T>, double>)
    {
        return 0.0001;
    }
    else if constexpr (std::same_as<complex_basetype_t<T>, float>)
    {
        return 0.01f;
    }
    else if constexpr (std::same_as<complex_basetype_t<T>, BFloat16>)
    {
        return 0.0_bf; // should always be identical on GPU and CPU
    }
    else if constexpr (std::same_as<complex_basetype_t<T>, HalfFp16>)
    {
        return 0.2_hf;
    }
    else
    {
        return static_cast<complex_basetype_t<T>>(0);
    }
}

template <typename T>
void fillData(std::vector<Vector4<T>> &aDataIn, std::vector<Vector4<T>> &aDataOut, std::vector<Pixel8uC4> &aComp,
              std::vector<complex_basetype_t<T>> &aEpsilon)
{
    std::default_random_engine e1(Catch::getSeed());

    // Create random values:
    for (auto &elem : aDataIn)
    {
        elem = GetRandomValue<T>(e1);
    }

    size_t counterOut  = 0;
    size_t counterIn   = 0;
    size_t counterComp = 0;

    // if (blockIdx.x == 0)
    {
        counterOut = 0;
        counterIn  = 0;

        if constexpr (ComplexNumber<T> || RealSignedNumber<T>)
        {
            aDataOut[counterOut] = -aDataIn[counterIn];
            counterOut += 1;
            counterIn += 1;
        }
        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] += aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] += aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4] + aDataIn[counterIn + 5];
        aEpsilon[counterOut]     = smallEps<T>();
        aEpsilon[counterOut + 1] = smallEps<T>();
        aEpsilon[counterOut + 2] = smallEps<T>();
        counterOut += 3;
        counterIn += 6;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] -= aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] -= aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
        aDataOut[counterOut + 2].SubInv(aDataIn[counterIn + 5]);
        aDataOut[counterOut + 3] = aDataIn[counterIn + 6] - aDataIn[counterIn + 7];
        aEpsilon[counterOut]     = smallEps<T>();
        aEpsilon[counterOut + 1] = smallEps<T>();
        aEpsilon[counterOut + 2] = smallEps<T>();
        aEpsilon[counterOut + 3] = smallEps<T>();
        /*counterOut += 4;
        counterIn += 8;

        counterOut = 8;
        counterIn  = 15;*/
    }
    // if (blockIdx.x == 1)
    {
        counterOut = 20; // 8
        counterIn  = 30; // 15

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] *= aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] *= aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4] * aDataIn[counterIn + 5];
        aEpsilon[counterOut]     = largeEps<T>();
        aEpsilon[counterOut + 1] = largeEps<T>();
        aEpsilon[counterOut + 2] = largeEps<T>();
        counterOut += 3;
        counterIn += 6;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] /= aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] /= aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
        aDataOut[counterOut + 2].DivInv(aDataIn[counterIn + 5]);
        aDataOut[counterOut + 3] = aDataIn[counterIn + 6] / aDataIn[counterIn + 7];
        aEpsilon[counterOut]     = largeEps<T>();
        aEpsilon[counterOut + 1] = largeEps<T>();
        aEpsilon[counterOut + 2] = largeEps<T>();
        aEpsilon[counterOut + 3] = largeEps<T>();
        counterOut += 4;
        counterIn += 8;

        aDataOut[counterOut]     = aDataIn[counterIn][Axis4D::X];
        aDataOut[counterOut + 1] = aDataIn[counterIn + 1][Axis4D::Y];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 2][Axis4D::Z];
        aDataOut[counterOut + 3] = aDataIn[counterIn + 3][Axis4D::W];
        /*counterOut += 4;
        counterIn += 4;

        counterOut = 11;
        counterIn  = 18;*/
    }
    // if (blockIdx.x == 2)
    {
        counterOut = 40; // 19
        counterIn  = 60; // 33;

        if constexpr (RealIntegral<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].LShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::LShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::RShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].And(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::And(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Or(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::Or(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Xor(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::Xor(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Not();
            aDataOut[counterOut + 1] = Vector4<T>::Not(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        /*counterOut = 12;
        counterIn  = 22;*/
    }
    // if (blockIdx.x == 3)
    {
        counterOut = 70;  // 31 - 12; // 31
        counterIn  = 100; // 55 - 22;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut].Sqr();
        aDataOut[counterOut + 1] = Vector4<T>::Sqr(aDataIn[counterIn + 1]);
        aEpsilon[counterOut]     = largeEps<T>();
        aEpsilon[counterOut + 1] = largeEps<T>();
        counterOut += 2;
        counterIn += 2;

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            // limit input range for exp:
            if constexpr (std::same_as<double, complex_basetype_t<T>>)
            {
                aDataIn[counterIn] /= 10000.0;
                aDataIn[counterIn + 1] /= 10000.0;
            }
            if constexpr (std::same_as<float, complex_basetype_t<T>>)
            {
                aDataIn[counterIn] /= 10.0f;
                aDataIn[counterIn + 1] /= 10.0f;
            }
            if constexpr (std::same_as<BFloat16, complex_basetype_t<T>>)
            {
                aDataIn[counterIn] /= 1000.0_bf;
                aDataIn[counterIn + 1] /= 1000.0_bf;
            }
            if constexpr (std::same_as<HalfFp16, complex_basetype_t<T>>)
            {
                aDataIn[counterIn] /= 1.0_hf;
                aDataIn[counterIn + 1] /= 1.0_hf;
            }
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Exp();
            aDataOut[counterOut + 1] = Vector4<T>::Exp(aDataIn[counterIn + 1]);
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ln();
            aDataOut[counterOut + 1] = Vector4<T>::Ln(aDataIn[counterIn + 1]);
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Sqrt();
            aDataOut[counterOut + 1] = Vector4<T>::Sqrt(aDataIn[counterIn + 1]);
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            counterOut += 2;
            counterIn += 2;
        }

        /*counterOut = 8;
        counterIn  = 8;*/
    }
    // if (blockIdx.x == 4)
    {
        counterOut = 100; // 39 - 12; // 39
        counterIn  = 130; // 63 - 22;

        if constexpr (RealSignedNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Abs();
            aDataOut[counterOut + 1] = Vector4<T>::Abs(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        if constexpr (RealFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].AbsDiff(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::AbsDiff(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aEpsilon[counterOut]     = smallEps<T>();
            aEpsilon[counterOut + 1] = smallEps<T>();
            counterOut += 2;
            counterIn += 4;
        }

        if constexpr (ComplexNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Conj();
            aDataOut[counterOut + 1] = Vector4<T>::Conj(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].ConjMul(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::ConjMul(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            counterOut += 2;
            counterIn += 4;
        }

        /*counterOut = 8;
        counterIn  = 12;*/
    }
    // if (blockIdx.x == 5)
    {
        counterOut = 120; // 47 - 16; // 47
        counterIn  = 150; // 75 - 28;

        if constexpr (ComplexFloatingPoint<T>)
        {
            aDataOut[counterOut]     = aDataIn[counterIn].Magnitude();
            aDataOut[counterOut + 1] = aDataIn[counterIn + 1].MagnitudeSqr();
            aDataOut[counterOut + 2] = aDataIn[counterIn + 2].Angle();
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            aEpsilon[counterOut + 2] = largeEps<T>();
            counterOut += 3;
            counterIn += 3;
        }

        /*counterOut = 3;
        counterIn  = 3;*/
    }
    // if (blockIdx.x == 6)
    {
        counterOut = 130; // 50 - 16; // 50
        counterIn  = 160; //  78 - 28;

        aDataOut[counterOut] = aDataIn[counterIn];
        // we must make sure that clampMax is > clampMin, otherwise the result is wrong!
        T minVal;
        T maxVal;
        if constexpr (NativeNumber<T>)
        {
            minVal = min(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
            maxVal = max(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
        }
        else
        {
            minVal = T::Min(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
            maxVal = T::Max(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
        }
        aDataOut[counterOut].Clamp(minVal, maxVal);

        aDataOut[counterOut + 1] = aDataIn[counterIn + 3];
        aDataOut[counterOut + 1].template ClampToTargetType<byte>();

        aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
        aDataOut[counterOut + 2].template ClampToTargetType<sbyte>();

        aDataOut[counterOut + 3] = aDataIn[counterIn + 5];
        aDataOut[counterOut + 3].template ClampToTargetType<ushort>();

        aDataOut[counterOut + 4] = aDataIn[counterIn + 6];
        aDataOut[counterOut + 4].template ClampToTargetType<short>();

        aDataOut[counterOut + 5] = aDataIn[counterIn + 7];
        aDataOut[counterOut + 6].template ClampToTargetType<uint>();

        aDataOut[counterOut + 6] = aDataIn[counterIn + 8];
        aDataOut[counterOut + 6].template ClampToTargetType<int>();

        aDataOut[counterOut + 7] = aDataIn[counterIn + 9];
        aDataOut[counterOut + 7].template ClampToTargetType<float>();

        aDataOut[counterOut + 8] = aDataIn[counterIn + 10];
        aDataOut[counterOut + 8].template ClampToTargetType<HalfFp16>();

        aDataOut[counterOut + 9] = aDataIn[counterIn + 11];
        aDataOut[counterOut + 9].template ClampToTargetType<BFloat16>();

        /*counterOut += 10;
        counterIn += 12;

        counterOut = 10;
        counterIn  = 12;*/
    }
    // if (blockIdx.x == 7)
    {
        counterOut = 150; // 60 - 16; // 60
        counterIn  = 180; // 90 - 28;

        if constexpr (RealNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Min(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4<T>::Min(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);

            aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
            aDataOut[counterOut + 2].Max(aDataIn[counterIn + 5]);
            aDataOut[counterOut + 3] = Vector4<T>::Max(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterOut += 4;
            counterIn += 8;
        }

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Round();
            aDataOut[counterOut + 1] = Vector4<T>::Round(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Floor();
            aDataOut[counterOut + 1] = Vector4<T>::Floor(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ceil();
            aDataOut[counterOut + 1] = Vector4<T>::Ceil(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundNearest();
            aDataOut[counterOut + 1] = Vector4<T>::RoundNearest(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundZero();
            aDataOut[counterOut + 1] = Vector4<T>::RoundZero(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        /*counterOut = 14;
        counterIn  = 18;*/
    }
    // if (blockIdx.x == 8)
    {
        // counterOut  = 74; // 74
        counterIn   = 200; // 108 - 36;
        counterComp = 0;

        aComp[counterComp]     = Vector4<T>::CompareEQ(aDataIn[counterIn], aDataIn[counterIn + 1]);
        aComp[counterComp + 1] = Vector4<T>::CompareNEQ(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
        counterComp += 2;
        counterIn += 4;

        if constexpr (RealNumber<T>)
        {
            aComp[counterComp]     = Vector4<T>::CompareGE(aDataIn[counterIn], aDataIn[counterIn + 1]);
            aComp[counterComp + 1] = Vector4<T>::CompareGT(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aComp[counterComp + 2] = Vector4<T>::CompareLE(aDataIn[counterIn + 4], aDataIn[counterIn + 5]);
            aComp[counterComp + 3] = Vector4<T>::CompareLT(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterComp += 4;
            counterIn += 8;
        }

        /*counterComp = 6;
        counterIn   = 12;*/
    }
    // if (blockIdx.x == 9)
    {
        counterComp = 10;  // 6 - 4; // 6
        counterIn   = 220; // 120 - 44;

        aComp[counterComp] =
            static_cast<byte>(static_cast<byte>(aDataIn[counterIn] == aDataIn[counterIn + 1]) * TRUE_VALUE);
        aComp[counterComp + 1] =
            static_cast<byte>(static_cast<byte>(aDataIn[counterIn + 2] != aDataIn[counterIn + 3]) * TRUE_VALUE);
        counterComp += 2;
        counterIn += 4;

        if constexpr (RealNumber<T>)
        {
            aComp[counterComp] =
                static_cast<byte>(static_cast<byte>(aDataIn[counterIn] >= aDataIn[counterIn + 1]) * TRUE_VALUE);
            aComp[counterComp + 1] =
                static_cast<byte>(static_cast<byte>(aDataIn[counterIn + 2] > aDataIn[counterIn + 3]) * TRUE_VALUE);
            aComp[counterComp + 2] =
                static_cast<byte>(static_cast<byte>(aDataIn[counterIn + 4] <= aDataIn[counterIn + 5]) * TRUE_VALUE);
            aComp[counterComp + 3] =
                static_cast<byte>(static_cast<byte>(aDataIn[counterIn + 6] < aDataIn[counterIn + 7]) * TRUE_VALUE);
            counterComp += 4;
            counterIn += 8;
        }

        /* counterComp = 6;
         counterIn   = 12;*/
    }
}

TEST_CASE("Pixel64fcC4 CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel64fcC4> dataIn(232);
    std::vector<Pixel64fcC4> dataOut(170, Pixel64fcC4(0.0));
    std::vector<Pixel64fcC4> dataOutGPU(170);
    std::vector<Pixel8uC4> dataComp(20);
    std::vector<Pixel8uC4> dataCompGPU(20);
    std::vector<double> epsilon(170, 0.0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel64fcC4> d_dataIn(232);
    DevVar<Pixel64fcC4> d_dataOut(170);
    DevVar<Pixel8uC4> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0)
        {
            if (!Pixel64fcC4::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]))
            {
                std::cout << i << ": " << std::endl << dataOut[i] << " != " << std::endl << dataOutGPU[i] << std::endl;
                Pixel64fcC4 diff = dataOut[i];
                diff.AbsDiff(dataOutGPU[i]);
                Complex<float> max = diff.Max();
                std::cout << "MaxDiff: " << max << std::endl;
            }
            CHECK(Pixel64fcC4::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
        }
        else
        {
            CHECK(dataOut[i] == dataOutGPU[i]);
        }
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel32fcC4 CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel32fcC4> dataIn(232);
    std::vector<Pixel32fcC4> dataOut(170, Pixel32fcC4(0.0f));
    std::vector<Pixel32fcC4> dataOutGPU(170);
    std::vector<Pixel8uC4> dataComp(20);
    std::vector<Pixel8uC4> dataCompGPU(20);
    std::vector<float> epsilon(170, 0.0f);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel32fcC4> d_dataIn(232);
    DevVar<Pixel32fcC4> d_dataOut(170);
    DevVar<Pixel8uC4> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0f)
        {
            CHECK(Pixel32fcC4::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
        }
        else
        {
            CHECK(dataOut[i] == dataOutGPU[i]);
        }
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel16bfcC4 CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel16bfcC4> dataIn(232);
    std::vector<Pixel16bfcC4> dataOut(170, Pixel16bfcC4(0.0_bf));
    std::vector<Pixel16bfcC4> dataOutGPU(170);
    std::vector<Pixel8uC4> dataComp(20);
    std::vector<Pixel8uC4> dataCompGPU(20);
    std::vector<BFloat16> epsilon(170, 0.0_bf);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel16bfcC4> d_dataIn(232);
    DevVar<Pixel16bfcC4> d_dataOut(170);
    DevVar<Pixel8uC4> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0_hf)
        {
            CHECK(Pixel16bfcC4::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
        }
        else
        {
            CHECK(dataOut[i] == dataOutGPU[i]);
        }
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel16fcC4 CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel16fcC4> dataIn(232);
    std::vector<Pixel16fcC4> dataOut(170, Pixel16fcC4(0.0_hf));
    std::vector<Pixel16fcC4> dataOutGPU(170);
    std::vector<Pixel8uC4> dataComp(20);
    std::vector<Pixel8uC4> dataCompGPU(20);
    std::vector<HalfFp16> epsilon(170, 0.0_hf);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel16fcC4> d_dataIn(232);
    DevVar<Pixel16fcC4> d_dataOut(170);
    DevVar<Pixel8uC4> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0_hf)
        {
            CHECK(Pixel16fcC4::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
        }
        else
        {
            CHECK(dataOut[i] == dataOutGPU[i]);
        }
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

// TEST_CASE("Vector<sbyte> SIMD CUDA", "[Common]")
//{
//     cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));
//
//     std::vector<Pixel8sC4> dataIn(103, 100);
//     std::vector<Pixel8sC4> dataOut(49);
//     std::vector<Pixel8uC4> dataComp(12);
//
//     fillData(dataIn, dataOut, dataComp);
//
//     runtest_vector4_kernel(dataIn.data(), dataOut.data(), dataComp.data());
// }

TEST_CASE("Vector<short> SIMD CUDA", "[Common]")
{
    // cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    // DevVar<Vector4<sbyte>> sbytes(37);
    // DevVar<Pixel8uC4> compSBytes(37);
    // DevVar<Pixel8uC4> bytes(37);
    // DevVar<Pixel8uC4> compBytes(37);
    // DevVar<Vector4<short>> shorts(37);
    // DevVar<Pixel8uC4> compShorts(37);
    // DevVar<Vector4<ushort>> ushorts(37);
    // DevVar<Pixel8uC4> compUShorts(37);
    // DevVar<Vector4<BFloat16>> bfloats(37);
    // DevVar<Pixel8uC4> compBFloats(37);
    // DevVar<Vector4<HalfFp16>> halfs(37);
    // DevVar<Pixel8uC4> compHalfs(37);
    // DevVar<Vector4<float>> floats(37);
    // DevVar<Pixel8uC4> compFloats(37);

    // runtest_vector4_kernel(bytes.Pointer(), compBytes.Pointer(),     //
    //                        sbytes.Pointer(), compSBytes.Pointer(),   //
    //                        shorts.Pointer(), compShorts.Pointer(),   //
    //                        ushorts.Pointer(), compUShorts.Pointer(), //
    //                        bfloats.Pointer(), compBFloats.Pointer(), //
    //                        halfs.Pointer(), compHalfs.Pointer(),     //
    //                        floats.Pointer(), compFloats.Pointer());

    // cudaSafeCall(cudaDeviceSynchronize());
    // std::vector<Vector4<sbyte>> resSBytes(37);
    // std::vector<Pixel8uC4> resCompSBytes(37);
    // std::vector<Pixel8uC4> resBytes(37);
    // std::vector<Pixel8uC4> resCompBytes(37);
    // std::vector<Vector4<short>> resShort(37);
    // std::vector<Pixel8uC4> resCompShort(37);
    // std::vector<Vector4<ushort>> resUShort(37);
    // std::vector<Pixel8uC4> resCompUShort(37);
    // std::vector<Vector4<BFloat16>> resBFloat(37);
    // std::vector<Pixel8uC4> resCompBFloat(37);
    // std::vector<Vector4<HalfFp16>> resHalf(37);
    // std::vector<Pixel8uC4> resCompHalf(37);
    // std::vector<Vector4<float>> resFloat(37);
    // std::vector<Pixel8uC4> resCompFloat(37);

    // sbytes >> resSBytes;
    // compSBytes >> resCompSBytes;

    // bytes >> resBytes;
    // compBytes >> resCompBytes;

    // shorts >> resShort;
    // compShorts >> resCompShort;

    // ushorts >> resUShort;
    // compUShorts >> resCompUShort;

    // bfloats >> resBFloat;
    // compBFloats >> resCompBFloat;

    // halfs >> resHalf;
    // compHalfs >> resCompHalf;

    // floats >> resFloat;
    // compFloats >> resCompFloat;

    //// sbyte
    // Vector4<sbyte> sbA(12, -120, -10, 120);
    // Vector4<sbyte> sbB(120, -80, 30, 5);
    // Vector4<sbyte> sbC(12, 20, 0, 120);

    // CHECK(resSBytes[0] == Vector4<sbyte>(Vector4<float>(sbA) + Vector4<float>(sbB)));
    // CHECK(resSBytes[1] == Vector4<sbyte>(Vector4<float>(sbA) - Vector4<float>(sbB)));
    // CHECK(resSBytes[2] == sbA * sbB);
    // CHECK(resSBytes[3] == sbA / sbB);

    // CHECK(resSBytes[4] == Vector4<sbyte>(Vector4<float>(sbA) + Vector4<float>(sbB)));
    // CHECK(resSBytes[5] == Vector4<sbyte>(Vector4<float>(sbA) - Vector4<float>(sbB)));
    // CHECK(resSBytes[6] == sbA * sbB);
    // CHECK(resSBytes[7] == sbA / sbB);

    // CHECK(resSBytes[8] == Vector4<sbyte>(-12, 120, 10, -120));
    // CHECK(resSBytes[12] == Vector4<sbyte>(12, 120, 10, 120));
    // CHECK(resSBytes[13] == Vector4<sbyte>(108, 40, 40, 115));
    // CHECK(resSBytes[17] == Vector4<sbyte>(12, 120, 10, 120));
    // CHECK(resSBytes[18] == Vector4<sbyte>(108, 40, 40, 115));

    // CHECK(resSBytes[19] == Vector4<sbyte>(12, -120, -10, 5));
    // CHECK(resSBytes[20] == Vector4<sbyte>(120, -80, 30, 120));

    // CHECK(resSBytes[21] == Vector4<sbyte>(12, -120, -10, 5));
    // CHECK(resSBytes[22] == Vector4<sbyte>(120, -80, 30, 120));

    // CHECK(resSBytes[35] == Vector4<sbyte>(Vector4<float>(sbB) - Vector4<float>(sbA)));
    // CHECK(resSBytes[36] == sbB / sbA);

    // CHECK(resCompSBytes[0] == Vector4<sbyte>::CompareEQ(sbA, sbC));
    // CHECK(resCompSBytes[1] == Vector4<sbyte>::CompareLE(sbA, sbC));
    // CHECK(resCompSBytes[2] == Vector4<sbyte>::CompareLT(sbA, sbC));
    // CHECK(resCompSBytes[3] == Vector4<sbyte>::CompareGE(sbA, sbC));
    // CHECK(resCompSBytes[4] == Vector4<sbyte>::CompareGT(sbA, sbC));
    // CHECK(resCompSBytes[5] == Vector4<sbyte>::CompareNEQ(sbA, sbC));

    //// ==
    // CHECK(resCompSBytes[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompSBytes[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompSBytes[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[18] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[19] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompSBytes[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompSBytes[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompSBytes[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompSBytes[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompSBytes[36] == Pixel8uC4(byte(1)));

    //// byte
    // Pixel8uC4 bA(12, 120, 100, 120);
    // Pixel8uC4 bB(120, 180, 30, 5);
    // Pixel8uC4 bC(12, 20, 100, 120);

    // CHECK(resBytes[0] == Pixel8uC4(Vector4<float>(bA) + Vector4<float>(bB)));
    // CHECK(resBytes[1] == Pixel8uC4(Vector4<float>(bA) - Vector4<float>(bB)));
    // CHECK(resBytes[2] == bA * bB);
    // CHECK(resBytes[3] == bA / bB);

    // CHECK(resBytes[4] == Pixel8uC4(Vector4<float>(bA) + Vector4<float>(bB)));
    // CHECK(resBytes[5] == Pixel8uC4(Vector4<float>(bA) - Vector4<float>(bB)));
    // CHECK(resBytes[6] == bA * bB);
    // CHECK(resBytes[7] == bA / bB);

    // CHECK(resBytes[13] == Pixel8uC4(108, 60, 70, 115));
    // CHECK(resBytes[18] == Pixel8uC4(108, 60, 70, 115));

    // CHECK(resBytes[19] == Pixel8uC4(12, 120, 30, 5));
    // CHECK(resBytes[20] == Pixel8uC4(120, 180, 100, 120));
    // CHECK(resBytes[21] == Pixel8uC4(12, 120, 30, 5));
    // CHECK(resBytes[22] == Pixel8uC4(120, 180, 100, 120));

    // CHECK(resBytes[35] == Pixel8uC4(Vector4<float>(bB) - Vector4<float>(bA)));
    // CHECK(resBytes[36] == bB / bA);

    // CHECK(resCompBytes[0] == Pixel8uC4::CompareEQ(bA, bC));
    // CHECK(resCompBytes[1] == Pixel8uC4::CompareLE(bA, bC));
    // CHECK(resCompBytes[2] == Pixel8uC4::CompareLT(bA, bC));
    // CHECK(resCompBytes[3] == Pixel8uC4::CompareGE(bA, bC));
    // CHECK(resCompBytes[4] == Pixel8uC4::CompareGT(bA, bC));
    // CHECK(resCompBytes[5] == Pixel8uC4::CompareNEQ(bA, bC));

    //// ==
    // CHECK(resCompBytes[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompBytes[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompBytes[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[18] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[19] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompBytes[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompBytes[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompBytes[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompBytes[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompBytes[36] == Pixel8uC4(byte(1)));

    //// short
    // Vector4<short> sA(12, -30000, -10, 31000);
    // Vector4<short> sB(120, -3000, 30, -4096);
    // Vector4<short> sC(12, -30000, 0, 31000);

    // CHECK(resShort[0] == Vector4<short>(Vector4<float>(sA) + Vector4<float>(sB)));
    // CHECK(resShort[1] == Vector4<short>(Vector4<float>(sA) - Vector4<float>(sB)));
    // CHECK(resShort[2] == sA * sB);
    // CHECK(resShort[3] == sA / sB);

    // CHECK(resShort[4] == Vector4<short>(Vector4<float>(sA) + Vector4<float>(sB)));
    // CHECK(resShort[5] == Vector4<short>(Vector4<float>(sA) - Vector4<float>(sB)));
    // CHECK(resShort[6] == sA * sB);
    // CHECK(resShort[7] == sA / sB);

    // CHECK(resShort[8] == Vector4<short>(-12, 30000, 10, -31000));
    // CHECK(resShort[12] == Vector4<short>(12, 30000, 10, 31000));
    // CHECK(resShort[13] == Vector4<short>(108, 27000, 40, -30440));
    // CHECK(resShort[17] == Vector4<short>(12, 30000, 10, 31000));
    // CHECK(resShort[18] == Vector4<short>(108, 27000, 40, -30440));

    // CHECK(resShort[19] == Vector4<short>(12, -30000, -10, -4096));
    // CHECK(resShort[20] == Vector4<short>(120, -3000, 30, 31000));

    // CHECK(resShort[21] == Vector4<short>(12, -30000, -10, -4096));
    // CHECK(resShort[22] == Vector4<short>(120, -3000, 30, 31000));

    // CHECK(resShort[35] == Vector4<short>(Vector4<float>(sB) - Vector4<float>(sA)));
    // CHECK(resShort[36] == sB / sA);

    // CHECK(resCompShort[0] == Vector4<short>::CompareEQ(sA, sC));
    // CHECK(resCompShort[1] == Vector4<short>::CompareLE(sA, sC));
    // CHECK(resCompShort[2] == Vector4<short>::CompareLT(sA, sC));
    // CHECK(resCompShort[3] == Vector4<short>::CompareGE(sA, sC));
    // CHECK(resCompShort[4] == Vector4<short>::CompareGT(sA, sC));
    // CHECK(resCompShort[5] == Vector4<short>::CompareNEQ(sA, sC));

    //// ==
    // CHECK(resCompShort[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompShort[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompShort[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[18] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[19] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompShort[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompShort[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompShort[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompShort[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompShort[36] == Pixel8uC4(byte(1)));

    //// ushort
    // Vector4<ushort> usA(12, 60000, 100, 120);
    // Vector4<ushort> usB(120, 7000, 30, 5);
    // Vector4<ushort> usC(12, 20, 100, 120);

    // CHECK(resUShort[0] == Vector4<ushort>(Vector4<float>(usA) + Vector4<float>(usB)));
    // CHECK(resUShort[1] == Vector4<ushort>(Vector4<float>(usA) - Vector4<float>(usB)));
    // CHECK(resUShort[2] == usA * usB);
    // CHECK(resUShort[3] == usA / usB);

    // CHECK(resUShort[4] == Vector4<ushort>(Vector4<float>(usA) + Vector4<float>(usB)));
    // CHECK(resUShort[5] == Vector4<ushort>(Vector4<float>(usA) - Vector4<float>(usB)));
    // CHECK(resUShort[6] == usA * usB);
    // CHECK(resUShort[7] == usA / usB);

    // CHECK(resUShort[13] == Vector4<ushort>(108, 53000, 70, 115));
    // CHECK(resUShort[18] == Vector4<ushort>(108, 53000, 70, 115));

    // CHECK(resUShort[19] == Vector4<ushort>(12, 7000, 30, 5));
    // CHECK(resUShort[20] == Vector4<ushort>(120, 60000, 100, 120));
    // CHECK(resUShort[21] == Vector4<ushort>(12, 7000, 30, 5));
    // CHECK(resUShort[22] == Vector4<ushort>(120, 60000, 100, 120));

    // CHECK(resUShort[35] == Vector4<ushort>(Vector4<float>(usB) - Vector4<float>(usA)));
    // CHECK(resUShort[36] == usB / usA);

    // CHECK(resCompUShort[0] == Vector4<ushort>::CompareEQ(usA, usC));
    // CHECK(resCompUShort[1] == Vector4<ushort>::CompareLE(usA, usC));
    // CHECK(resCompUShort[2] == Vector4<ushort>::CompareLT(usA, usC));
    // CHECK(resCompUShort[3] == Vector4<ushort>::CompareGE(usA, usC));
    // CHECK(resCompUShort[4] == Vector4<ushort>::CompareGT(usA, usC));
    // CHECK(resCompUShort[5] == Vector4<ushort>::CompareNEQ(usA, usC));

    //// ==
    // CHECK(resCompUShort[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompUShort[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompUShort[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[18] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[19] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompUShort[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompUShort[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompUShort[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompUShort[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompUShort[36] == Pixel8uC4(byte(1)));

    //// BFloat16
    // Vector4<BFloat16> bfA(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f), BFloat16(310.12f));
    // Vector4<BFloat16> bfB(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f), BFloat16(-4096.9f));
    // Vector4<BFloat16> bfC(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(0.0f), BFloat16(310.12f));

    // CHECK(resBFloat[0] == bfA + bfB);
    // CHECK(resBFloat[1] == bfA - bfB);
    // CHECK(resBFloat[2] == bfA * bfB);
    // CHECK(resBFloat[3] == bfA / bfB);

    // CHECK(resBFloat[4] == bfA + bfB);
    // CHECK(resBFloat[5] == bfA - bfB);
    // CHECK(resBFloat[6] == bfA * bfB);
    // CHECK(resBFloat[7] == bfA / bfB);

    // CHECK(resBFloat[8] == Vector4<BFloat16>(BFloat16(-12.4f), BFloat16(30000.2f), BFloat16(10.5f),
    // BFloat16(-310.12f))); CHECK(resBFloat[9] == Vector4<BFloat16>::Exp(bfC));

    // CHECK(resBFloat[10].x == Vector4<BFloat16>::Ln(bfC).x);
    // CHECK(isnan(resBFloat[10].y));
    // CHECK(resBFloat[10].z == Vector4<BFloat16>::Ln(bfC).z);
    // CHECK(resBFloat[10].w == Vector4<BFloat16>::Ln(bfC).w);

    // CHECK(resBFloat[11].x == Vector4<BFloat16>::Sqrt(bfC).x);
    // CHECK(isnan(resBFloat[11].y));
    // CHECK(resBFloat[11].z == Vector4<BFloat16>::Sqrt(bfC).z);
    // CHECK(resBFloat[11].w == Vector4<BFloat16>::Sqrt(bfC).w);

    // CHECK(resBFloat[12] == Vector4<BFloat16>(BFloat16(12.4f), BFloat16(30000.2f), BFloat16(10.5f),
    // BFloat16(310.12f))); CHECK(resBFloat[17] == Vector4<BFloat16>(BFloat16(12.4f), BFloat16(30000.2f),
    // BFloat16(10.5f), BFloat16(310.12f)));

    // CHECK(resBFloat[19] ==
    //       Vector4<BFloat16>(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f), BFloat16(-4096.9f)));
    // CHECK(resBFloat[20] == Vector4<BFloat16>(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f),
    // BFloat16(310.12f)));

    // CHECK(resBFloat[21] ==
    //       Vector4<BFloat16>(BFloat16(12.4f), BFloat16(-30000.2f), BFloat16(-10.5f), BFloat16(-4096.9f)));
    // CHECK(resBFloat[22] == Vector4<BFloat16>(BFloat16(120.1f), BFloat16(-3000.1f), BFloat16(30.2f),
    // BFloat16(310.12f)));

    // CHECK(resBFloat[23] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f),
    // BFloat16(310.0f))); CHECK(resBFloat[24] == Vector4<BFloat16>(BFloat16(13.0f), BFloat16(-29952.0f),
    // BFloat16(-10.0f), BFloat16(310.0f))); CHECK(resBFloat[25] == Vector4<BFloat16>(BFloat16(12.0f),
    // BFloat16(-29952.0f), BFloat16(-11.0f), BFloat16(310.0f))); CHECK(resBFloat[26] ==
    // Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f))); CHECK(resBFloat[27]
    // == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));

    // CHECK(resBFloat[28] == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-11.0f),
    // BFloat16(310.0f))); CHECK(resBFloat[29] == Vector4<BFloat16>(BFloat16(13.0f), BFloat16(-29952.0f),
    // BFloat16(-10.0f), BFloat16(310.0f))); CHECK(resBFloat[30] == Vector4<BFloat16>(BFloat16(12.0f),
    // BFloat16(-29952.0f), BFloat16(-11.0f), BFloat16(310.0f))); CHECK(resBFloat[31] ==
    // Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f))); CHECK(resBFloat[32]
    // == Vector4<BFloat16>(BFloat16(12.0f), BFloat16(-29952.0f), BFloat16(-10.0f), BFloat16(310.0f)));

    // CHECK(resBFloat[33] == Vector4<BFloat16>(BFloat16(1.0f), BFloat16(2.0f), BFloat16(3.0f), BFloat16(4.0f)));
    // CHECK(resBFloat[34] == Vector4<BFloat16>(BFloat16(2.0f), BFloat16(4.0f), BFloat16(6.0f), BFloat16(8.0f)));

    // CHECK(resBFloat[35] == bfB - bfA);
    // CHECK(resBFloat[36] == bfB / bfA);

    // CHECK(resCompBFloat[0] == Vector4<BFloat16>::CompareEQ(bfA, bfC));
    // CHECK(resCompBFloat[1] == Vector4<BFloat16>::CompareLE(bfA, bfC));
    // CHECK(resCompBFloat[2] == Vector4<BFloat16>::CompareLT(bfA, bfC));
    // CHECK(resCompBFloat[3] == Vector4<BFloat16>::CompareGE(bfA, bfC));
    // CHECK(resCompBFloat[4] == Vector4<BFloat16>::CompareGT(bfA, bfC));
    // CHECK(resCompBFloat[5] == Vector4<BFloat16>::CompareNEQ(bfA, bfC));

    //// ==
    // CHECK(resCompBFloat[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompBFloat[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompBFloat[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[18] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[19] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompBFloat[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompBFloat[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompBFloat[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompBFloat[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompBFloat[36] == Pixel8uC4(byte(1)));

    //// Half
    // Vector4<HalfFp16> hA(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f), HalfFp16(310.12f));
    // Vector4<HalfFp16> hB(HalfFp16(120.1f), HalfFp16(-3000.1f), HalfFp16(30.2f), HalfFp16(-4096.9f));
    // Vector4<HalfFp16> hC(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(0.0f), HalfFp16(310.12f));

    // CHECK(resHalf[0] == hA + hB);
    // CHECK(resHalf[1] == hA - hB);
    // CHECK(resHalf[2] == hA * hB);
    // CHECK(resHalf[3] == hA / hB);

    // CHECK(resHalf[4] == hA + hB);
    // CHECK(resHalf[5] == hA - hB);
    // CHECK(resHalf[6] == hA * hB);
    // CHECK(resHalf[7] == hA / hB);

    // CHECK(resHalf[8] == Vector4<HalfFp16>(HalfFp16(-12.4f), HalfFp16(30000.2f), HalfFp16(10.5f),
    // HalfFp16(-310.12f))); CHECK(resHalf[9] == Vector4<HalfFp16>::Exp(hC));

    // CHECK(resHalf[10].x == Vector4<HalfFp16>::Ln(hC).x);
    // CHECK(isnan(resHalf[10].y));
    // CHECK(resHalf[10].z == Vector4<HalfFp16>::Ln(hC).z);
    // CHECK(resHalf[10].w == Vector4<HalfFp16>::Ln(hC).w);

    // CHECK(resHalf[11].x == Vector4<HalfFp16>::Sqrt(hC).x);
    // CHECK(isnan(resBFloat[11].y));
    // CHECK(resHalf[11].z == Vector4<HalfFp16>::Sqrt(hC).z);
    // CHECK(resHalf[11].w == Vector4<HalfFp16>::Sqrt(hC).w);

    // CHECK(resHalf[12] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(30000.2f), HalfFp16(10.5f), HalfFp16(310.12f)));
    // CHECK(resHalf[17] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(30000.2f), HalfFp16(10.5f), HalfFp16(310.12f)));

    // CHECK(resHalf[19] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f),
    // HalfFp16(-4096.9f))); CHECK(resHalf[20] == Vector4<HalfFp16>(HalfFp16(120.1f), HalfFp16(-3000.1f),
    // HalfFp16(30.2f), HalfFp16(310.12f)));

    // CHECK(resHalf[21] == Vector4<HalfFp16>(HalfFp16(12.4f), HalfFp16(-30000.2f), HalfFp16(-10.5f),
    // HalfFp16(-4096.9f))); CHECK(resHalf[22] == Vector4<HalfFp16>(HalfFp16(120.1f), HalfFp16(-3000.1f),
    // HalfFp16(30.2f), HalfFp16(310.12f)));

    // CHECK(resHalf[23] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    // CHECK(resHalf[24] == Vector4<HalfFp16>(HalfFp16(13), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    // CHECK(resHalf[25] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    // CHECK(resHalf[26] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    // CHECK(resHalf[27] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));

    // CHECK(resHalf[28] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    // CHECK(resHalf[29] == Vector4<HalfFp16>(HalfFp16(13), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    // CHECK(resHalf[30] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-11), HalfFp16(310)));
    // CHECK(resHalf[31] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));
    // CHECK(resHalf[32] == Vector4<HalfFp16>(HalfFp16(12), HalfFp16(-30000), HalfFp16(-10), HalfFp16(310)));

    // CHECK(resHalf[33] == Vector4<HalfFp16>(HalfFp16(1), HalfFp16(2), HalfFp16(3), HalfFp16(4)));
    // CHECK(resHalf[34] == Vector4<HalfFp16>(HalfFp16(2), HalfFp16(4), HalfFp16(6), HalfFp16(8)));

    // CHECK(resHalf[35] == hB - hA);
    // CHECK(resHalf[36] == hB / hA);

    // CHECK(resCompHalf[0] == Vector4<HalfFp16>::CompareEQ(hA, hC));
    // CHECK(resCompHalf[1] == Vector4<HalfFp16>::CompareLE(hA, hC));
    // CHECK(resCompHalf[2] == Vector4<HalfFp16>::CompareLT(hA, hC));
    // CHECK(resCompHalf[3] == Vector4<HalfFp16>::CompareGE(hA, hC));
    // CHECK(resCompHalf[4] == Vector4<HalfFp16>::CompareGT(hA, hC));
    // CHECK(resCompHalf[5] == Vector4<HalfFp16>::CompareNEQ(hA, hC));

    //// ==
    // CHECK(resCompHalf[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompHalf[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompHalf[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[18] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[19] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompHalf[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompHalf[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompHalf[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompHalf[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompHalf[36] == Pixel8uC4(byte(1)));

    //// Float (32bit)
    // Vector4<float> fA(12.4f, -30000.2f, -10.5f, 310.12f);
    // Vector4<float> fB(120.1f, -3000.1f, 30.2f, -4096.9f);
    // Vector4<float> fC(12.4f, -30000.2f, 0.0f, 310.12f);

    // CHECK(resFloat[0] == fA + fB);
    // CHECK(resFloat[1] == fA - fB);
    // CHECK(resFloat[2] == fA * fB);
    // CHECK(resFloat[3] == fA / fB);

    // CHECK(resFloat[4] == fA + fB);
    // CHECK(resFloat[5] == fA - fB);
    // CHECK(resFloat[6] == fA * fB);
    // CHECK(resFloat[7] == fA / fB);

    // CHECK(resFloat[8] == Vector4<float>(-12.4f, 30000.2f, 10.5f, -310.12f));
    //// GCC slighlty differs for exact comparison:
    // CHECK(resFloat[9].x == Approx(Vector4<float>::Exp(fC).x).margin(0.001));
    // CHECK(resFloat[9].y == Vector4<float>::Exp(fC).y);
    // CHECK(resFloat[9].z == Vector4<float>::Exp(fC).z);
    // CHECK(resFloat[9].w == Vector4<float>::Exp(fC).w);

    // CHECK(resFloat[10].x == Vector4<float>::Ln(fC).x);
    // CHECK(isnan(resFloat[10].y));
    // CHECK(resFloat[10].z == Vector4<float>::Ln(fC).z);
    // CHECK(resFloat[10].w == Vector4<float>::Ln(fC).w);

    // CHECK(resFloat[11].x == Vector4<float>::Sqrt(fC).x);
    // CHECK(isnan(resFloat[11].y));
    // CHECK(resFloat[11].z == Vector4<float>::Sqrt(fC).z);
    // CHECK(resFloat[11].w == Vector4<float>::Sqrt(fC).w);

    // CHECK(resFloat[12] == Vector4<float>(12.4f, 30000.2f, 10.5f, 310.12f));
    // CHECK(resFloat[17] == Vector4<float>(12.4f, 30000.2f, 10.5f, 310.12f));

    // CHECK(resFloat[19] == Vector4<float>(12.4f, -30000.2f, -10.5f, -4096.9f));
    // CHECK(resFloat[20] == Vector4<float>(120.1f, -3000.1f, 30.2f, 310.12f));

    // CHECK(resFloat[21] == Vector4<float>(12.4f, -30000.2f, -10.5f, -4096.9f));
    // CHECK(resFloat[22] == Vector4<float>(120.1f, -3000.1f, 30.2f, 310.12f));

    //// Vector4<float> fA(12.4f, -30000.2f, -10.5f, 310.12f);
    // CHECK(resFloat[23] == Vector4<float>(12.0f, -30000.0f, -11.0f, 310.0f));
    // CHECK(resFloat[24] == Vector4<float>(13.0f, -30000.0f, -10.0f, 311.0f));
    // CHECK(resFloat[25] == Vector4<float>(12.0f, -30001.0f, -11.0f, 310.0f));
    // CHECK(resFloat[26] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));
    // CHECK(resFloat[27] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));

    // CHECK(resFloat[28] == Vector4<float>(12.0f, -30000.0f, -11.0f, 310.0f));
    // CHECK(resFloat[29] == Vector4<float>(13.0f, -30000.0f, -10.0f, 311.0f));
    // CHECK(resFloat[30] == Vector4<float>(12.0f, -30001.0f, -11.0f, 310.0f));
    // CHECK(resFloat[31] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));
    // CHECK(resFloat[32] == Vector4<float>(12.0f, -30000.0f, -10.0f, 310.0f));

    // CHECK(resFloat[33] == Vector4<float>(1.0f, 2.0f, 3.0f, 4.0f));
    // CHECK(resFloat[34] == Vector4<float>(2.0f, 4.0f, 6.0f, 8.0f));

    // CHECK(resFloat[35] == fB - fA);
    // CHECK(resFloat[36] == fB / fA);

    // CHECK(resCompFloat[0] == Vector4<float>::CompareEQ(fA, fC));
    // CHECK(resCompFloat[1] == Vector4<float>::CompareLE(fA, fC));
    // CHECK(resCompFloat[2] == Vector4<float>::CompareLT(fA, fC));
    // CHECK(resCompFloat[3] == Vector4<float>::CompareGE(fA, fC));
    // CHECK(resCompFloat[4] == Vector4<float>::CompareGT(fA, fC));
    // CHECK(resCompFloat[5] == Vector4<float>::CompareNEQ(fA, fC));

    //// ==
    // CHECK(resCompFloat[6] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[7] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[8] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[9] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[10] == Pixel8uC4(byte(0)));
    //// <=
    // CHECK(resCompFloat[11] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[12] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[13] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[14] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[15] == Pixel8uC4(byte(0)));
    //// <
    // CHECK(resCompFloat[16] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[17] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[18] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[19] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[20] == Pixel8uC4(byte(0)));
    //// >=
    // CHECK(resCompFloat[21] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[22] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[23] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[24] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[25] == Pixel8uC4(byte(0)));
    //// >
    // CHECK(resCompFloat[26] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[27] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[28] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[29] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[30] == Pixel8uC4(byte(0)));
    //// !=
    // CHECK(resCompFloat[31] == Pixel8uC4(byte(0)));
    // CHECK(resCompFloat[32] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[33] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[34] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[35] == Pixel8uC4(byte(1)));
    // CHECK(resCompFloat[36] == Pixel8uC4(byte(1)));
}