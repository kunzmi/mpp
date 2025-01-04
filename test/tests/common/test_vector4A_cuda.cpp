#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVar.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
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
template <typename T> void runtest_vector4A_kernel(Vector4A<T> *aDataIn, Vector4A<T> *aDataOut, Pixel8uC4A *aComp);
} // namespace opp::cuda

template <typename T> Vector4A<T> GetRandomValue(std::default_random_engine & /*aEngine*/)
{
    return Vector4A<T>();
}
template <typename T>
Vector4A<T> GetRandomValue(std::default_random_engine &aEngine)
    requires NativeFloatingPoint<T>
{
    std::uniform_real_distribution<T> uniform_dist(numeric_limits<T>::lowest() / static_cast<T>(100000),
                                                   numeric_limits<T>::max() / static_cast<T>(100000));
    return Vector4A<T>(uniform_dist(aEngine), uniform_dist(aEngine), uniform_dist(aEngine));
}
template <typename T>
Vector4A<T> GetRandomValue(std::default_random_engine &aEngine)
    requires NativeIntegral<T> && (!ByteSizeType<T>) || ComplexIntegral<T>
{
    std::uniform_int_distribution<complex_basetype_t<T>> uniform_dist(numeric_limits<complex_basetype_t<T>>::lowest(),
                                                                      numeric_limits<complex_basetype_t<T>>::max());
    return Vector4A<T>(uniform_dist(aEngine), uniform_dist(aEngine), uniform_dist(aEngine));
}
template <typename T>
Vector4A<T> GetRandomValue(std::default_random_engine &aEngine)
    requires NativeIntegral<T> && ByteSizeType<T>
{
    std::uniform_int_distribution<int> uniform_dist(static_cast<int>(numeric_limits<T>::lowest()),
                                                    static_cast<int>(numeric_limits<T>::max()));
    return Vector4A<T>(static_cast<T>(uniform_dist(aEngine)), static_cast<T>(uniform_dist(aEngine)),
                       static_cast<T>(uniform_dist(aEngine)));
}
template <> Vector4A<BFloat16> GetRandomValue<BFloat16>(std::default_random_engine &aEngine)
{
    std::uniform_real_distribution<float> uniform_dist(-10000.0_bf, 10000.0_bf);
    return Vector4A<BFloat16>(BFloat16(uniform_dist(aEngine)), BFloat16(uniform_dist(aEngine)),
                              BFloat16(uniform_dist(aEngine)));
}
template <> Vector4A<HalfFp16> GetRandomValue<HalfFp16>(std::default_random_engine &aEngine)
{
    std::uniform_real_distribution<float> uniform_dist(-10, 10);
    return Vector4A<HalfFp16>(HalfFp16(uniform_dist(aEngine)), HalfFp16(uniform_dist(aEngine)),
                              HalfFp16(uniform_dist(aEngine)));
}

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
        return numeric_limits<BFloat16>::min(); // should always be identical on GPU and CPU but we want to check for
                                                // NAN
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
        return numeric_limits<BFloat16>::min(); // should always be identical on GPU and CPU but we want to check for
                                                // NAN
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
void fillData(std::vector<Vector4A<T>> &aDataIn, std::vector<Vector4A<T>> &aDataOut, std::vector<Pixel8uC4A> &aComp,
              std::vector<complex_basetype_t<T>> &aEpsilon)
{
    std::default_random_engine e1(Catch::getSeed());
    // std::default_random_engine e1(3351725616UL);

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

        // no integer division by 0:
        if constexpr (RealIntegral<T>)
        {
            if (aDataIn[counterIn + 1].x == T(0))
            {
                aDataIn[counterIn + 1].x = T(2);
            }
            if (aDataIn[counterIn + 3].x == T(0))
            {
                aDataIn[counterIn + 3].x = T(2);
            }
            if (aDataIn[counterIn + 3].y == T(0))
            {
                aDataIn[counterIn + 3].y = T(2);
            }
            if (aDataIn[counterIn + 3].z == T(0))
            {
                aDataIn[counterIn + 3].z = T(2);
            }
            if (aDataIn[counterIn + 4].x == T(0))
            {
                aDataIn[counterIn + 4].x = T(2);
            }
            if (aDataIn[counterIn + 4].y == T(0))
            {
                aDataIn[counterIn + 4].y = T(2);
            }
            if (aDataIn[counterIn + 4].z == T(0))
            {
                aDataIn[counterIn + 4].z = T(2);
            }
            if (aDataIn[counterIn + 7].x == T(0))
            {
                aDataIn[counterIn + 7].x = T(2);
            }
            if (aDataIn[counterIn + 7].y == T(0))
            {
                aDataIn[counterIn + 7].y = T(2);
            }
            if (aDataIn[counterIn + 7].z == T(0))
            {
                aDataIn[counterIn + 7].z = T(2);
            }
        }

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
            // make all input values for shift positive:
            if constexpr (RealSignedIntegral<T>)
            {
                aDataIn[counterIn].Abs();
                aDataIn[counterIn + 2].Abs();
                aDataIn[counterIn + 4].Abs();
                aDataIn[counterIn + 6].Abs();
            }

            // limit bit shift to int size in bits minus 1 (also always positive)
            uint bitmask = static_cast<uint>(sizeof(T)) * 8 - 1;

            aDataIn[counterIn + 1].And(static_cast<T>(bitmask));
            aDataIn[counterIn + 3].And(static_cast<T>(bitmask));
            aDataIn[counterIn + 5].And(static_cast<T>(bitmask));
            aDataIn[counterIn + 7].And(static_cast<T>(bitmask));

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].LShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::LShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::RShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].And(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::And(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Or(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::Or(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Xor(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::Xor(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Not();
            aDataOut[counterOut + 1] = Vector4A<T>::Not(aDataIn[counterIn + 1]);
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
        aDataOut[counterOut + 1] = Vector4A<T>::Sqr(aDataIn[counterIn + 1]);
        aEpsilon[counterOut]     = largeEps<T>();
        aEpsilon[counterOut + 1] = largeEps<T>();
        counterOut += 2;
        counterIn += 2;

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            // limit input range for exp:
            if constexpr (std::same_as<double, complex_basetype_t<T>>)
            {
                T factor(10000.0);
                aDataIn[counterIn] /= factor;
                aDataIn[counterIn + 1] /= factor;
            }
            if constexpr (std::same_as<float, complex_basetype_t<T>>)
            {
                T factor(10.0f);
                aDataIn[counterIn] /= factor;
                aDataIn[counterIn + 1] /= factor;
            }
            if constexpr (std::same_as<BFloat16, complex_basetype_t<T>>)
            {
                aDataIn[counterIn] /= 10000.0_bf;
                aDataIn[counterIn + 1] /= 10000.0_bf;
            }
            if constexpr (std::same_as<HalfFp16, complex_basetype_t<T>>)
            {
                aDataIn[counterIn] /= 1.0_hf;
                aDataIn[counterIn + 1] /= 1.0_hf;
            }
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Exp();
            aDataOut[counterOut + 1] = Vector4A<T>::Exp(aDataIn[counterIn + 1]);
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ln();
            aDataOut[counterOut + 1] = Vector4A<T>::Ln(aDataIn[counterIn + 1]);
            aEpsilon[counterOut]     = largeEps<T>();
            aEpsilon[counterOut + 1] = largeEps<T>();
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Sqrt();
            aDataOut[counterOut + 1] = Vector4A<T>::Sqrt(aDataIn[counterIn + 1]);
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
            aDataOut[counterOut + 1] = Vector4A<T>::Abs(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        if constexpr (RealFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].AbsDiff(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::AbsDiff(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aEpsilon[counterOut]     = smallEps<T>();
            aEpsilon[counterOut + 1] = smallEps<T>();
            counterOut += 2;
            counterIn += 4;
        }

        if constexpr (ComplexNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Conj();
            aDataOut[counterOut + 1] = Vector4A<T>::Conj(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].ConjMul(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector4A<T>::ConjMul(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
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
        complex_basetype_t<T> minVal;
        complex_basetype_t<T> maxVal;
        if constexpr (NativeNumber<T>)
        {
            minVal = std::min(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
            maxVal = std::max(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
        }
        else if constexpr (ComplexNumber<T> && NativeNumber<complex_basetype_t<T>>)
        {
            minVal = std::min(aDataIn[counterIn + 1].x.real, aDataIn[counterIn + 2].x.real);
            maxVal = std::max(aDataIn[counterIn + 1].x.real, aDataIn[counterIn + 2].x.real);
        }
        else if constexpr (ComplexNumber<T> && !NativeNumber<complex_basetype_t<T>>)
        {
            minVal = complex_basetype_t<T>::Min(aDataIn[counterIn + 1].x.real, aDataIn[counterIn + 2].x.real);
            maxVal = complex_basetype_t<T>::Max(aDataIn[counterIn + 1].x.real, aDataIn[counterIn + 2].x.real);
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
            aDataOut[counterOut + 1] = Vector4A<T>::Min(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);

            aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
            aDataOut[counterOut + 2].Max(aDataIn[counterIn + 5]);
            aDataOut[counterOut + 3] = Vector4A<T>::Max(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterOut += 4;
            counterIn += 8;
        }

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Round();
            aDataOut[counterOut + 1] = Vector4A<T>::Round(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Floor();
            aDataOut[counterOut + 1] = Vector4A<T>::Floor(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ceil();
            aDataOut[counterOut + 1] = Vector4A<T>::Ceil(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundNearest();
            aDataOut[counterOut + 1] = Vector4A<T>::RoundNearest(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundZero();
            aDataOut[counterOut + 1] = Vector4A<T>::RoundZero(aDataIn[counterIn + 1]);
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
        if constexpr (ComplexNumber<T>)
        {
            // the number being random, they will unlikely be similiar enough...
            if (aDataIn[counterIn].x.real < aDataIn[counterIn + 1].x.real)
            {
                // half of the time it will be true
                aDataIn[counterIn + 1] = aDataIn[counterIn];
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2];
            }
            else
            {
                // half of the time it will be false
                aDataIn[counterIn + 1] = aDataIn[counterIn] + largeEps<T>();
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2] + largeEps<T>();
            }
        }
        else
        {
            // the number being random, they will unlikely be similiar enough...
            if (aDataIn[counterIn].x < aDataIn[counterIn + 1].x)
            {
                // half of the time it will be true
                aDataIn[counterIn + 1] = aDataIn[counterIn];
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2];
            }
            else
            {
                // half of the time it will be false
                aDataIn[counterIn + 1] = aDataIn[counterIn] + largeEps<T>();
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2] + largeEps<T>();
            }
        }

        aComp[counterComp]     = Vector4A<T>::CompareEQ(aDataIn[counterIn], aDataIn[counterIn + 1]);
        aComp[counterComp + 1] = Vector4A<T>::CompareNEQ(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
        counterComp += 2;
        counterIn += 4;

        if constexpr (RealNumber<T>)
        {
            aComp[counterComp]     = Vector4A<T>::CompareGE(aDataIn[counterIn], aDataIn[counterIn + 1]);
            aComp[counterComp + 1] = Vector4A<T>::CompareGT(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aComp[counterComp + 2] = Vector4A<T>::CompareLE(aDataIn[counterIn + 4], aDataIn[counterIn + 5]);
            aComp[counterComp + 3] = Vector4A<T>::CompareLT(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterComp += 4;
            counterIn += 8;
        }

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            if constexpr (ComplexNumber<T>)
            {
                // the number being random, they will unlikely be similiar enough...
                if (aDataIn[counterIn].x.real < aDataIn[counterIn + 1].x.real)
                {
                    // half of the time it will be true
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + T(smallEps<T>(), smallEps<T>());
                    aDataIn[counterIn + 2] = T(largeEps<T>(), largeEps<T>());
                }
                else
                {
                    // half of the time it will be false
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + T(largeEps<T>(), largeEps<T>());
                    aDataIn[counterIn + 2] = T(smallEps<T>(), smallEps<T>());
                }
            }
            else
            {
                // the number being random, they will unlikely be similiar enough...
                if (aDataIn[counterIn].x < aDataIn[counterIn + 1].x)
                {
                    // half of the time it will be true
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + smallEps<T>();
                    aDataIn[counterIn + 2] = largeEps<T>();
                }
                else
                {
                    // half of the time it will be false
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + largeEps<T>();
                    aDataIn[counterIn + 2] = smallEps<T>();
                }
            }
            if constexpr (ComplexNumber<T>)
            {
                aComp[counterComp] = Vector4A<T>::CompareEQEps(aDataIn[counterIn], aDataIn[counterIn + 1],
                                                               aDataIn[counterIn + 2].x.real);
            }
            else
            {
                aComp[counterComp] =
                    Vector4A<T>::CompareEQEps(aDataIn[counterIn], aDataIn[counterIn + 1], aDataIn[counterIn + 2].x);
            }
            counterComp += 1;
            counterIn += 3;
        }

        /*counterComp = 6;
        counterIn   = 12;*/
    }
    // if (blockIdx.x == 9)
    {
        counterComp = 10;  // 6 - 4; // 6
        counterIn   = 220; // 120 - 44;

        if constexpr (ComplexNumber<T>)
        {
            // the number being random, they will unlikely be similiar enough...
            if (aDataIn[counterIn].x.real < aDataIn[counterIn + 1].x.real)
            {
                // half of the time it will be true
                aDataIn[counterIn + 1] = aDataIn[counterIn];
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2];
            }
            else
            {
                // half of the time it will be false
                aDataIn[counterIn + 1] = aDataIn[counterIn] + largeEps<T>();
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2] + largeEps<T>();
            }
        }
        else
        {
            // the number being random, they will unlikely be similiar enough...
            if (aDataIn[counterIn].x < aDataIn[counterIn + 1].x)
            {
                // half of the time it will be true
                aDataIn[counterIn + 1] = aDataIn[counterIn];
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2];
            }
            else
            {
                // half of the time it will be false
                aDataIn[counterIn + 1] = aDataIn[counterIn] + largeEps<T>();
                aDataIn[counterIn + 3] = aDataIn[counterIn + 2] + largeEps<T>();
            }
        }

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
        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            if constexpr (ComplexNumber<T>)
            {
                // the number being random, they will unlikely be similiar enough...
                if (aDataIn[counterIn].x.real < aDataIn[counterIn + 1].x.real)
                {
                    // half of the time it will be true
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + T(smallEps<T>(), smallEps<T>());
                    aDataIn[counterIn + 2] = T(largeEps<T>(), largeEps<T>());
                }
                else
                {
                    // half of the time it will be false
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + T(largeEps<T>(), largeEps<T>());
                    aDataIn[counterIn + 2] = T(smallEps<T>(), smallEps<T>());
                }
            }
            else
            {
                // the number being random, they will unlikely be similiar enough...
                if (aDataIn[counterIn].x < aDataIn[counterIn + 1].x)
                {
                    // half of the time it will be true
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + smallEps<T>();
                    aDataIn[counterIn + 2] = largeEps<T>();
                }
                else
                {
                    // half of the time it will be false
                    aDataIn[counterIn + 1] = aDataIn[counterIn] + largeEps<T>();
                    aDataIn[counterIn + 2] = smallEps<T>();
                }
            }

            if constexpr (ComplexNumber<T>)
            {
                aComp[counterComp] =
                    static_cast<byte>(static_cast<byte>(Vector4A<T>::EqEps(aDataIn[counterIn], aDataIn[counterIn + 1],
                                                                           aDataIn[counterIn + 2].x.real)) *
                                      TRUE_VALUE);
            }
            else
            {
                aComp[counterComp] =
                    static_cast<byte>(static_cast<byte>(Vector4A<T>::EqEps(aDataIn[counterIn], aDataIn[counterIn + 1],
                                                                           aDataIn[counterIn + 2].x)) *
                                      TRUE_VALUE);
            }
            counterComp += 1;
            counterIn += 3;
        }

        /* counterComp = 7;
         counterIn   = 15;*/
    }
}

TEST_CASE("Pixel64fC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel64fC4A> dataIn(235);
    std::vector<Pixel64fC4A> dataOut(170, Pixel64fC4A(0));
    std::vector<Pixel64fC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<double> epsilon(170, 0.0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel64fC4A> d_dataIn(235);
    DevVar<Pixel64fC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0)
        {
            CHECK(Pixel64fC4A::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
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

TEST_CASE("Pixel32fC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel32fC4A> dataIn(235);
    std::vector<Pixel32fC4A> dataOut(170, Pixel32fC4A(0));
    std::vector<Pixel32fC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<float> epsilon(170, 0.0f);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel32fC4A> d_dataIn(235);
    DevVar<Pixel32fC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0f)
        {
            CHECK(Pixel32fC4A::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
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

TEST_CASE("Pixel16bfC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel16bfC4A> dataIn(235);
    std::vector<Pixel16bfC4A> dataOut(170, Pixel16bfC4A(0));
    std::vector<Pixel16bfC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<BFloat16> epsilon(170, 0.0_bf);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel16bfC4A> d_dataIn(235);
    DevVar<Pixel16bfC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0_hf)
        {
            CHECK(Pixel16bfC4A::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
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

TEST_CASE("Pixel16fC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel16fC4A> dataIn(235);
    std::vector<Pixel16fC4A> dataOut(170, Pixel16fC4A(0.0_hf));
    std::vector<Pixel16fC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<HalfFp16> epsilon(170, 0.0_hf);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel16fC4A> d_dataIn(235);
    DevVar<Pixel16fC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (epsilon[i] != 0.0_hf)
        {
            CHECK(Pixel16fC4A::EqEps(dataOut[i], dataOutGPU[i], epsilon[i]));
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

TEST_CASE("Pixel32sC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel32sC4A> dataIn(235);
    std::vector<Pixel32sC4A> dataOut(170, Pixel32sC4A(0));
    std::vector<Pixel32sC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<int> epsilon(170, 0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel32sC4A> d_dataIn(235);
    DevVar<Pixel32sC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        CHECK(dataOut[i] == dataOutGPU[i]);
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel32uC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel32uC4A> dataIn(235);
    std::vector<Pixel32uC4A> dataOut(170, Pixel32uC4A(0));
    std::vector<Pixel32uC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<uint> epsilon(170, 0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel32uC4A> d_dataIn(235);
    DevVar<Pixel32uC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        if (dataOut[i] != dataOutGPU[i])
        {
            std::cout << "Index wrong: " << i << std::endl;
        }
        CHECK(dataOut[i] == dataOutGPU[i]);
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel16sC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel16sC4A> dataIn(235);
    std::vector<Pixel16sC4A> dataOut(170, Pixel16sC4A(0));
    std::vector<Pixel16sC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<short> epsilon(170, 0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel16sC4A> d_dataIn(235);
    DevVar<Pixel16sC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    if constexpr (EnableSIMD<short>)
    {
        // In case we use SIMD on GPU, the numbers will be saturated to short.min/max() value, but the CPU overflows
        // and wraps arounds. To compensate for this, each result that is short.min/max() is seen as correct by setting
        // the CPU element to the same value as the GPU one.
        size_t counter = 0;
        for (size_t i = 0; i < dataOut.size(); i++)
        {
            for (int channel = 0; channel < 4; channel++)
            {
                if (dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<short>::lowest() ||
                    dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<short>::max())
                {
                    dataOut[i][static_cast<Axis4D>(channel)] = dataOutGPU[i][static_cast<Axis4D>(channel)];
                    counter++;
                }
            }
        }
    }

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        CHECK(dataOut[i] == dataOutGPU[i]);
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel16uC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel16uC4A> dataIn(235);
    std::vector<Pixel16uC4A> dataOut(170, Pixel16uC4A(0));
    std::vector<Pixel16uC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<ushort> epsilon(170, 0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel16uC4A> d_dataIn(235);
    DevVar<Pixel16uC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    if constexpr (EnableSIMD<ushort>)
    {
        // In case we use SIMD on GPU, the numbers will be saturated to ushort.min/max() value, but the CPU overflows
        // and wraps arounds. To compensate for this, each result that is ushort.min/max() is seen as correct by setting
        // the CPU element to the same value as the GPU one.
        size_t counter = 0;
        for (size_t i = 0; i < dataOut.size(); i++)
        {
            for (int channel = 0; channel < 4; channel++)
            {
                if (dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<ushort>::lowest() ||
                    dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<ushort>::max())
                {
                    dataOut[i][static_cast<Axis4D>(channel)] = dataOutGPU[i][static_cast<Axis4D>(channel)];
                    counter++;
                }
            }
        }
    }

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        CHECK(dataOut[i] == dataOutGPU[i]);
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel8sC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel8sC4A> dataIn(235);
    std::vector<Pixel8sC4A> dataOut(170, Pixel8sC4A(0));
    std::vector<Pixel8sC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<sbyte> epsilon(170, 0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel8sC4A> d_dataIn(235);
    DevVar<Pixel8sC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    if constexpr (EnableSIMD<sbyte>)
    {
        // In case we use SIMD on GPU, the numbers will be saturated to short.min/max() value, but the CPU overflows
        // and wraps arounds. To compensate for this, each result that is short.min/max() is seen as correct by setting
        // the CPU element to the same value as the GPU one.
        size_t counter = 0;
        for (size_t i = 0; i < dataOut.size(); i++)
        {
            for (int channel = 0; channel < 4; channel++)
            {
                if (dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<sbyte>::lowest() ||
                    dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<sbyte>::max())
                {
                    dataOut[i][static_cast<Axis4D>(channel)] = dataOutGPU[i][static_cast<Axis4D>(channel)];
                    counter++;
                }
            }
        }
    }

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        CHECK(dataOut[i] == dataOutGPU[i]);
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}

TEST_CASE("Pixel8uC4A CUDA", "[Common]")
{
    cudaSafeCall(cudaSetDevice(DEFAULT_CUDA_DEVICE_ID));

    std::vector<Pixel8uC4A> dataIn(235);
    std::vector<Pixel8uC4A> dataOut(170, Pixel8uC4A(0));
    std::vector<Pixel8uC4A> dataOutGPU(170);
    std::vector<Pixel8uC4A> dataComp(20);
    std::vector<Pixel8uC4A> dataCompGPU(20);
    std::vector<byte> epsilon(170, 0);

    fillData(dataIn, dataOut, dataComp, epsilon);

    DevVar<Pixel8uC4A> d_dataIn(235);
    DevVar<Pixel8uC4A> d_dataOut(170);
    DevVar<Pixel8uC4A> d_dataComp(20);

    d_dataIn << dataIn;
    d_dataOut.Memset(0);

    runtest_vector4A_kernel(d_dataIn.Pointer(), d_dataOut.Pointer(), d_dataComp.Pointer());

    d_dataOut >> dataOutGPU;
    d_dataComp >> dataCompGPU;

    if constexpr (EnableSIMD<byte>)
    {
        // In case we use SIMD on GPU, the numbers will be saturated to short.min/max() value, but the CPU overflows
        // and wraps arounds. To compensate for this, each result that is short.min/max() is seen as correct by setting
        // the CPU element to the same value as the GPU one.
        size_t counter = 0;
        for (size_t i = 0; i < dataOut.size(); i++)
        {
            for (int channel = 0; channel < 4; channel++)
            {
                if (dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<byte>::lowest() ||
                    dataOutGPU[i][static_cast<Axis4D>(channel)] == numeric_limits<byte>::max())
                {
                    dataOut[i][static_cast<Axis4D>(channel)] = dataOutGPU[i][static_cast<Axis4D>(channel)];
                    counter++;
                }
            }
        }
    }

    for (size_t i = 0; i < dataOut.size(); i++)
    {
        CHECK(dataOut[i] == dataOutGPU[i]);
    }
    for (size_t i = 0; i < dataComp.size(); i++)
    {
        CHECK(dataCompGPU[i] == dataComp[i]);
    }
}
