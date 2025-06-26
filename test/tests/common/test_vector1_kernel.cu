#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/vector1.h>
#include <common/vectorTypes_impl.h>
#include <device_launch_parameters.h>

using namespace mpp::image;

namespace mpp
{
namespace cuda
{

template <typename T> __global__ void test_vector1_kernel(Vector1<T> *aDataIn, Vector1<T> *aDataOut, Pixel8uC1 *aComp)
{
    if (threadIdx.x > 0 || blockIdx.x > 9)
    {
        return;
    }

    size_t counterOut  = 0;
    size_t counterIn   = 0;
    size_t counterComp = 0;

    if (blockIdx.x == 0)
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
        counterOut += 3;
        counterIn += 6;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] -= aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] -= aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
        aDataOut[counterOut + 2].SubInv(aDataIn[counterIn + 5]);
        aDataOut[counterOut + 3] = aDataIn[counterIn + 6] - aDataIn[counterIn + 7];
    }
    if (blockIdx.x == 1)
    {
        counterOut = 20;
        counterIn  = 30;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] *= aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] *= aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4] * aDataIn[counterIn + 5];
        counterOut += 3;
        counterIn += 6;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut] /= aDataIn[counterIn + 1].x;
        aDataOut[counterOut + 1] = aDataIn[counterIn + 2];
        aDataOut[counterOut + 1] /= aDataIn[counterIn + 3];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
        aDataOut[counterOut + 2].DivInv(aDataIn[counterIn + 5]);
        aDataOut[counterOut + 3] = aDataIn[counterIn + 6] / aDataIn[counterIn + 7];
        counterOut += 4;
        counterIn += 8;

        aDataOut[counterOut] = aDataIn[counterIn][Axis1D::X];
    }
    if (blockIdx.x == 2)
    {
        counterOut = 40;
        counterIn  = 60;

        if constexpr (RealIntegral<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].LShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::LShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::RShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].And(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::And(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Or(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::Or(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Xor(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::Xor(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Not();
            aDataOut[counterOut + 1] = Vector1<T>::Not(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        counterOut = 12;
        counterIn  = 22;
    }
    if (blockIdx.x == 3)
    {
        counterOut = 70;
        counterIn  = 100;

        aDataOut[counterOut] = aDataIn[counterIn];
        aDataOut[counterOut].Sqr();
        aDataOut[counterOut + 1] = Vector1<T>::Sqr(aDataIn[counterIn + 1]);
        counterOut += 2;
        counterIn += 2;

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Exp();
            aDataOut[counterOut + 1] = Vector1<T>::Exp(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ln();
            aDataOut[counterOut + 1] = Vector1<T>::Ln(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Sqrt();
            aDataOut[counterOut + 1] = Vector1<T>::Sqrt(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }
    }
    if (blockIdx.x == 4)
    {
        counterOut = 100;
        counterIn  = 130;

        if constexpr (RealSignedNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Abs();
            aDataOut[counterOut + 1] = Vector1<T>::Abs(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        if constexpr (RealFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].AbsDiff(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::AbsDiff(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;
        }

        if constexpr (ComplexNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Conj();
            aDataOut[counterOut + 1] = Vector1<T>::Conj(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].ConjMul(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::ConjMul(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;
        }
    }
    if (blockIdx.x == 5)
    {
        counterOut = 120;
        counterIn  = 150;

        if constexpr (ComplexFloatingPoint<T>)
        {
            aDataOut[counterOut]     = aDataIn[counterIn].Magnitude();
            aDataOut[counterOut + 1] = aDataIn[counterIn + 1].MagnitudeSqr();
            aDataOut[counterOut + 2] = aDataIn[counterIn + 2].Angle();
            counterOut += 3;
            counterIn += 3;
        }
    }
    if (blockIdx.x == 6)
    {
        counterOut = 130;
        counterIn  = 160;

        aDataOut[counterOut] = aDataIn[counterIn];
        // we must make sure that clampMax is > clampMin, otherwise the result is not defined!
        complex_basetype_t<T> minVal;
        complex_basetype_t<T> maxVal;
        if constexpr (NativeNumber<T>)
        {
            minVal = min(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
            maxVal = max(aDataIn[counterIn + 1].x, aDataIn[counterIn + 2].x);
        }
        else if constexpr (ComplexNumber<T> && NativeNumber<complex_basetype_t<T>>)
        {
            minVal = min(aDataIn[counterIn + 1].x.real, aDataIn[counterIn + 2].x.real);
            maxVal = max(aDataIn[counterIn + 1].x.real, aDataIn[counterIn + 2].x.real);
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
    }
    if (blockIdx.x == 7)
    {
        counterOut = 150;
        counterIn  = 180;

        if constexpr (RealNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Min(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector1<T>::Min(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);

            aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
            aDataOut[counterOut + 2].Max(aDataIn[counterIn + 5]);
            aDataOut[counterOut + 3] = Vector1<T>::Max(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterOut += 4;
            counterIn += 8;
        }

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Round();
            aDataOut[counterOut + 1] = Vector1<T>::Round(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Floor();
            aDataOut[counterOut + 1] = Vector1<T>::Floor(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ceil();
            aDataOut[counterOut + 1] = Vector1<T>::Ceil(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundNearest();
            aDataOut[counterOut + 1] = Vector1<T>::RoundNearest(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundZero();
            aDataOut[counterOut + 1] = Vector1<T>::RoundZero(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }
    }
    if (blockIdx.x == 8)
    {
        counterIn   = 200;
        counterComp = 0;

        aComp[counterComp]     = Vector1<T>::CompareEQ(aDataIn[counterIn], aDataIn[counterIn + 1]);
        aComp[counterComp + 1] = Vector1<T>::CompareNEQ(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
        counterComp += 2;
        counterIn += 4;

        if constexpr (RealNumber<T>)
        {
            aComp[counterComp]     = Vector1<T>::CompareGE(aDataIn[counterIn], aDataIn[counterIn + 1]);
            aComp[counterComp + 1] = Vector1<T>::CompareGT(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aComp[counterComp + 2] = Vector1<T>::CompareLE(aDataIn[counterIn + 4], aDataIn[counterIn + 5]);
            aComp[counterComp + 3] = Vector1<T>::CompareLT(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterComp += 4;
            counterIn += 8;
        }
        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            if constexpr (ComplexNumber<T>)
            {
                aComp[counterComp] =
                    Vector1<T>::CompareEQEps(aDataIn[counterIn], aDataIn[counterIn + 1], aDataIn[counterIn + 2].x.real);
            }
            else
            {
                aComp[counterComp] =
                    Vector1<T>::CompareEQEps(aDataIn[counterIn], aDataIn[counterIn + 1], aDataIn[counterIn + 2].x);
            }
            counterComp += 1;
            counterIn += 3;
        }
    }
    if (blockIdx.x == 9)
    {
        counterComp = 10;
        counterIn   = 220;

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
                aComp[counterComp] =
                    static_cast<byte>(static_cast<byte>(Vector1<T>::EqEps(aDataIn[counterIn], aDataIn[counterIn + 1],
                                                                          aDataIn[counterIn + 2].x.real)) *
                                      TRUE_VALUE);
            }
            else
            {
                aComp[counterComp] =
                    static_cast<byte>(static_cast<byte>(Vector1<T>::EqEps(aDataIn[counterIn], aDataIn[counterIn + 1],
                                                                          aDataIn[counterIn + 2].x)) *
                                      TRUE_VALUE);
            }
            counterComp += 1;
            counterIn += 3;
        }
    }
    // 74  12
    //  132-52
    //  template <Number T2> [[nodiscard]] static Vector1<T> Convert(const Vector1<T2> &aVec)
}

template <typename T> void runtest_vector1_kernel(Vector1<T> *aDataIn, Vector1<T> *aDataOut, Pixel8uC1 *aComp)
{
    test_vector1_kernel<<<10, 1>>>(aDataIn, aDataOut, aComp);
}

template void runtest_vector1_kernel<sbyte>(Pixel8sC1 *aDataIn, Pixel8sC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<byte>(Pixel8uC1 *aDataIn, Pixel8uC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<short>(Pixel16sC1 *aDataIn, Pixel16sC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<ushort>(Pixel16uC1 *aDataIn, Pixel16uC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<int>(Pixel32sC1 *aDataIn, Pixel32sC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<uint>(Pixel32uC1 *aDataIn, Pixel32uC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<BFloat16>(Pixel16bfC1 *aDataIn, Pixel16bfC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<HalfFp16>(Pixel16fC1 *aDataIn, Pixel16fC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<float>(Pixel32fC1 *aDataIn, Pixel32fC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<double>(Pixel64fC1 *aDataIn, Pixel64fC1 *aDataOut, Pixel8uC1 *aComp);

template void runtest_vector1_kernel<c_short>(Pixel16scC1 *aDataIn, Pixel16scC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<c_int>(Pixel32scC1 *aDataIn, Pixel32scC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<c_BFloat16>(Pixel16bfcC1 *aDataIn, Pixel16bfcC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<c_HalfFp16>(Pixel16fcC1 *aDataIn, Pixel16fcC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<c_float>(Pixel32fcC1 *aDataIn, Pixel32fcC1 *aDataOut, Pixel8uC1 *aComp);
template void runtest_vector1_kernel<c_double>(Pixel64fcC1 *aDataIn, Pixel64fcC1 *aDataOut, Pixel8uC1 *aComp);

} // namespace cuda
} // namespace mpp