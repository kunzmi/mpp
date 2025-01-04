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
#include <common/vector3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace opp::image;

namespace opp
{
namespace cuda
{

template <typename T> __global__ void test_vector3_kernel(Vector3<T> *aDataIn, Vector3<T> *aDataOut, Pixel8uC3 *aComp)
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

        aDataOut[counterOut]     = aDataIn[counterIn][Axis3D::X];
        aDataOut[counterOut + 1] = aDataIn[counterIn + 1][Axis3D::Y];
        aDataOut[counterOut + 2] = aDataIn[counterIn + 2][Axis3D::Z];
    }
    if (blockIdx.x == 2)
    {
        counterOut = 40;
        counterIn  = 60;

        if constexpr (RealIntegral<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].LShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::LShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RShift(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::RShift(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].And(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::And(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Or(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::Or(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Xor(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::Xor(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Not();
            aDataOut[counterOut + 1] = Vector3<T>::Not(aDataIn[counterIn + 1]);
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
        aDataOut[counterOut + 1] = Vector3<T>::Sqr(aDataIn[counterIn + 1]);
        counterOut += 2;
        counterIn += 2;

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Exp();
            aDataOut[counterOut + 1] = Vector3<T>::Exp(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ln();
            aDataOut[counterOut + 1] = Vector3<T>::Ln(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Sqrt();
            aDataOut[counterOut + 1] = Vector3<T>::Sqrt(aDataIn[counterIn + 1]);
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
            aDataOut[counterOut + 1] = Vector3<T>::Abs(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }

        if constexpr (RealFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].AbsDiff(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::AbsDiff(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            counterOut += 2;
            counterIn += 4;
        }

        if constexpr (ComplexNumber<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Conj();
            aDataOut[counterOut + 1] = Vector3<T>::Conj(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].ConjMul(aDataIn[counterIn + 1]);
            aDataOut[counterOut + 1] = Vector3<T>::ConjMul(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
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
            aDataOut[counterOut + 1] = Vector3<T>::Min(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);

            aDataOut[counterOut + 2] = aDataIn[counterIn + 4];
            aDataOut[counterOut + 2].Max(aDataIn[counterIn + 5]);
            aDataOut[counterOut + 3] = Vector3<T>::Max(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterOut += 4;
            counterIn += 8;
        }

        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Round();
            aDataOut[counterOut + 1] = Vector3<T>::Round(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Floor();
            aDataOut[counterOut + 1] = Vector3<T>::Floor(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].Ceil();
            aDataOut[counterOut + 1] = Vector3<T>::Ceil(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundNearest();
            aDataOut[counterOut + 1] = Vector3<T>::RoundNearest(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;

            aDataOut[counterOut] = aDataIn[counterIn];
            aDataOut[counterOut].RoundZero();
            aDataOut[counterOut + 1] = Vector3<T>::RoundZero(aDataIn[counterIn + 1]);
            counterOut += 2;
            counterIn += 2;
        }
    }
    if (blockIdx.x == 8)
    {
        counterIn   = 200;
        counterComp = 0;

        aComp[counterComp]     = Vector3<T>::CompareEQ(aDataIn[counterIn], aDataIn[counterIn + 1]);
        aComp[counterComp + 1] = Vector3<T>::CompareNEQ(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
        counterComp += 2;
        counterIn += 4;

        if constexpr (RealNumber<T>)
        {
            aComp[counterComp]     = Vector3<T>::CompareGE(aDataIn[counterIn], aDataIn[counterIn + 1]);
            aComp[counterComp + 1] = Vector3<T>::CompareGT(aDataIn[counterIn + 2], aDataIn[counterIn + 3]);
            aComp[counterComp + 2] = Vector3<T>::CompareLE(aDataIn[counterIn + 4], aDataIn[counterIn + 5]);
            aComp[counterComp + 3] = Vector3<T>::CompareLT(aDataIn[counterIn + 6], aDataIn[counterIn + 7]);
            counterComp += 4;
            counterIn += 8;
        }
        if constexpr (RealOrComplexFloatingPoint<T>)
        {
            if constexpr (ComplexNumber<T>)
            {
                aComp[counterComp] =
                    Vector3<T>::CompareEQEps(aDataIn[counterIn], aDataIn[counterIn + 1], aDataIn[counterIn + 2].x.real);
            }
            else
            {
                aComp[counterComp] =
                    Vector3<T>::CompareEQEps(aDataIn[counterIn], aDataIn[counterIn + 1], aDataIn[counterIn + 2].x);
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
                    static_cast<byte>(static_cast<byte>(Vector3<T>::EqEps(aDataIn[counterIn], aDataIn[counterIn + 1],
                                                                          aDataIn[counterIn + 2].x.real)) *
                                      TRUE_VALUE);
            }
            else
            {
                aComp[counterComp] =
                    static_cast<byte>(static_cast<byte>(Vector3<T>::EqEps(aDataIn[counterIn], aDataIn[counterIn + 1],
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

template <typename T> void runtest_vector3_kernel(Vector3<T> *aDataIn, Vector3<T> *aDataOut, Pixel8uC3 *aComp)
{
    test_vector3_kernel<<<10, 1>>>(aDataIn, aDataOut, aComp);
}

template void runtest_vector3_kernel<sbyte>(Pixel8sC3 *aDataIn, Pixel8sC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<byte>(Pixel8uC3 *aDataIn, Pixel8uC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<short>(Pixel16sC3 *aDataIn, Pixel16sC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<ushort>(Pixel16uC3 *aDataIn, Pixel16uC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<int>(Pixel32sC3 *aDataIn, Pixel32sC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<uint>(Pixel32uC3 *aDataIn, Pixel32uC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<BFloat16>(Pixel16bfC3 *aDataIn, Pixel16bfC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<HalfFp16>(Pixel16fC3 *aDataIn, Pixel16fC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<float>(Pixel32fC3 *aDataIn, Pixel32fC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<double>(Pixel64fC3 *aDataIn, Pixel64fC3 *aDataOut, Pixel8uC3 *aComp);

template void runtest_vector3_kernel<c_short>(Pixel16scC3 *aDataIn, Pixel16scC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<c_int>(Pixel32scC3 *aDataIn, Pixel32scC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<c_BFloat16>(Pixel16bfcC3 *aDataIn, Pixel16bfcC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<c_HalfFp16>(Pixel16fcC3 *aDataIn, Pixel16fcC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<c_float>(Pixel32fcC3 *aDataIn, Pixel32fcC3 *aDataOut, Pixel8uC3 *aComp);
template void runtest_vector3_kernel<c_double>(Pixel64fcC3 *aDataIn, Pixel64fcC3 *aDataOut, Pixel8uC3 *aComp);

} // namespace cuda
} // namespace opp