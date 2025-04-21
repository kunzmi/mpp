#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/opp_defs.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{

template <AnyVector T> struct SumScalar
{
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc) const
    {
        remove_vector_t<T> res = aSrc.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            res += aSrc.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            res += aSrc.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            res += aSrc.w;
        }
        return res;
    }
};

template <AnyVector T> struct SumThenSqrtScalar
{
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc) const
    {
        remove_vector_t<T> res = aSrc.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            res += aSrc.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            res += aSrc.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            res += aSrc.w;
        }
        return sqrt(res);
    }
};
template <AnyVector T> struct DivScalar
{
    const complex_basetype_t<remove_vector_t<T>> Divisor;
    DEVICE_CODE DivScalar(complex_basetype_t<remove_vector_t<T>> aDivisor)
        : Divisor(aDivisor * static_cast<complex_basetype_t<remove_vector_t<T>>>(vector_active_size_v<T>))
    {
    }

    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc) const
    {
        remove_vector_t<T> res = aSrc.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            res += aSrc.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            res += aSrc.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            res += aSrc.w;
        }
        return res /= Divisor;
    }
};
template <AnyVector T> struct PSNRScalar
{
    const remove_vector_t<T> Divisor; // pixel count for MSE
    const remove_vector_t<T> ValueRangeSqr;

    DEVICE_CODE PSNRScalar(remove_vector_t<T> aDivisor, remove_vector_t<T> aValueRange)
        : Divisor(aDivisor * static_cast<remove_vector_t<T>>(vector_active_size_v<T>)),
          ValueRangeSqr(aValueRange * aValueRange)
    {
    }

    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc) const
    {
        remove_vector_t<T> mse = aSrc.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            mse += aSrc.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            mse += aSrc.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            mse += aSrc.w;
        }
        mse /= Divisor;

        mse = ValueRangeSqr / mse;
        return static_cast<remove_vector_t<T>>(10) * log10(mse);
    }
};
template <AnyVector T> struct MinScalar
{
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc) const
    {
        if constexpr (vector_size_v<T> == 1)
        {
            return aSrc.x;
        }
        else
        {
            return aSrc.Min();
        }
    }
    // for compatibility with StdDeviation (needed in MinMax kernel)
    DEVICE_CODE void operator()(T & /*aSrc1*/, const T &aSrc2, remove_vector_t<T> &aDst) const
    {
        if constexpr (vector_size_v<T> == 1)
        {
            aDst = aSrc2.x;
        }
        else
        {
            aDst = aSrc2.Min();
        }
    }
};
template <AnyVector T> struct MaxScalar
{
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc) const
    {
        if constexpr (vector_size_v<T> == 1)
        {
            return aSrc.x;
        }
        else
        {
            return aSrc.Max();
        }
    }
    // for compatibility with StdDeviation (needed in MinMax kernel)
    DEVICE_CODE void operator()(T & /*aSrc1*/, const T &aSrc2, remove_vector_t<T> &aDst) const
    {
        if constexpr (vector_size_v<T> == 1)
        {
            aDst = aSrc2.x;
        }
        else
        {
            aDst = aSrc2.Max();
        }
    }
};

template <AnyVector T> struct Nothing
{
    DEVICE_CODE void operator()(T & /*aSrcDst*/) const
    {
    }
    // for compatibility with StdDeviation (needed in MinMax kernel)
    DEVICE_CODE void operator()(T & /*aSrc1*/, T &aSrc2, T &aDst) const
    {
        aDst = aSrc2;
    }
};

template <AnyVector T> struct NothingScalar
{
    // for compatibility with NormRel
    DEVICE_CODE remove_vector_t<T> operator()(T &aSrc1) const
    {
        return aSrc1.x;
    }
};

template <AnyVector T> struct SqrtPostOp
{
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Sqrt();
    }
};

template <AnyVector T> struct DivPostOp
{
    const remove_vector_t<T> Divisor;
    DEVICE_CODE DivPostOp(remove_vector_t<T> aDivisor) : Divisor(aDivisor)
    {
    }

    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst /= Divisor;
    }
};

template <AnyVector T> struct PSNR
{
    const remove_vector_t<T> Divisor; // pixel count for MSE
    const remove_vector_t<T> ValueRangeSqr;
    DEVICE_CODE PSNR(remove_vector_t<T> aDivisor, remove_vector_t<T> aValueRange)
        : Divisor(aDivisor), ValueRangeSqr(aValueRange * aValueRange)
    {
    }

    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst /= Divisor; // == MSE

        aSrcDst.DivInv(ValueRangeSqr);
        constexpr remove_vector_t<T> ten = static_cast<remove_vector_t<T>>(10);

        aSrcDst.x = ten * log10(aSrcDst.x);
        if constexpr (vector_active_size_v<T> > 1)
        {
            aSrcDst.y = ten * log10(aSrcDst.y);
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aSrcDst.z = ten * log10(aSrcDst.z);
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aSrcDst.w = ten * log10(aSrcDst.w);
        }
    }
};

template <AnyVector T> struct StdDeviation
{
    const remove_vector_t<T> Divisor;
    DEVICE_CODE StdDeviation(remove_vector_t<T> aDivisor) : Divisor(aDivisor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc, const T &aSrcSqr, T &aDst) const
    {
        // aSrc is supposed to be the sum of all elements
        // aSrcDstSqr is supposed to be the sum of all elements squared
        // StdDev is computed as sqrt((sum(elements^2) - sum(elements)^2 / elementCount) / (elementCount -  1))
        aDst = (aSrcSqr - (aSrc * aSrc) / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
        aDst.Sqrt();
    }

    DEVICE_CODE void operator()(const T &aSrc, const T &aSrcSqr, remove_vector_t<T> &aDst) const
    {
        // aSrc is supposed to be the sum of all elements
        // aSrcDstSqr is supposed to be the sum of all elements squared
        // StdDev is computed as sqrt((sum(elements^2) - sum(elements)^2 / elementCount) / (elementCount -  1))

        Vector1<remove_vector_t<T>> sum    = aSrc.x;
        Vector1<remove_vector_t<T>> sumSqr = aSrcSqr.x;

        if constexpr (vector_active_size_v<T> > 1)
        {
            sum.x += aSrc.y;
            sumSqr.x += aSrcSqr.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            sum.x += aSrc.z;
            sumSqr.x += aSrcSqr.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            sum.x += aSrc.w;
            sumSqr.x += aSrcSqr.w;
        }

        sum.x = (sumSqr.x - (sum.x * sum.x) / (Divisor * vector_active_size_v<T>)) /
                ((Divisor * vector_active_size_v<T>)-static_cast<remove_vector_t<T>>(1));
        sum.Sqrt();
        aDst = sum.x;
    }

    DEVICE_CODE void operator()(const same_vector_size_different_type_t<T, Complex<remove_vector_t<T>>> &aSrc,
                                const same_vector_size_different_type_t<T, Complex<remove_vector_t<T>>> &aSrcSqr,
                                T &aDst) const
        requires RealSignedVector<T>
    {
        // complex stdDev result in a real value T:
        // std(X) = sqrt(std(real(X))^2 + std(imag(X))^2 )

        // aSrc is supposed to be the sum of all elements
        // aSrcDstSqr is supposed to be the sum of all elements squared
        // StdDev is computed as sqrt((sum(elements^2) - sum(elements)^2 / elementCount) / (elementCount -  1))

        remove_vector_t<T> temp;
        temp   = aSrc.x.real * aSrc.x.real;
        temp   = (aSrcSqr.x.real - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
        aDst.x = temp; // temp is std(real part) squared
        temp   = aSrc.x.imag * aSrc.x.imag;
        temp   = (aSrcSqr.x.imag - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
        aDst.x += temp;

        if constexpr (vector_active_size_v<T> > 1)
        {
            temp   = aSrc.y.real * aSrc.y.real;
            temp   = (aSrcSqr.y.real - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
            aDst.y = temp;
            temp   = aSrc.y.imag * aSrc.y.imag;
            temp   = (aSrcSqr.y.imag - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
            aDst.y += temp;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            temp   = aSrc.z.real * aSrc.z.real;
            temp   = (aSrcSqr.z.real - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
            aDst.z = temp;
            temp   = aSrc.z.imag * aSrc.z.imag;
            temp   = (aSrcSqr.z.imag - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
            aDst.z += temp;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            temp   = aSrc.w.real * aSrc.w.real;
            temp   = (aSrcSqr.w.real - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
            aDst.w = temp;
            temp   = aSrc.w.imag * aSrc.w.imag;
            temp   = (aSrcSqr.w.imag - temp / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
            aDst.w += temp;
        }
        aDst.Sqrt();
    }

    DEVICE_CODE void operator()(const same_vector_size_different_type_t<T, Complex<remove_vector_t<T>>> &aSrc,
                                const same_vector_size_different_type_t<T, Complex<remove_vector_t<T>>> &aSrcSqr,
                                remove_vector_t<T> &aDst) const
        requires RealSignedVector<T>
    {
        // complex stdDev result in a real value T:
        // std(X) = sqrt(std(real(X))^2 + std(imag(X))^2 )

        // aSrc is supposed to be the sum of all elements
        // aSrcDstSqr is supposed to be the sum of all elements squared
        // StdDev is computed as sqrt((sum(elements^2) - sum(elements)^2 / elementCount) / (elementCount -  1))

        remove_vector_t<T> tempReal    = aSrc.x.real;
        remove_vector_t<T> tempImag    = aSrc.x.imag;
        remove_vector_t<T> tempSqrReal = aSrcSqr.x.real;
        remove_vector_t<T> tempSqrImag = aSrcSqr.x.imag;

        if constexpr (vector_active_size_v<T> > 1)
        {
            tempReal += aSrc.y.real;
            tempImag += aSrc.y.imag;
            tempSqrReal += aSrcSqr.y.real;
            tempSqrImag += aSrcSqr.y.imag;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            tempReal += aSrc.z.real;
            tempImag += aSrc.z.imag;
            tempSqrReal += aSrcSqr.z.real;
            tempSqrImag += aSrcSqr.z.imag;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            tempReal += aSrc.w.real;
            tempImag += aSrc.w.imag;
            tempSqrReal += aSrcSqr.w.real;
            tempSqrImag += aSrcSqr.w.imag;
        }

        Vector1<remove_vector_t<T>> temp;
        aDst = tempReal * tempReal;
        aDst = (tempSqrReal - aDst / (Divisor * vector_active_size_v<T>)) /
               ((Divisor * vector_active_size_v<T>)-static_cast<remove_vector_t<T>>(1));
        temp.x = aDst; // temp is std(real part) squared
        aDst   = tempImag * tempImag;
        aDst   = (tempSqrImag - aDst / (Divisor * vector_active_size_v<T>)) /
               ((Divisor * vector_active_size_v<T>)-static_cast<remove_vector_t<T>>(1));
        temp.x += aDst;

        temp.Sqrt();
        aDst = temp.x;
    }
};

template <AnyVector T> struct NormRelInfPost
{
    DEVICE_CODE void operator()(const T &aNormDiff, const T &aNormSrc2, T &aDst) const
    {
        aDst = aNormDiff / aNormSrc2;
    }
    DEVICE_CODE void operator()(const T &aNormDiff, const T &aNormSrc2, remove_vector_t<T> &aDst) const
    {
        if constexpr (vector_active_size_v<T> == 1)
        {
            aDst = aNormDiff.x / aNormSrc2.x;
        }
        else
        {
            aDst = aNormDiff.Max() / aNormSrc2.Max();
        }
    }
};

template <AnyVector T> struct NormRelL1Post
{
    DEVICE_CODE void operator()(const T &aNormDiff, const T &aNormSrc2, T &aDst) const
    {
        aDst = aNormDiff / aNormSrc2;
    }
    DEVICE_CODE void operator()(const T &aNormDiff, const T &aNormSrc2, remove_vector_t<T> &aDst) const
    {
        remove_vector_t<T> temp = aNormSrc2.x;
        aDst                    = aNormDiff.x;

        if constexpr (vector_active_size_v<T> > 1)
        {
            temp += aNormSrc2.y;
            aDst += aNormDiff.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            temp += aNormSrc2.z;
            aDst += aNormDiff.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            temp += aNormSrc2.w;
            aDst += aNormDiff.w;
        }
        aDst /= temp;
    }
};

template <AnyVector T> struct NormRelL2Post
{
    DEVICE_CODE void operator()(const T &aNormDiff, const T &aNormSrc2, T &aDst) const
    {
        aDst = T::Sqrt(aNormDiff) / T::Sqrt(aNormSrc2);
    }
    DEVICE_CODE void operator()(const T &aNormDiff, const T &aNormSrc2, remove_vector_t<T> &aDst) const
    {
        remove_vector_t<T> temp = aNormSrc2.x;
        aDst                    = aNormDiff.x;

        if constexpr (vector_active_size_v<T> > 1)
        {
            temp += aNormSrc2.y;
            aDst += aNormDiff.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            temp += aNormSrc2.z;
            aDst += aNormDiff.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            temp += aNormSrc2.w;
            aDst += aNormDiff.w;
        }
        aDst = sqrt(aDst) / sqrt(temp);
    }
};

template <AnyVector T> struct QualityIndex
{
    const remove_vector_t<T> PixelCount;
    const remove_vector_t<T> PixelCount_1;
    DEVICE_CODE QualityIndex(remove_vector_t<T> aPixelCount)
        : PixelCount(aPixelCount), PixelCount_1(aPixelCount - static_cast<remove_vector_t<T>>(1))
    {
    }

    DEVICE_CODE void operator()(const T &aSumSrc1, const T &aSumSrc1Sqr, const T &aSumSrc2, const T &aSumSrc2Sqr,
                                const T &aSumSrc1Src2, T &aDst) const
    {
        // formulas see https://ece.uwaterloo.ca/~z70wang/publications/quality_2c.pdf
        T meanSrc1 = aSumSrc1 / PixelCount;
        T meanSrc2 = aSumSrc2 / PixelCount;
        T varSrc1  = (aSumSrc1Sqr - (aSumSrc1 * aSumSrc1) / PixelCount) / (PixelCount_1);

        T varSrc2 = (aSumSrc2Sqr - (aSumSrc2 * aSumSrc2) / PixelCount) / (PixelCount_1);

        T crossVar = (aSumSrc1Src2 - PixelCount * meanSrc1 * meanSrc2) / PixelCount_1;

        aDst = (static_cast<remove_vector_t<T>>(4) * crossVar * meanSrc1 * meanSrc2) /
               ((varSrc1 + varSrc2) * (meanSrc1 * meanSrc1 + meanSrc2 * meanSrc2));
    }
};

template <AnyVector T> struct SSIM
{
    const remove_vector_t<T> PixelCount;
    const remove_vector_t<T> PixelCount_1;
    const remove_vector_t<T> C1;
    const remove_vector_t<T> C2;
    DEVICE_CODE SSIM(remove_vector_t<T> aPixelCount, remove_vector_t<T> aDynamicRange, remove_vector_t<T> aK1,
                     remove_vector_t<T> aK2)
        : PixelCount(aPixelCount), PixelCount_1(aPixelCount - static_cast<remove_vector_t<T>>(1)),
          C1(aK1 * aK1 * aDynamicRange * aDynamicRange), C2(aK2 * aK2 * aDynamicRange * aDynamicRange)
    {
    }

    DEVICE_CODE void operator()(const T &aSumSrc1, const T &aSumSrc1Sqr, const T &aSumSrc2, const T &aSumSrc2Sqr,
                                const T &aSumSrc1Src2, T &aDst) const
    {
        // formulas see https://en.wikipedia.org/wiki/Structural_similarity_index_measure

        T meanSrc1 = aSumSrc1 / PixelCount;
        T meanSrc2 = aSumSrc2 / PixelCount;
        T varSrc1  = (aSumSrc1Sqr - (aSumSrc1 * aSumSrc1) / PixelCount) / (PixelCount_1);

        T varSrc2 = (aSumSrc2Sqr - (aSumSrc2 * aSumSrc2) / PixelCount) / (PixelCount_1);

        T crossVar = (aSumSrc1Src2 - PixelCount * meanSrc1 * meanSrc2) / PixelCount_1;

        aDst = (static_cast<remove_vector_t<T>>(2) * meanSrc1 * meanSrc2 + C1) *
               (static_cast<remove_vector_t<T>>(2) * crossVar + C2) /
               ((meanSrc1 * meanSrc1 + meanSrc2 * meanSrc2 + C1) * (varSrc1 + varSrc2 + C2));
    }
};

} // namespace opp
