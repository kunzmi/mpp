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
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc)
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
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc)
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
template <AnyVector T> struct MeanScalar
{
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc)
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
        return res /= static_cast<remove_vector_t<T>>(vector_active_size_v<T>);
    }
};
template <AnyVector T> struct MinScalar
{
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc)
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
    DEVICE_CODE void operator()(T & /*aSrc1*/, const T &aSrc2, remove_vector_t<T> &aDst)
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
    DEVICE_CODE remove_vector_t<T> operator()(const T &aSrc)
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
    DEVICE_CODE void operator()(T & /*aSrc1*/, const T &aSrc2, remove_vector_t<T> &aDst)
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
    DEVICE_CODE void operator()(T & /*aSrcDst*/)
    {
    }
    // for compatibility with StdDeviation (needed in MinMax kernel)
    DEVICE_CODE void operator()(T & /*aSrc1*/, T &aSrc2, T &aDst)
    {
        aDst = aSrc2;
    }
};

template <AnyVector T> struct Sqrt
{
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Sqrt();
    }
};

template <AnyVector T> struct Div
{
    remove_vector_t<T> Divisor;
    Div(remove_vector_t<T> aDivisor) : Divisor(aDivisor)
    {
    }

    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst /= Divisor;
    }
};

template <AnyVector T> struct StdDeviation
{
    remove_vector_t<T> Divisor;
    StdDeviation(remove_vector_t<T> aDivisor) : Divisor(aDivisor)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc, const T &aSrcSqr, T &aDst)
    {
        // aSrc is supposed to be the sum of all elements
        // aSrcDstSqr is supposed to be the sum of all elements squared
        // StdDev is computed as sqrt((sum(elements^2) - sum(elements)^2 / elementCount) / (elementCount -  1))
        aDst = (aSrcSqr - (aSrc * aSrc) / Divisor) / (Divisor - static_cast<remove_vector_t<T>>(1));
        aDst.Sqrt();
    }

    DEVICE_CODE void operator()(const same_vector_size_different_type_t<T, Complex<remove_vector_t<T>>> &aSrc,
                                const same_vector_size_different_type_t<T, Complex<remove_vector_t<T>>> &aSrcSqr,
                                T &aDst)
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
                                remove_vector_t<T> &aDst)
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

} // namespace opp
