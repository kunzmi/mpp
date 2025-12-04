#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace mpp
{
// template <AnyVector T> struct Set
//{
//     DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
//     {
//         aDst = aSrc1;
//     }
// };

template <AnyVector T> struct Neg
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = -aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst = -aSrcDst;
    }
};

template <RealIntVector T> struct Not
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Not(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Not();
    }
};

template <AnyVector T> struct Sqr
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Sqr(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Sqr();
    }
};

template <AnyVector T> struct Sqrt
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Sqrt(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Sqrt();
    }
};

template <AnyVector T> struct Ln
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Ln(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Ln();
    }
};

template <AnyVector T> struct Exp
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Exp(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Exp();
    }
};

template <RealVector T> struct Abs
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Abs(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Abs();
    }
};

template <RealOrComplexFloatingVector T> struct Round
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Round(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Round();
    }
};

template <RealOrComplexFloatingVector T> struct Floor
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Floor(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Floor();
    }
};

template <RealOrComplexFloatingVector T> struct Ceil
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Ceil(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Ceil();
    }
};

template <RealOrComplexFloatingVector T> struct RoundNearest
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::RoundNearest(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.RoundNearest();
    }
};

template <RealOrComplexFloatingVector T> struct RoundZero
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::RoundZero(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.RoundZero();
    }
};

template <ComplexVector T> struct Magnitude
{
    DEVICE_CODE void operator()(
        const T &aSrc1, same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst) const
    {
        aDst = aSrc1.Magnitude();
    }
};

template <ComplexVector T> struct MagnitudeSqr
{
    DEVICE_CODE void operator()(
        const T &aSrc1, same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst) const
    {
        aDst = aSrc1.MagnitudeSqr();
    }
};

template <ComplexVector T> struct Conj
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst = T::Conj(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.Conj();
    }
};

template <ComplexVector T> struct Angle
{
    DEVICE_CODE void operator()(
        const T &aSrc1, same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst) const
    {
        aDst = aSrc1.Angle();
    }
};

template <ComplexVector T> struct Real
{
    DEVICE_CODE void operator()(
        const T &aSrc1, same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst) const
    {
        aDst.x = aSrc1.x.real;
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = aSrc1.y.real;
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = aSrc1.z.real;
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = aSrc1.w.real;
        }
    }
};

template <ComplexVector T> struct Imag
{
    DEVICE_CODE void operator()(
        const T &aSrc1, same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst) const
    {
        aDst.x = aSrc1.x.imag;
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = aSrc1.y.imag;
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = aSrc1.z.imag;
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = aSrc1.w.imag;
        }
    }
};

template <ComplexVector T> struct MakeComplex
{
    DEVICE_CODE void operator()(
        const same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aReal, T &aDst) const
    {
        aDst.x = remove_vector_t<T>(aReal.x);
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = remove_vector_t<T>(aReal.y);
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = remove_vector_t<T>(aReal.z);
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = remove_vector_t<T>(aReal.w);
        }
    }
    DEVICE_CODE void operator()(
        const same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aReal,
        const same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aImag, T &aDst) const
    {
        aDst.x = remove_vector_t<T>(aReal.x, aImag.x);
        if constexpr (vector_size_v<T> > 1)
        {
            aDst.y = remove_vector_t<T>(aReal.y, aImag.y);
        }
        if constexpr (vector_size_v<T> > 2)
        {
            aDst.z = remove_vector_t<T>(aReal.z, aImag.z);
        }
        if constexpr (vector_size_v<T> > 3)
        {
            aDst.w = remove_vector_t<T>(aReal.w, aImag.w);
        }
    }
};

template <RealVector T, RealVector SrcT> struct AlphaPremul
{
    const remove_vector_t<T> InvAlphaMax; // Inverse of the maximum value of the source or destination type

    AlphaPremul()
        : InvAlphaMax(static_cast<remove_vector_t<T>>(1) /
                      static_cast<remove_vector_t<T>>(numeric_limits<remove_vector_t<SrcT>>::max()))
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        const remove_vector_t<T> alpha = aSrc1.w * InvAlphaMax;

        aDst.x = aSrc1.x * alpha;
        aDst.y = aSrc1.y * alpha;
        aDst.z = aSrc1.z * alpha;
        aDst.w = aSrc1.w; // copy alpha value
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
        requires RealFloatingVector<SrcT>
    {
        const remove_vector_t<T> alpha = aSrc1.w;

        aDst.x = aSrc1.x * alpha;
        aDst.y = aSrc1.y * alpha;
        aDst.z = aSrc1.z * alpha;
        aDst.w = aSrc1.w; // copy alpha value
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        const remove_vector_t<T> alpha = aSrcDst.w * InvAlphaMax;

        aSrcDst.x *= alpha;
        aSrcDst.y *= alpha;
        aSrcDst.z *= alpha;
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
        requires RealFloatingVector<SrcT>
    {
        const remove_vector_t<T> alpha = aSrcDst.w;

        aSrcDst.x *= alpha;
        aSrcDst.y *= alpha;
        aSrcDst.z *= alpha;
    }
};

// special case of AlphaPremul with a constant alpha value that also sets the alpha channel to the used alpha value
template <RealVector T, RealVector SrcT> struct AlphaPremulAC
{
    // Alpha value scaled with inverse of the maximum value of the source or destination type for integer types:
    const remove_vector_t<T> AlphaValueDivMax;
    const remove_vector_t<T> AlphaValue;

    AlphaPremulAC(remove_vector_t<SrcT> aAlphaValue)
        : AlphaValueDivMax(static_cast<remove_vector_t<T>>(aAlphaValue) /
                           static_cast<remove_vector_t<T>>(numeric_limits<remove_vector_t<SrcT>>::max())),
          AlphaValue(static_cast<remove_vector_t<T>>(aAlphaValue))
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = aSrc1.x * AlphaValueDivMax;
        aDst.y = aSrc1.y * AlphaValueDivMax;
        aDst.z = aSrc1.z * AlphaValueDivMax;
        aDst.w = AlphaValue; // set alpha value
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
        requires RealFloatingVector<SrcT>
    {
        aDst.x = aSrc1.x * AlphaValue;
        aDst.y = aSrc1.y * AlphaValue;
        aDst.z = aSrc1.z * AlphaValue;
        aDst.w = AlphaValue; // set alpha value
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x *= AlphaValueDivMax;
        aSrcDst.y *= AlphaValueDivMax;
        aSrcDst.z *= AlphaValueDivMax;
        aSrcDst.w = AlphaValue; // set alpha value
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
        requires RealFloatingVector<SrcT>
    {
        aSrcDst.x *= AlphaValue;
        aSrcDst.y *= AlphaValue;
        aSrcDst.z *= AlphaValue;
        aSrcDst.w = AlphaValue; // set alpha value
    }
};

// due to the high number of constants, we put them all here in the operator and not in the functor...
template <RealVector T> struct MaxVal
{
    T Val;
    T Threshold;
    MaxVal(T aVal, T aThreshold) : Val(aVal), Threshold(aThreshold)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = aSrc1.x < Threshold.x ? Val.x : aSrc1.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            aDst.y = aSrc1.y < Threshold.y ? Val.y : aSrc1.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aDst.z = aSrc1.z < Threshold.z ? Val.z : aSrc1.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aDst.w = aSrc1.w < Threshold.w ? Val.w : aSrc1.w;
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = aSrcDst.x < Threshold.x ? Val.x : aSrcDst.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            aSrcDst.y = aSrcDst.y < Threshold.y ? Val.y : aSrcDst.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aSrcDst.z = aSrcDst.z < Threshold.z ? Val.z : aSrcDst.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aSrcDst.w = aSrcDst.w < Threshold.w ? Val.w : aSrcDst.w;
        }
    }
};

template <RealVector T> struct MinVal
{
    T Val;
    T Threshold;
    MinVal(T aVal, T aThreshold) : Val(aVal), Threshold(aThreshold)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = aSrc1.x > Threshold.x ? Val.x : aSrc1.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            aDst.y = aSrc1.y > Threshold.y ? Val.y : aSrc1.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aDst.z = aSrc1.z > Threshold.z ? Val.z : aSrc1.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aDst.w = aSrc1.w > Threshold.w ? Val.w : aSrc1.w;
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = aSrcDst.x > Threshold.x ? Val.x : aSrcDst.x;
        if constexpr (vector_active_size_v<T> > 1)
        {
            aSrcDst.y = aSrcDst.y > Threshold.y ? Val.y : aSrcDst.y;
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aSrcDst.z = aSrcDst.z > Threshold.z ? Val.z : aSrcDst.z;
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aSrcDst.w = aSrcDst.w > Threshold.w ? Val.w : aSrcDst.w;
        }
    }
};

template <RealVector T> struct MinValMaxVal
{
    T MinVal;
    T MinThreshold;
    T MaxVal;
    T MaxThreshold;
    MinValMaxVal(T aMinVal, T aMinThreshold, T aMaxVal, T aMaxThreshold)
        : MinVal(aMinVal), MinThreshold(aMinThreshold), MaxVal(aMaxVal), MaxThreshold(aMaxThreshold)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst) const
    {
        aDst.x = aSrc1.x > MinThreshold.x ? MinVal.x : (aSrc1.x < MaxThreshold.x ? MaxVal.x : aSrc1.x);
        if constexpr (vector_active_size_v<T> > 1)
        {
            aDst.y = aSrc1.y > MinThreshold.y ? MinVal.y : (aSrc1.y < MaxThreshold.y ? MaxVal.y : aSrc1.y);
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aDst.z = aSrc1.z > MinThreshold.z ? MinVal.z : (aSrc1.z < MaxThreshold.z ? MaxVal.z : aSrc1.z);
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aDst.w = aSrc1.w > MinThreshold.w ? MinVal.w : (aSrc1.w < MaxThreshold.w ? MaxVal.w : aSrc1.w);
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        aSrcDst.x = aSrcDst.x > MinThreshold.x ? MinVal.x : (aSrcDst.x < MaxThreshold.x ? MaxVal.x : aSrcDst.x);
        if constexpr (vector_active_size_v<T> > 1)
        {
            aSrcDst.y = aSrcDst.y > MinThreshold.y ? MinVal.y : (aSrcDst.y < MaxThreshold.y ? MaxVal.y : aSrcDst.y);
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aSrcDst.z = aSrcDst.z > MinThreshold.z ? MinVal.z : (aSrcDst.z < MaxThreshold.z ? MaxVal.z : aSrcDst.z);
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aSrcDst.w = aSrcDst.w > MinThreshold.w ? MinVal.w : (aSrcDst.w < MaxThreshold.w ? MaxVal.w : aSrcDst.w);
        }
    }
};

/// <summary>
/// For Integer type, IsScaleDown must be set to TRUE in case aScaleFactor &lt; 1.0
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="IsScaleDown"></typeparam>
template <AnyVector T, bool IsScaleDown = false, RoundingMode roundingMode = RoundingMode::NearestTiesToEven>
struct Scale
{
    complex_basetype_t<image::pixel_basetype_t<T>> ScaleVal;

    // aScaleFactor is a factor < 1 for downscaling and > 1 for upscaling
    Scale(double aScaleFactor)
    {
        if constexpr (RealOrComplexFloatingVector<T>)
        {
            // for float types we always scale by multiplication
            ScaleVal = static_cast<complex_basetype_t<image::pixel_basetype_t<T>>>(aScaleFactor);
        }
        else
        {
            if constexpr (IsScaleDown) // scale down
            {
                assert(aScaleFactor < 1.0);
                //  for integer types we scale down by division -> inverse of factor
                ScaleVal = static_cast<complex_basetype_t<image::pixel_basetype_t<T>>>(1.0 / aScaleFactor);
            }
            else
            {
                assert(aScaleFactor >= 1.0);
                ScaleVal = static_cast<complex_basetype_t<image::pixel_basetype_t<T>>>(aScaleFactor);
            }
        }
    }

    DEVICE_CODE void operator()(T &aSrcDst) const
    {
        if constexpr (RealOrComplexFloatingVector<T>)
        {
            aSrcDst *= ScaleVal;
        }
        else
        {
            if constexpr (IsScaleDown)
            {
                if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
                {
                    aSrcDst.DivScaleRound(ScaleVal);
                }
                else if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
                {
                    aSrcDst.DivScaleRoundNearest(ScaleVal);
                }
                else if constexpr (roundingMode == RoundingMode::TowardZero)
                {
                    aSrcDst.DivScaleRoundZero(ScaleVal);
                }
                else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
                {
                    aSrcDst.DivScaleFloor(ScaleVal);
                }
                else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
                {
                    aSrcDst.DivScaleCeil(ScaleVal);
                }
                else
                {
                    static_assert(roundingMode != RoundingMode::NearestTiesAwayFromZero, "Unsupported rounding mode.");
                }
            }
            else
            {
                aSrcDst *= ScaleVal;
            }
        }
    }
};
} // namespace mpp
