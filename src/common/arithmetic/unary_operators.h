#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/pixelTypes.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp
{
// template <AnyVector T> struct Set
//{
//     DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
//     {
//         aDst = aSrc1;
//     }
// };

template <AnyVector T> struct Neg
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = -aSrc1;
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst = -aSrcDst;
    }
};

template <RealIntVector T> struct Not
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Not(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Not();
    }
};

template <AnyVector T> struct Sqr
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Sqr(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Sqr();
    }
};

template <AnyVector T> struct Sqrt
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Sqrt(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Sqrt();
    }
};

template <AnyVector T> struct Ln
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Ln(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Ln();
    }
};

template <AnyVector T> struct Exp
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Exp(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Exp();
    }
};

template <RealVector T> struct Abs
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Abs(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Abs();
    }
};

template <RealOrComplexFloatingVector T> struct Round
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Round(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Round();
    }
};

template <RealOrComplexFloatingVector T> struct Floor
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Floor(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Floor();
    }
};

template <RealOrComplexFloatingVector T> struct Ceil
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Ceil(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Ceil();
    }
};

template <RealOrComplexFloatingVector T> struct RoundNearest
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::RoundNearest(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.RoundNearest();
    }
};

template <RealOrComplexFloatingVector T> struct RoundZero
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::RoundZero(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.RoundZero();
    }
};

template <ComplexVector T> struct Magnitude
{
    DEVICE_CODE void operator()(const T &aSrc1,
                                same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst)
    {
        aDst = aSrc1.Magnitude();
    }
};

template <ComplexVector T> struct MagnitudeSqr
{
    DEVICE_CODE void operator()(const T &aSrc1,
                                same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst)
    {
        aDst = aSrc1.MagnitudeSqr();
    }
};

template <ComplexVector T> struct Conj
{
    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst = T::Conj(aSrc1);
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.Conj();
    }
};

template <ComplexVector T> struct Angle
{
    DEVICE_CODE void operator()(const T &aSrc1,
                                same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst)
    {
        aDst = aSrc1.Angle();
    }
};

template <ComplexVector T> struct Real
{
    DEVICE_CODE void operator()(const T &aSrc1,
                                same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst)
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
    DEVICE_CODE void operator()(const T &aSrc1,
                                same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aDst)
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
        const same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aReal, T &aDst)
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
        const same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>> &aImag, T &aDst)
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

template <RealVector T> struct AlphaPremul
{
    const remove_vector_t<T> InvAlphaMax; // Inverse of the maximum value of the source or destination type

    AlphaPremul(remove_vector_t<T> aInvAlphaMax) : InvAlphaMax(aInvAlphaMax)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        const remove_vector_t<T> alpha = aSrc1.w * InvAlphaMax;

        aDst.x = aSrc1.x * alpha;
        aDst.y = aSrc1.y * alpha;
        aDst.z = aSrc1.z * alpha;
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
        requires RealFloatingVector<T>
    {
        const remove_vector_t<T> alpha = aSrc1.w;

        aDst.x = aSrc1.x * alpha;
        aDst.y = aSrc1.y * alpha;
        aDst.z = aSrc1.z * alpha;
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        const remove_vector_t<T> alpha = aSrcDst.w * InvAlphaMax;

        aSrcDst.x *= alpha;
        aSrcDst.y *= alpha;
        aSrcDst.z *= alpha;
    }
    DEVICE_CODE void operator()(T &aSrcDst)
        requires RealFloatingVector<T>
    {
        const remove_vector_t<T> alpha = aSrcDst.w;

        aSrcDst.x *= alpha;
        aSrcDst.y *= alpha;
        aSrcDst.z *= alpha;
    }
};

template <AnyVector SrcT, AnyVector DstT = SrcT> struct SwapChannel
{
    SwapChannel()
    {
        static_assert(AlwaysFalse<SrcT>::value, "SwapChannel is only implemented for 3 and 4 channel vector types.");
    }
};
template <AnyVector SrcDstT>
    requires opp::image::ThreeChannel<SrcDstT> || opp::image::FourChannelAlpha<SrcDstT>
struct SwapChannel<SrcDstT, SrcDstT>
{
    const opp::image::Channel DstOrder[3];

    SwapChannel(const opp::image::ChannelList<3> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2]}
    {
    }

    DEVICE_CODE void operator()(const SrcDstT &aSrc1, SrcDstT &aDst)
    {
        aDst.x = aSrc1[DstOrder[0]];
        aDst.y = aSrc1[DstOrder[1]];
        aDst.z = aSrc1[DstOrder[2]];
    }
    DEVICE_CODE void operator()(SrcDstT &aSrcDst)
    {
        SrcDstT temp = aSrcDst;
        aSrcDst.x    = temp[DstOrder[0]];
        aSrcDst.y    = temp[DstOrder[1]];
        aSrcDst.z    = temp[DstOrder[2]];
    }
};
template <AnyVector SrcDstT>
    requires opp::image::FourChannelNoAlpha<SrcDstT>
struct SwapChannel<SrcDstT, SrcDstT>
{
    const opp::image::Channel DstOrder[4];

    SwapChannel(const opp::image::ChannelList<4> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2], aChannel.data()[3]}
    {
    }

    DEVICE_CODE void operator()(const SrcDstT &aSrc1, SrcDstT &aDst)
    {
        aDst.x = aSrc1[DstOrder[0]];
        aDst.y = aSrc1[DstOrder[1]];
        aDst.z = aSrc1[DstOrder[2]];
        aDst.w = aSrc1[DstOrder[3]];
    }
    DEVICE_CODE void operator()(SrcDstT &aSrcDst)
    {
        SrcDstT temp = aSrcDst;
        aSrcDst.x    = temp[DstOrder[0]];
        aSrcDst.y    = temp[DstOrder[1]];
        aSrcDst.z    = temp[DstOrder[2]];
        aSrcDst.w    = temp[DstOrder[3]];
    }
};

template <AnyVector SrcT, AnyVector DstT>
    requires opp::image::ThreeChannel<SrcT> && opp::image::FourChannel<DstT>
struct SwapChannel<SrcT, DstT>
{
    const opp::image::Channel DstOrder[4];
    const remove_vector_t<DstT> mValue;

    SwapChannel(const opp::image::ChannelList<4> &aChannel, remove_vector_t<DstT> aValue)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2], aChannel.data()[3]}, mValue(aValue)
    {
    }

    // In case that one DstOrder-value is > 3, i.e. undefined we leave the destination pixel value untouched. This then
    // means that we have to load the value before hand from memory. Here the initial Dst-Value is unknown as aDst has
    // not been loaded from memory. Thus this operator() can only be called when all DstOrder-values are <= 3.
    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst)
    {
        if (DstOrder[0].Value() == 3)
        {
            aDst.x = mValue;
        }
        else
        {
            aDst.x = aSrc1[DstOrder[0]];
        }

        if (DstOrder[1].Value() == 3)
        {
            aDst.y = mValue;
        }
        else
        {
            aDst.y = aSrc1[DstOrder[1]];
        }

        if (DstOrder[2].Value() == 3)
        {
            aDst.z = mValue;
        }
        else
        {
            aDst.z = aSrc1[DstOrder[2]];
        }

        if (DstOrder[3].Value() == 3)
        {
            aDst.w = mValue;
        }
        else
        {
            aDst.w = aSrc1[DstOrder[3]];
        }
    }

    // In case that a DstOrder-value is > 3, i.e. undefined, we leave that destination pixel value untouched. This then
    // means that we have to load the value before hand from memory. To get this done, we use the Dst-array also as a
    // second source array and use a SrcSrcFunctor for the kernel.
    DEVICE_CODE void operator()(const SrcT &aSrc1, const DstT &aSrc2, DstT &aDst)
    {
        if (DstOrder[0].Value() <= 2)
        {
            aDst.x = aSrc1[DstOrder[0]];
        }
        else if (DstOrder[0].Value() == 3)
        {
            aDst.x = mValue;
        }
        else
        {
            aDst.x = aSrc2.x;
        }

        if (DstOrder[1].Value() <= 2)
        {
            aDst.y = aSrc1[DstOrder[1]];
        }
        else if (DstOrder[1].Value() == 3)
        {
            aDst.y = mValue;
        }
        else
        {
            aDst.y = aSrc2.y;
        }

        if (DstOrder[2].Value() <= 2)
        {
            aDst.z = aSrc1[DstOrder[2]];
        }
        else if (DstOrder[2].Value() == 3)
        {
            aDst.z = mValue;
        }
        else
        {
            aDst.z = aSrc2.z;
        }

        if (DstOrder[3].Value() <= 2)
        {
            aDst.w = aSrc1[DstOrder[3]];
        }
        else if (DstOrder[3].Value() == 3)
        {
            aDst.w = mValue;
        }
        else
        {
            aDst.w = aSrc2.w;
        }
    }
};

template <AnyVector SrcT, AnyVector DstT>
    requires opp::image::FourChannel<SrcT> && opp::image::ThreeChannel<DstT>
struct SwapChannel<SrcT, DstT>
{
    const opp::image::Channel DstOrder[3];

    SwapChannel(const opp::image::ChannelList<3> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2]}
    {
    }

    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst)
    {
        aDst.x = aSrc1[DstOrder[0]];
        aDst.y = aSrc1[DstOrder[1]];
        aDst.z = aSrc1[DstOrder[2]];
    }
};

template <AnyVector SrcT, AnyVector DstT = SrcT> struct Dup
{
    Dup()
    {
        static_assert(AlwaysFalse<SrcT>::value,
                      "Dup is only implemented from 1 channel to 3 or 4 channel vector types.");
    }
};
template <AnyVector SrcT, AnyVector DstT>
    requires opp::image::SingleChannel<SrcT> &&
             (opp::image::TwoChannel<DstT> || opp::image::ThreeChannel<DstT> || opp::image::FourChannel<DstT>)
struct Dup<SrcT, DstT>
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst)
    {
        aDst = DstT(aSrc1.x);
    }
};

template <AnyVector SrcT, AnyVector DstT = SrcT> struct Copy
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst)
    {
        aDst = aSrc1;
    }
};
template <AnyVector SrcT, AnyVector DstT>
    requires opp::image::SingleChannel<SrcT> &&
             (opp::image::TwoChannel<DstT> || opp::image::ThreeChannel<DstT> || opp::image::FourChannel<DstT>)
struct Copy<SrcT, DstT>
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst)
        requires opp::image::TwoChannel<DstT>
    {
        aDst = DstT(aSrc1.x, aSrc2.x);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, const SrcT &aSrc3, DstT &aDst)
        requires opp::image::ThreeChannel<DstT> || opp::image::FourChannelAlpha<DstT>
    {
        aDst = DstT(aSrc1.x, aSrc2.x, aSrc3.x);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, const SrcT &aSrc3, const SrcT &aSrc4, DstT &aDst)
        requires opp::image::FourChannelNoAlpha<DstT>
    {
        aDst = DstT(aSrc1.x, aSrc2.x, aSrc3.x, aSrc4.x);
    }
};

// due to the high number of constants, we put them all here in the operator and not in the functor...
template <RealVector T> struct MinVal
{
    T Val;
    T Threshold;
    MinVal(T aVal, T aThreshold) : Val(aVal), Threshold(aThreshold)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
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
    DEVICE_CODE void operator()(T &aSrcDst)
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

template <RealVector T> struct MaxVal
{
    T Val;
    T Threshold;
    MaxVal(T aVal, T aThreshold) : Val(aVal), Threshold(aThreshold)
    {
    }

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
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
    DEVICE_CODE void operator()(T &aSrcDst)
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

    DEVICE_CODE void operator()(const T &aSrc1, T &aDst)
    {
        aDst.x = aSrc1.x < MinThreshold.x ? MinVal.x : (aSrc1.x > MaxThreshold.x ? MaxVal.x : aSrc1.x);
        if constexpr (vector_active_size_v<T> > 1)
        {
            aDst.y = aSrc1.y < MinThreshold.y ? MinVal.y : (aSrc1.y > MaxThreshold.y ? MaxVal.y : aSrc1.y);
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aDst.z = aSrc1.z < MinThreshold.z ? MinVal.z : (aSrc1.z > MaxThreshold.z ? MaxVal.z : aSrc1.z);
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aDst.w = aSrc1.w < MinThreshold.w ? MinVal.w : (aSrc1.w > MaxThreshold.w ? MaxVal.w : aSrc1.w);
        }
    }
    DEVICE_CODE void operator()(T &aSrcDst)
    {
        aSrcDst.x = aSrcDst.x < MinThreshold.x ? MinVal.x : (aSrcDst.x > MaxThreshold.x ? MaxVal.x : aSrcDst.x);
        if constexpr (vector_active_size_v<T> > 1)
        {
            aSrcDst.y = aSrcDst.y < MinThreshold.y ? MinVal.y : (aSrcDst.y > MaxThreshold.y ? MaxVal.y : aSrcDst.y);
        }
        if constexpr (vector_active_size_v<T> > 2)
        {
            aSrcDst.z = aSrcDst.z < MinThreshold.z ? MinVal.z : (aSrcDst.z > MaxThreshold.z ? MaxVal.z : aSrcDst.z);
        }
        if constexpr (vector_active_size_v<T> > 3)
        {
            aSrcDst.w = aSrcDst.w < MinThreshold.w ? MinVal.w : (aSrcDst.w > MaxThreshold.w ? MaxVal.w : aSrcDst.w);
        }
    }
};
} // namespace opp
