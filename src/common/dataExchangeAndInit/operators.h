#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/mpp_defs.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace mpp
{
template <AnyVector TFrom, AnyVector TTo> struct Convert
{
    // not const TFrom as we want to clamp value range inplace in aSrc1
    DEVICE_CODE void operator()(TFrom &aSrc1, TTo &aDst) const
    {
        aDst = static_cast<TTo>(aSrc1);
    }
};

template <AnyVector TFrom, AnyVector TTo> struct ConvertRound
{
    RoundingMode mRoundingMode;

    ConvertRound(RoundingMode aRoundingMode) : mRoundingMode(aRoundingMode)
    {
    }

    DEVICE_CODE void operator()(const TFrom &aSrc1, TTo &aDst) const
    {
        aDst = TTo(aSrc1, mRoundingMode);
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
    requires mpp::image::ThreeChannel<SrcDstT> || mpp::image::FourChannelAlpha<SrcDstT>
struct SwapChannel<SrcDstT, SrcDstT>
{
    const mpp::image::Channel DstOrder[3];

    SwapChannel(const mpp::image::ChannelList<3> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2]}
    {
    }

    DEVICE_CODE void operator()(const SrcDstT &aSrc1, SrcDstT &aDst) const
    {
        aDst.x = aSrc1[DstOrder[0]];
        aDst.y = aSrc1[DstOrder[1]];
        aDst.z = aSrc1[DstOrder[2]];
    }
    DEVICE_CODE void operator()(SrcDstT &aSrcDst) const
    {
        SrcDstT temp = aSrcDst;
        aSrcDst.x    = temp[DstOrder[0]];
        aSrcDst.y    = temp[DstOrder[1]];
        aSrcDst.z    = temp[DstOrder[2]];
    }
};
template <AnyVector SrcDstT>
    requires mpp::image::FourChannelNoAlpha<SrcDstT>
struct SwapChannel<SrcDstT, SrcDstT>
{
    const mpp::image::Channel DstOrder[4];

    SwapChannel(const mpp::image::ChannelList<4> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2], aChannel.data()[3]}
    {
    }

    DEVICE_CODE void operator()(const SrcDstT &aSrc1, SrcDstT &aDst) const
    {
        aDst.x = aSrc1[DstOrder[0]];
        aDst.y = aSrc1[DstOrder[1]];
        aDst.z = aSrc1[DstOrder[2]];
        aDst.w = aSrc1[DstOrder[3]];
    }
    DEVICE_CODE void operator()(SrcDstT &aSrcDst) const
    {
        SrcDstT temp = aSrcDst;
        aSrcDst.x    = temp[DstOrder[0]];
        aSrcDst.y    = temp[DstOrder[1]];
        aSrcDst.z    = temp[DstOrder[2]];
        aSrcDst.w    = temp[DstOrder[3]];
    }
};

template <AnyVector SrcT, AnyVector DstT>
    requires mpp::image::ThreeChannel<SrcT> && mpp::image::FourChannel<DstT>
struct SwapChannel<SrcT, DstT>
{
    const mpp::image::Channel DstOrder[4];
    const remove_vector_t<DstT> mValue;

    SwapChannel(const mpp::image::ChannelList<4> &aChannel, remove_vector_t<DstT> aValue)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2], aChannel.data()[3]}, mValue(aValue)
    {
    }

    SwapChannel(const mpp::image::ChannelList<4> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2], aChannel.data()[3]}, mValue(0)
    {
    }

    // In case that one DstOrder-value is > 3, i.e. undefined we leave the destination pixel value untouched. This then
    // means that we have to load the value before hand from memory. Here the initial Dst-Value is unknown as aDst has
    // not been loaded from memory. Thus this operator() can only be called when all DstOrder-values are <= 3.
    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst) const
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
    DEVICE_CODE void operator()(const SrcT &aSrc1, const DstT &aSrc2, DstT &aDst) const
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
    requires mpp::image::FourChannel<SrcT> && mpp::image::ThreeChannel<DstT>
struct SwapChannel<SrcT, DstT>
{
    const mpp::image::Channel DstOrder[3];

    SwapChannel(const mpp::image::ChannelList<3> &aChannel)
        : DstOrder{aChannel.data()[0], aChannel.data()[1], aChannel.data()[2]}
    {
    }

    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst) const
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
                      "Dup is only implemented from 1 channel to multi-channel vector types.");
    }
};
template <AnyVector SrcT, AnyVector DstT>
    requires mpp::image::SingleChannel<SrcT> &&
             (mpp::image::TwoChannel<DstT> || mpp::image::ThreeChannel<DstT> || mpp::image::FourChannel<DstT>)
struct Dup<SrcT, DstT>
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst) const
    {
        aDst = DstT(aSrc1.x);
    }
};

template <AnyVector SrcT, AnyVector DstT = SrcT> struct Copy
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, DstT &aDst) const
    {
        aDst = aSrc1;
    }
};
template <AnyVector SrcT, AnyVector DstT>
    requires mpp::image::SingleChannel<SrcT> &&
             (mpp::image::TwoChannel<DstT> || mpp::image::ThreeChannel<DstT> || mpp::image::FourChannel<DstT>)
struct Copy<SrcT, DstT>
{
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, DstT &aDst) const
        requires mpp::image::TwoChannel<DstT>
    {
        aDst = DstT(aSrc1.x, aSrc2.x);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, const SrcT &aSrc3, DstT &aDst) const
        requires mpp::image::ThreeChannel<DstT> || mpp::image::FourChannelAlpha<DstT>
    {
        aDst = DstT(aSrc1.x, aSrc2.x, aSrc3.x);
    }
    DEVICE_CODE void operator()(const SrcT &aSrc1, const SrcT &aSrc2, const SrcT &aSrc3, const SrcT &aSrc4,
                                DstT &aDst) const
        requires mpp::image::FourChannelNoAlpha<DstT>
    {
        aDst = DstT(aSrc1.x, aSrc2.x, aSrc3.x, aSrc4.x);
    }
};
} // namespace mpp
