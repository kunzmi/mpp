#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from one src array -&gt; dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct SrcSingleChannelFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    const size_t SrcPitch1;
    const Channel SrcChannel;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    SrcSingleChannelFunctor()
    {
    }

    SrcSingleChannelFunctor(const SrcT *aSrc1, size_t aSrcPitch1, Channel aChannel, operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), SrcChannel(aChannel), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires std::same_as<ComputeT, DstT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const remove_vector_t<SrcT> *subPixelSrc1 =
            reinterpret_cast<const remove_vector_t<SrcT> *>(pixelSrc1) + SrcChannel.Value();

        Op(static_cast<ComputeT>(*subPixelSrc1), aDst);
        return true;
    }

    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires(!std::same_as<ComputeT, DstT>)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const remove_vector_t<SrcT> *subPixelSrc1 =
            reinterpret_cast<const remove_vector_t<SrcT> *>(pixelSrc1) + SrcChannel.Value();
        ComputeT temp;
        Op(static_cast<ComputeT>(*subPixelSrc1), temp);
        round(temp); // NOP for integer ComputeT
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires std::same_as<ComputeT, DstT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            const remove_vector_t<SrcT> *subPixelSrc1 =
                reinterpret_cast<const remove_vector_t<SrcT> *>(pixelSrc1 + i) + SrcChannel.Value();

            Op(static_cast<ComputeT>(*subPixelSrc1), aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires(!std::same_as<ComputeT, DstT>)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            const remove_vector_t<SrcT> *subPixelSrc1 =
                reinterpret_cast<const remove_vector_t<SrcT> *>(pixelSrc1 + i) + SrcChannel.Value();
            ComputeT temp;
            Op(static_cast<ComputeT>(*subPixelSrc1), temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
