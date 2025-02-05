#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from one src array and dst array -&gt; dst pixel (mainly for swap channel with invalid
/// channel IDs)
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename DstT, typename operation>
struct SrcDstAsSrcFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    const DstT *RESTRICT DstAsSrc; // The RESTRICT here is not entirely legal and only valid under the assumption that
                                   // the destination pixel is only read before it gets updated.
    size_t DstAsSrcPitch;

    [[no_unique_address]] operation Op;

#pragma region Constructors
    SrcDstAsSrcFunctor()
    {
    }

    SrcDstAsSrcFunctor(const SrcT *aSrc1, size_t aSrcPitch1, const DstT *aDstAsSrc, size_t aDstAsSrcPitch,
                       operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), DstAsSrc(aDstAsSrc), DstAsSrcPitch(aDstAsSrcPitch), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
    {
        const SrcT *pixelSrc1     = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const DstT *pixelDstAsSrc = gotoPtr(DstAsSrc, DstAsSrcPitch, aPixelX, aPixelY);
        Op(*pixelSrc1, *pixelDstAsSrc, aDst);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
    {
        const SrcT *pixelSrc1     = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const DstT *pixelDstAsSrc = gotoPtr(DstAsSrc, DstAsSrcPitch, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1     = Tupel<SrcT, tupelSize>::Load(pixelSrc1);
        Tupel<DstT, tupelSize> tupelDstAsSrc = Tupel<SrcT, tupelSize>::Load(pixelDstAsSrc);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(tupelSrc1.value[i], tupelDstAsSrc.value[i], aDst.value[i]);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
