#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/opp_defs.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp
{
template <typename DstT> struct Nothing
{
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst) const
    {
    }
};
template <typename SrcDstT, typename ComputeT> struct TopHat
{
    const SrcDstT *RESTRICT Src;
    const size_t SrcPitch;

    TopHat(const SrcDstT *aSrc, size_t aSrcPitch) : Src(aSrc), SrcPitch(aSrcPitch)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcDstT &aDst) const
    {
        const SrcDstT *pixelSrc = opp::image::gotoPtr(Src, SrcPitch, aPixelX, aPixelY);

        ComputeT srcPixel = ComputeT(*pixelSrc);
        ComputeT dstPixel = ComputeT(aDst);
        aDst              = SrcDstT(srcPixel - dstPixel);
    }
};
template <typename SrcDstT, typename ComputeT> struct BlackHat
{
    const SrcDstT *RESTRICT Src;
    const size_t SrcPitch;

    BlackHat(const SrcDstT *aSrc, size_t aSrcPitch) : Src(aSrc), SrcPitch(aSrcPitch)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcDstT &aDst) const
    {
        const SrcDstT *pixelSrc = opp::image::gotoPtr(Src, SrcPitch, aPixelX, aPixelY);

        ComputeT srcPixel = ComputeT(*pixelSrc);
        ComputeT dstPixel = ComputeT(aDst);
        aDst              = SrcDstT(dstPixel - srcPixel);
    }
};

} // namespace opp
