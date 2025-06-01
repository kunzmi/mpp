#pragma once
#include "imageFunctors.h"
#include <common/defines.h>
#include <common/image/filterArea.h>
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
/// Specialized functor to compute StdDev from integral images.
/// </summary>
template <size_t tupelSize, typename Src1T, typename Src2T, typename ComputeT, typename DstT>
struct RectStdDevFunctor : public ImageFunctor<false>
{
    const Src1T *RESTRICT SrcSum;
    size_t SrcPitch1;

    const Src2T *RESTRICT SrcSumSqr;
    size_t SrcPitch2;

    Vector2<int> SizeRoiMinus1;
    FilterArea Area;
    remove_vector_t<ComputeT> AreaSize;
    remove_vector_t<ComputeT> AreaSizeSqrInv;

#pragma region Constructors
    RectStdDevFunctor()
    {
    }

    RectStdDevFunctor(const Src1T *aSrc1, size_t aSrcPitch1, const Src2T *aSrc2, size_t aSrcPitch2,
                      const Size2D &aSizeRoi, const FilterArea &aFilterArea)
        : SrcSum(aSrc1), SrcPitch1(aSrcPitch1), SrcSumSqr(aSrc2), SrcPitch2(aSrcPitch2), SizeRoiMinus1(aSizeRoi - 1),
          Area(aFilterArea), AreaSize(static_cast<remove_vector_t<ComputeT>>(Area.Size.x) *
                                      static_cast<remove_vector_t<ComputeT>>(Area.Size.y)),
          AreaSizeSqrInv(static_cast<remove_vector_t<ComputeT>>(1) /
                         (static_cast<remove_vector_t<ComputeT>>(Area.Size.x) *
                          static_cast<remove_vector_t<ComputeT>>(Area.Size.x) *
                          static_cast<remove_vector_t<ComputeT>>(Area.Size.y) *
                          static_cast<remove_vector_t<ComputeT>>(Area.Size.y)))
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    // Case ComputeT==DstT, no conversion, no rounding
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
    {
        Vector2<int> pMin = Vector2<int>(aPixelX, aPixelY) - Area.Center;
        Vector2<int> pMax = pMin + Vector2<int>(Area.Size);
        pMin.Max({0, 0});
        pMax.Min(SizeRoiMinus1);

        const Src1T *topLeftSum  = gotoPtr(SrcSum, SrcPitch1, pMin.x, pMin.y);
        const Src1T *topRightSum = gotoPtr(SrcSum, SrcPitch1, pMax.x, pMin.y);
        const Src1T *lowLeftSum  = gotoPtr(SrcSum, SrcPitch1, pMin.x, pMax.y);
        const Src1T *lowRightSum = gotoPtr(SrcSum, SrcPitch1, pMax.x, pMax.y);

        const Src2T *topLeftSumSqr  = gotoPtr(SrcSumSqr, SrcPitch2, pMin.x, pMin.y);
        const Src2T *topRightSumSqr = gotoPtr(SrcSumSqr, SrcPitch2, pMax.x, pMin.y);
        const Src2T *lowLeftSumSqr  = gotoPtr(SrcSumSqr, SrcPitch2, pMin.x, pMax.y);
        const Src2T *lowRightSumSqr = gotoPtr(SrcSumSqr, SrcPitch2, pMax.x, pMax.y);

        ComputeT sum = ComputeT(*topLeftSum) - ComputeT(*topRightSum) - ComputeT(*lowLeftSum) + ComputeT(*lowRightSum);
        ComputeT sumSqr =
            ComputeT(*topLeftSumSqr) - ComputeT(*topRightSumSqr) - ComputeT(*lowLeftSumSqr) + ComputeT(*lowRightSumSqr);

        sum.Sqr();
        sumSqr *= AreaSize;
        sumSqr -= sum;
        sumSqr *= AreaSizeSqrInv;
        sumSqr.Max({0});
        sumSqr.Sqrt();

        aDst = DstT(sumSqr);
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Vector2<int> pMin = Vector2<int>(aPixelX + int(i), aPixelY) - Area.Center;
            Vector2<int> pMax = pMin + Vector2<int>(Area.Size);
            pMin.Max({0, 0});
            pMax.Min(SizeRoiMinus1);

            const Src1T *topLeftSum  = gotoPtr(SrcSum, SrcPitch1, pMin.x, pMin.y);
            const Src1T *topRightSum = gotoPtr(SrcSum, SrcPitch1, pMax.x, pMin.y);
            const Src1T *lowLeftSum  = gotoPtr(SrcSum, SrcPitch1, pMin.x, pMax.y);
            const Src1T *lowRightSum = gotoPtr(SrcSum, SrcPitch1, pMax.x, pMax.y);

            const Src2T *topLeftSumSqr  = gotoPtr(SrcSumSqr, SrcPitch2, pMin.x, pMin.y);
            const Src2T *topRightSumSqr = gotoPtr(SrcSumSqr, SrcPitch2, pMax.x, pMin.y);
            const Src2T *lowLeftSumSqr  = gotoPtr(SrcSumSqr, SrcPitch2, pMin.x, pMax.y);
            const Src2T *lowRightSumSqr = gotoPtr(SrcSumSqr, SrcPitch2, pMax.x, pMax.y);

            ComputeT sum =
                ComputeT(*topLeftSum) - ComputeT(*topRightSum) - ComputeT(*lowLeftSum) + ComputeT(*lowRightSum);
            ComputeT sumSqr = ComputeT(*topLeftSumSqr) - ComputeT(*topRightSumSqr) - ComputeT(*lowLeftSumSqr) +
                              ComputeT(*lowRightSumSqr);

            sum.Sqr();
            sumSqr *= AreaSize;
            sumSqr -= sum;
            sumSqr *= AreaSizeSqrInv;
            sumSqr.Max({0});
            sumSqr.Sqrt();

            aDst.value[i] = DstT(sumSqr);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
