#pragma once
#include "borderControl.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/vector2.h>
#ifdef IS_HOST_COMPILER
#include <utility>
#endif

namespace mpp::image
{

// Same as normal borderControl but only acting on the horizontal / x-component
template <typename PixelT, BorderType borderType, bool onlyForInterpolation = false, bool iterative = false,
          bool avoidBranching = false, bool planar = false>
struct BorderControlHorizontal
{
    // indicates if border control is supposed to only handle out of ROI pixels for interpolation or also for accessing
    // out of ROI pixels in general (e.g. geometric transforms):
    static constexpr bool only_for_interpolation = onlyForInterpolation;
    static constexpr BorderType border_type      = borderType;

    // pointing to first pixel in allowedAccessRoi
    const BorderControlDataSource<PixelT, planar> DataSource;

    const int LastValidPixel; // == SizeAllowedAccessRoi - 1, to reduce computations...
    const int SizeAllowedAccessRoi;
    const int OffsetToActualRoi;

    const ConstantDataSource<PixelT, borderType> ConstantPixel;

#pragma region Constructors
    BorderControlHorizontal(const PixelT *aSrcRoiBasePointer, size_t aSrcPitch, int aSizeAllowedAccessRoi,
                            int aOffsetToActualRoi, const PixelT &aConstantPixel = PixelT(0))
        requires(planar == false)
        : DataSource({aSrcRoiBasePointer, aSrcPitch}), LastValidPixel(aSizeAllowedAccessRoi - 1),
          SizeAllowedAccessRoi(aSizeAllowedAccessRoi), OffsetToActualRoi(aOffsetToActualRoi),
          ConstantPixel({aConstantPixel})
    {
    }

    BorderControlHorizontal(int aSizeAllowedAccessRoi, int aOffsetToActualRoi)
        requires(planar == false)
        : DataSource({}), LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel(PixelT(0))
    {
    }

    BorderControlHorizontal(const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer0, size_t aSrcPitch0,
                            const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer1, size_t aSrcPitch1,
                            int aSizeAllowedAccessRoi, int aOffsetToActualRoi, const PixelT &aConstantPixel = PixelT(0))
        requires(planar == true) && (vector_active_size_v<PixelT> == 2)
        : DataSource({aSrcRoiBasePointer0, aSrcPitch0, aSrcRoiBasePointer1, aSrcPitch1}),
          LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel({aConstantPixel})
    {
    }

    BorderControlHorizontal(const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer0, size_t aSrcPitch0,
                            const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer1, size_t aSrcPitch1,
                            const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer2, size_t aSrcPitch2,
                            int aSizeAllowedAccessRoi, int aOffsetToActualRoi, const PixelT &aConstantPixel = PixelT(0))
        requires(planar == true) && (vector_active_size_v<PixelT> == 3)
        : DataSource(
              {aSrcRoiBasePointer0, aSrcPitch0, aSrcRoiBasePointer1, aSrcPitch1, aSrcRoiBasePointer2, aSrcPitch2}),
          LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel({aConstantPixel})
    {
    }

    BorderControlHorizontal(const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer0, size_t aSrcPitch0,
                            const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer1, size_t aSrcPitch1,
                            const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer2, size_t aSrcPitch2,
                            const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer3, size_t aSrcPitch3,
                            int aSizeAllowedAccessRoi, int aOffsetToActualRoi, const PixelT &aConstantPixel = PixelT(0))
        requires(planar == true) && (vector_active_size_v<PixelT> == 4)
        : DataSource({aSrcRoiBasePointer0, aSrcPitch0, aSrcRoiBasePointer1, aSrcPitch1, aSrcRoiBasePointer2, aSrcPitch2,
                      aSrcRoiBasePointer3, aSrcPitch3}),
          LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel({aConstantPixel})
    {
    }
#pragma endregion

    /// <summary>
    /// Takes as input the coordinates relative to the active ROI. Returns in the same variables, the coordinates
    /// adjusted to the allowed-access-ROI and the corresponding border handling.
    /// </summary>
    DEVICE_CODE void AdjustCoordinates(int &aPixelX, const int &aPixelY) const
    {
        // take into account the offset from actual ROI to data base pointer:
        aPixelX += OffsetToActualRoi;

        if constexpr (borderType == BorderType::None)
        {
            // nothing to do...
        }
        else if constexpr (borderType == BorderType::Constant)
        {
            // nothing to do...
        }
        else if constexpr (borderType == BorderType::Replicate || borderType == BorderType::SmoothEdge)
        {
#ifdef IS_HOST_COMPILER
            aPixelX = std::max(std::min(aPixelX, LastValidPixel), 0);
#else
            aPixelX = max(min(aPixelX, LastValidPixel), 0);
#endif
        }
        else if constexpr (borderType == BorderType::Mirror)
        {
            if constexpr (iterative)
            {
                while (aPixelX < 0 || aPixelX > LastValidPixel)
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX;
                    }
                    else if (aPixelX > LastValidPixel)
                    {
                        aPixelX = 2 * LastValidPixel - aPixelX;
                    }
                }
            }
            else
            {
                if constexpr (avoidBranching)
                {
                    int isSmallerAsMin = aPixelX < 0 ? -1 : 0;
                    int isLargerAsMax  = aPixelX > LastValidPixel ? 1 : 0;
                    int isInTheMiddle  = isSmallerAsMin == isLargerAsMax ? 1 : 0;

                    aPixelX = isSmallerAsMin * aPixelX + isLargerAsMax * (2 * LastValidPixel - aPixelX) +
                              isInTheMiddle * aPixelX;
                }
                else
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX;
                    }
                    else if (aPixelX > LastValidPixel)
                    {
                        aPixelX = 2 * LastValidPixel - aPixelX;
                    }
                }
            }
        }
        else if constexpr (borderType == BorderType::MirrorReplicate)
        {
            if constexpr (iterative)
            {
                while (aPixelX < 0 || aPixelX > LastValidPixel)
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX - 1;
                    }
                    else if (aPixelX > LastValidPixel)
                    {
                        aPixelX = 2 * LastValidPixel - aPixelX + 1;
                    }
                }
            }
            else
            {
                if constexpr (avoidBranching)
                {
                    int isSmallerAsMin = aPixelX < 0 ? -1 : 0;
                    int isLargerAsMax  = aPixelX > LastValidPixel ? 1 : 0;
                    int isInTheMiddle  = isSmallerAsMin == isLargerAsMax ? 1 : 0;

                    aPixelX = isSmallerAsMin * (aPixelX + 1) + isLargerAsMax * (2 * LastValidPixel - aPixelX + 1) +
                              isInTheMiddle * aPixelX;
                }
                else
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX - 1;
                    }
                    else if (aPixelX > LastValidPixel)
                    {
                        aPixelX = 2 * LastValidPixel - aPixelX + 1;
                    }
                }
            }
        }
        else if constexpr (borderType == BorderType::Wrap)
        {
            if constexpr (iterative)
            {
                while (aPixelX < 0)
                {
                    aPixelX += SizeAllowedAccessRoi;
                }
                while (aPixelX > LastValidPixel)
                {
                    aPixelX -= SizeAllowedAccessRoi;
                }
            }
            else
            {
                if (aPixelX < 0)
                {
                    aPixelX += SizeAllowedAccessRoi;
                }
                else if (aPixelX > LastValidPixel)
                {
                    aPixelX -= SizeAllowedAccessRoi;
                }
            }
        }
    }

    DEVICE_CODE PixelT operator()(int aPixelX, const int aPixelY) const
    {
        AdjustCoordinates(aPixelX, aPixelY);

        if constexpr (borderType == BorderType::Constant)
        {
            if (aPixelX < 0 || aPixelX > LastValidPixel)
            {
                return ConstantPixel.Value;
            }
        }

        if constexpr (planar)
        {
            PixelT ret;
            ret.x = (*gotoPtr(DataSource.SrcRoiBasePointer0, DataSource.SrcPitch0, aPixelX, aPixelY)).x;
            if constexpr (vector_active_size_v<PixelT> > 1)
            {
                ret.y = (*gotoPtr(DataSource.SrcRoiBasePointer1, DataSource.SrcPitch1, aPixelX, aPixelY)).x;
            }
            if constexpr (vector_active_size_v<PixelT> > 2)
            {
                ret.z = (*gotoPtr(DataSource.SrcRoiBasePointer2, DataSource.SrcPitch2, aPixelX, aPixelY)).x;
            }
            if constexpr (vector_active_size_v<PixelT> > 3)
            {
                ret.w = (*gotoPtr(DataSource.SrcRoiBasePointer3, DataSource.SrcPitch3, aPixelX, aPixelY)).x;
            }
            return ret;
        }
        else
        {
            return *gotoPtr(DataSource.SrcRoiBasePointer, DataSource.SrcPitch, aPixelX, aPixelY);
        }
    }
};
} // namespace mpp::image