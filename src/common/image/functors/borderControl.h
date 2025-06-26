#pragma once
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

template <typename PixelT, BorderType borderType> struct ConstantDataSource
{
    ConstantDataSource()
    {
    }

    ConstantDataSource(const PixelT & /*aValue*/)
    {
        // dummy constructor for simplicity
    }
};

template <typename PixelT> struct ConstantDataSource<PixelT, BorderType::Constant>
{
    const PixelT Value;
};

template <typename PixelT, bool planar> struct BorderControlDataSource
{
    const PixelT *RESTRICT SrcRoiBasePointer;
    const size_t SrcPitch;
};

template <typename PixelT>
    requires(vector_active_size_v<PixelT> == 2)
struct BorderControlDataSource<PixelT, true>
{
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer0;
    const size_t SrcPitch0;
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer1;
    const size_t SrcPitch1;
};

template <typename PixelT>
    requires(vector_active_size_v<PixelT> == 3)
struct BorderControlDataSource<PixelT, true>
{
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer0;
    const size_t SrcPitch0;
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer1;
    const size_t SrcPitch1;
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer2;
    const size_t SrcPitch2;
};

template <typename PixelT>
    requires(vector_active_size_v<PixelT> == 4)
struct BorderControlDataSource<PixelT, true>
{
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer0;
    const size_t SrcPitch0;
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer1;
    const size_t SrcPitch1;
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer2;
    const size_t SrcPitch2;
    const Vector1<remove_vector_t<PixelT>> *RESTRICT SrcRoiBasePointer3;
    const size_t SrcPitch3;
};

template <typename PixelT, BorderType borderType, bool onlyForInterpolation = false, bool iterative = false,
          bool avoidBranching = false, bool planar = false>
struct BorderControl
{
    // indicates if border control is supposed to only handle out of ROI pixels for interpolation or also for accessing
    // out of ROI pixels in general (e.g. geometric transforms):
    static constexpr bool only_for_interpolation = onlyForInterpolation;

    // pointing to first pixel in allowedAccessRoi
    const BorderControlDataSource<PixelT, planar> DataSource;

    const Vector2<int> LastValidPixel; // == SizeAllowedAccessRoi - 1, to reduce computations...
    const Vector2<int> SizeAllowedAccessRoi;
    const Vector2<int> OffsetToActualRoi;

    const ConstantDataSource<PixelT, borderType> ConstantPixel;

#pragma region Constructors
    BorderControl(const PixelT *aSrcRoiBasePointer, size_t aSrcPitch, const Size2D &aSizeAllowedAccessRoi,
                  const Vector2<int> &aOffsetToActualRoi, const PixelT &aConstantPixel = PixelT(0))
        requires(planar == false)
        : DataSource({aSrcRoiBasePointer, aSrcPitch}), LastValidPixel(aSizeAllowedAccessRoi - 1),
          SizeAllowedAccessRoi(aSizeAllowedAccessRoi), OffsetToActualRoi(aOffsetToActualRoi),
          ConstantPixel({aConstantPixel})
    {
    }

    BorderControl(const Size2D &aSizeAllowedAccessRoi, const Vector2<int> &aOffsetToActualRoi)
        requires(planar == false)
        : DataSource({}), LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel(PixelT(0))
    {
    }

    BorderControl(const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer0, size_t aSrcPitch0,
                  const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer1, size_t aSrcPitch1,
                  const Size2D &aSizeAllowedAccessRoi, const Vector2<int> &aOffsetToActualRoi,
                  const PixelT &aConstantPixel = PixelT(0))
        requires(planar == true) && (vector_active_size_v<PixelT> == 2)
        : DataSource({aSrcRoiBasePointer0, aSrcPitch0, aSrcRoiBasePointer1, aSrcPitch1}),
          LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel({aConstantPixel})
    {
    }

    BorderControl(const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer0, size_t aSrcPitch0,
                  const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer1, size_t aSrcPitch1,
                  const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer2, size_t aSrcPitch2,
                  const Size2D &aSizeAllowedAccessRoi, const Vector2<int> &aOffsetToActualRoi,
                  const PixelT &aConstantPixel = PixelT(0))
        requires(planar == true) && (vector_active_size_v<PixelT> == 3)
        : DataSource(
              {aSrcRoiBasePointer0, aSrcPitch0, aSrcRoiBasePointer1, aSrcPitch1, aSrcRoiBasePointer2, aSrcPitch2}),
          LastValidPixel(aSizeAllowedAccessRoi - 1), SizeAllowedAccessRoi(aSizeAllowedAccessRoi),
          OffsetToActualRoi(aOffsetToActualRoi), ConstantPixel({aConstantPixel})
    {
    }

    BorderControl(const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer0, size_t aSrcPitch0,
                  const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer1, size_t aSrcPitch1,
                  const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer2, size_t aSrcPitch2,
                  const Vector1<remove_vector_t<PixelT>> *aSrcRoiBasePointer3, size_t aSrcPitch3,
                  const Size2D &aSizeAllowedAccessRoi, const Vector2<int> &aOffsetToActualRoi,
                  const PixelT &aConstantPixel = PixelT(0))
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
    DEVICE_CODE void AdjustCoordinates(int &aPixelX, int &aPixelY) const
    {
        // take into account the offset from actual ROI to data base pointer:
        aPixelX += OffsetToActualRoi.x;
        aPixelY += OffsetToActualRoi.y;

        if constexpr (borderType == BorderType::None)
        {
            // nothing to do...
        }
        else if constexpr (borderType == BorderType::Constant)
        {
            // nothing to do...
        }
        else if constexpr (borderType == BorderType::Replicate)
        {
#ifdef IS_HOST_COMPILER
            aPixelX = std::max(std::min(aPixelX, LastValidPixel.x), 0);
            aPixelY = std::max(std::min(aPixelY, LastValidPixel.y), 0);
#else
            aPixelX = max(min(aPixelX, LastValidPixel.x), 0);
            aPixelY = max(min(aPixelY, LastValidPixel.y), 0);
#endif
        }
        else if constexpr (borderType == BorderType::Mirror)
        {
            if constexpr (iterative)
            {
                while (aPixelX < 0 || aPixelX > LastValidPixel.x)
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX;
                    }
                    else if (aPixelX > LastValidPixel.x)
                    {
                        aPixelX = 2 * LastValidPixel.x - aPixelX;
                    }
                }

                while (aPixelY < 0 || aPixelY > LastValidPixel.y)
                {
                    if (aPixelY < 0)
                    {
                        aPixelY = -aPixelY;
                    }
                    else if (aPixelY > LastValidPixel.y)
                    {
                        aPixelY = 2 * LastValidPixel.y - aPixelY;
                    }
                }
            }
            else
            {
                if constexpr (avoidBranching)
                {
                    int isSmallerAsMin = aPixelX < 0 ? -1 : 0;
                    int isLargerAsMax  = aPixelX > LastValidPixel.x ? 1 : 0;
                    int isInTheMiddle  = isSmallerAsMin == isLargerAsMax ? 1 : 0;

                    aPixelX = isSmallerAsMin * aPixelX + isLargerAsMax * (2 * LastValidPixel.x - aPixelX) +
                              isInTheMiddle * aPixelX;

                    isSmallerAsMin = aPixelY < 0 ? -1 : 0;
                    isLargerAsMax  = aPixelY > LastValidPixel.y ? 1 : 0;
                    isInTheMiddle  = isSmallerAsMin == isLargerAsMax ? 1 : 0;

                    aPixelY = isSmallerAsMin * aPixelY + isLargerAsMax * (2 * LastValidPixel.y - aPixelY) +
                              isInTheMiddle * aPixelY;
                }
                else
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX;
                    }
                    else if (aPixelX > LastValidPixel.x)
                    {
                        aPixelX = 2 * LastValidPixel.x - aPixelX;
                    }

                    if (aPixelY < 0)
                    {
                        aPixelY = -aPixelY;
                    }
                    else if (aPixelY > LastValidPixel.y)
                    {
                        aPixelY = 2 * LastValidPixel.y - aPixelY;
                    }
                }
            }
        }
        else if constexpr (borderType == BorderType::MirrorReplicate)
        {
            if constexpr (iterative)
            {
                while (aPixelX < 0 || aPixelX > LastValidPixel.x)
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX - 1;
                    }
                    else if (aPixelX > LastValidPixel.x)
                    {
                        aPixelX = 2 * LastValidPixel.x - aPixelX + 1;
                    }
                }

                while (aPixelY < 0 || aPixelY > LastValidPixel.y)
                {
                    if (aPixelY < 0)
                    {
                        aPixelY = -aPixelY - 1;
                    }
                    else if (aPixelY > LastValidPixel.y)
                    {
                        aPixelY = 2 * LastValidPixel.y - aPixelY + 1;
                    }
                }
            }
            else
            {
                if constexpr (avoidBranching)
                {
                    int isSmallerAsMin = aPixelX < 0 ? -1 : 0;
                    int isLargerAsMax  = aPixelX > LastValidPixel.x ? 1 : 0;
                    int isInTheMiddle  = isSmallerAsMin == isLargerAsMax ? 1 : 0;

                    aPixelX = isSmallerAsMin * (aPixelX + 1) + isLargerAsMax * (2 * LastValidPixel.x - aPixelX + 1) +
                              isInTheMiddle * aPixelX;

                    isSmallerAsMin = aPixelY < 0 ? -1 : 0;
                    isLargerAsMax  = aPixelY > LastValidPixel.y ? 1 : 0;
                    isInTheMiddle  = isSmallerAsMin == isLargerAsMax ? 1 : 0;

                    aPixelY = isSmallerAsMin * (aPixelY + 1) + isLargerAsMax * (2 * LastValidPixel.y - aPixelY + 1) +
                              isInTheMiddle * aPixelY;
                }
                else
                {
                    if (aPixelX < 0)
                    {
                        aPixelX = -aPixelX - 1;
                    }
                    else if (aPixelX > LastValidPixel.x)
                    {
                        aPixelX = 2 * LastValidPixel.x - aPixelX + 1;
                    }

                    if (aPixelY < 0)
                    {
                        aPixelY = -aPixelY - 1;
                    }
                    else if (aPixelY > LastValidPixel.y)
                    {
                        aPixelY = 2 * LastValidPixel.y - aPixelY + 1;
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
                    aPixelX += SizeAllowedAccessRoi.x;
                }
                while (aPixelX > LastValidPixel.x)
                {
                    aPixelX -= SizeAllowedAccessRoi.x;
                }

                while (aPixelY < 0)
                {
                    aPixelY += SizeAllowedAccessRoi.y;
                }
                while (aPixelY > LastValidPixel.y)
                {
                    aPixelY -= SizeAllowedAccessRoi.y;
                }
            }
            else
            {
                if (aPixelX < 0)
                {
                    aPixelX += SizeAllowedAccessRoi.x;
                }
                else if (aPixelX > LastValidPixel.x)
                {
                    aPixelX -= SizeAllowedAccessRoi.x;
                }

                if (aPixelY < 0)
                {
                    aPixelY += SizeAllowedAccessRoi.y;
                }
                else if (aPixelY > LastValidPixel.y)
                {
                    aPixelY -= SizeAllowedAccessRoi.y;
                }
            }
        }
    }

    DEVICE_CODE PixelT operator()(int aPixelX, int aPixelY) const
    {
        AdjustCoordinates(aPixelX, aPixelY);

        if constexpr (borderType == BorderType::Constant)
        {
            if (aPixelX < 0 || aPixelX > LastValidPixel.x || aPixelY < 0 || aPixelY > LastValidPixel.y)
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