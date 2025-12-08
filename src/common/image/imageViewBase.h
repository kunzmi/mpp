#pragma once

#include <common/defines.h>
#include <common/image/border.h>
#include <common/image/bound.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <common/vector_typetraits.h>
#include <cstddef>
#include <utility>
#include <vector>

namespace mpp::image
{

template <PixelType T> class ImageViewBase
{
  public:
    enum class MemoryType_enum
    {
        HostDefault,
        CudaDefault
    };
    virtual MemoryType_enum MemoryType() const = 0;

    /// <summary>
    /// Type size in bytes of one pixel in one channel.
    /// </summary>
    static constexpr size_t TypeSize = sizeof(remove_vector_t<T>);
    /// <summary>
    /// Size in bytes of one pixel for all image channels
    /// </summary>
    static constexpr size_t PixelSizeInBytes = sizeof(T);
    /// <summary>
    /// Channel count
    /// </summary>
    static constexpr size_t ChannelCount = to_size_t(channel_count_v<T>);

    using pixel_type_t = T;

  protected:
    /// <summary>
    /// Base pointer to image data.
    /// </summary>
    T *mPtr{nullptr};

    /// <summary>
    /// Width in bytes of one image line + alignment bytes.
    /// </summary>
    size_t mPitch{0};

    /// <summary>
    /// Base pointer moved to actual ROI.
    /// </summary>
    T *mPtrRoi{nullptr};

    /// <summary>
    /// Size of the allocated image buffer (full ROI).
    /// </summary>
    Size2D mSizeAlloc;

    /// <summary>
    /// ROI.
    /// </summary>
    Roi mRoi;

    ImageViewBase() = default;

    T *&PointerRef()
    {
        return mPtr;
    }
    size_t &PitchRef()
    {
        return mPitch;
    }
    T *&PointerRoiRef()
    {
        return mPtrRoi;
    }
    Size2D &SizeAllocRef()
    {
        return mSizeAlloc;
    }
    Roi &ROIRef()
    {
        return mRoi;
    }

    explicit ImageViewBase(const Size2D &aSize) noexcept : mSizeAlloc(aSize), mRoi(0, 0, aSize)
    {
    }

    ImageViewBase(T *aBasePointer, const SizePitched &aSizeAlloc) noexcept
        : mPtr(aBasePointer), mPitch(aSizeAlloc.Pitch()), mPtrRoi(aBasePointer), mSizeAlloc(aSizeAlloc.Size()),
          mRoi(0, 0, aSizeAlloc.Size())
    {
    }
    ImageViewBase(T *aBasePointer, const SizePitched &aSizeAlloc, const Roi &aRoi)
        : mPtr(aBasePointer), mPitch(aSizeAlloc.Pitch()),
          mPtrRoi(gotoPtr(aBasePointer, aSizeAlloc.Pitch(), aRoi.x, aRoi.y)), mSizeAlloc(aSizeAlloc.Size()), mRoi(aRoi)
    {
        checkRoiIsInRoi(aRoi, Roi(0, 0, mSizeAlloc));
    }

  public:
    virtual ~ImageViewBase() = default;

    ImageViewBase(const ImageViewBase &)     = default;
    ImageViewBase(ImageViewBase &&) noexcept = default;

    ImageViewBase &operator=(const ImageViewBase &)     = default;
    ImageViewBase &operator=(ImageViewBase &&) noexcept = default;

    /// <summary>
    /// Base pointer to image data.
    /// </summary>
    [[nodiscard]] T *Pointer()
    {
        return mPtr;
    }
    /// <summary>
    /// Base pointer to image data.
    /// </summary>
    [[nodiscard]] const T *Pointer() const
    {
        return mPtr;
    }
    /// <summary>
    /// Base pointer moved to actual ROI.
    /// </summary>
    [[nodiscard]] T *PointerRoi()
    {
        return mPtrRoi;
    }
    /// <summary>
    /// Base pointer moved to actual ROI.
    /// </summary>
    [[nodiscard]] const T *PointerRoi() const
    {
        return mPtrRoi;
    }
    /// <summary>
    /// Size of the entire allocated image.
    /// </summary>
    [[nodiscard]] const Size2D &SizeAlloc() const
    {
        return mSizeAlloc;
    }
    /// <summary>
    /// Size of the current image ROI.
    /// </summary>
    [[nodiscard]] Size2D SizeRoi() const
    {
        return {mRoi.width, mRoi.height};
    }
    /// <summary>
    /// ROI.
    /// </summary>
    [[nodiscard]] const Roi &ROI() const
    {
        return mRoi;
    }
    /// <summary>
    /// Width of one image line + alignment bytes.
    /// </summary>
    [[nodiscard]] size_t Pitch() const
    {
        return mPitch;
    }

    /// <summary>
    /// Image width in pixels
    /// </summary>
    [[nodiscard]] int Width() const
    {
        return mSizeAlloc.x;
    }

    /// <summary>
    /// Image width in bytes (without padding)
    /// </summary>
    [[nodiscard]] size_t WidthInBytes() const
    {
        return to_size_t(mSizeAlloc.x) * PixelSizeInBytes;
    }

    /// <summary>
    /// Height in pixels
    /// </summary>
    [[nodiscard]] int Height() const
    {
        return mSizeAlloc.y;
    }

    /// <summary>
    /// Roi width in pixels
    /// </summary>
    [[nodiscard]] int WidthRoi() const
    {
        return mRoi.width;
    }

    /// <summary>
    /// Roi width in bytes
    /// </summary>
    [[nodiscard]] size_t WidthRoiInBytes() const
    {
        return to_size_t(mRoi.width) * PixelSizeInBytes;
    }

    /// <summary>
    /// Height in pixels
    /// </summary>
    [[nodiscard]] int HeightRoi() const
    {
        return mRoi.height;
    }

    /// <summary>
    /// Total size in bytes (Pitch * Height)
    /// </summary>
    [[nodiscard]] size_t TotalSizeInBytes() const
    {
        return mPitch * to_size_t(mSizeAlloc.y);
    }

    /// <summary>
    /// Defines the ROI on which all following operations take place
    /// </summary>
    void SetRoi(const Roi &aRoi)
    {
        checkRoiIsInRoi(aRoi, Roi(0, 0, mSizeAlloc));

        mPtrRoi = gotoPtr(mPtr, mPitch, aRoi.x, aRoi.y);
        mRoi    = aRoi;
    }

    /// <summary>
    /// Defines the ROI on which all following operations take place relative to the current ROI
    /// </summary>
    void SetRoi(const Border &aBorder)
    {
        const Roi newRoi = mRoi + aBorder;
        SetRoi(newRoi);
    }

    /// <summary>
    /// Resets the ROI to the full image
    /// </summary>
    void ResetRoi()
    {
        mRoi    = Roi(0, 0, mSizeAlloc);
        mPtrRoi = mPtr;
    }

    /// <summary>
    /// Copy from this view to other view
    /// </summary>
    /// <param name="aDstView">Destination view</param>
    virtual void CopyTo(ImageViewBase &aDstView) const = 0;

    /// <summary>
    /// Copy from other view to this view
    /// </summary>
    /// <param name="aSrcView">Source view</param>
    virtual void CopyFrom(const ImageViewBase &aSrcView) = 0;

    /// <summary>
    /// Copy from this view to other view
    /// </summary>
    void operator>>(ImageViewBase &aDstView) const
    {
        CopyTo(aDstView);
    }

    /// <summary>
    /// Copy from other view to this view
    /// </summary>
    void operator<<(const ImageViewBase &aSrcView)
    {
        CopyFrom(aSrcView);
    }

    /// <summary>
    /// Copy from this view to other view only in ROI
    /// </summary>
    /// <param name="aDstView">Destination view</param>
    virtual void CopyToRoi(ImageViewBase &aDstView) const = 0;

    /// <summary>
    /// Copy from other view to this view only in ROI
    /// </summary>
    /// <param name="aSrcView">Source view</param>
    virtual void CopyFromRoi(const ImageViewBase &aDstView) = 0;
};
} // namespace mpp::image