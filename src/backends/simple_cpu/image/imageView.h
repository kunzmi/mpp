#pragma once
#include <backends/cuda/image/imageView.h>
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <vector>

namespace opp::image::cpuSimple
{

template <PixelType T> class ImageView
{
  public:
    // With this iterator we can use a simple foreach-loop over the imageView to iterate through the pixels
    template <bool isConst> struct _iterator
    {
      private:
        Vec2i mPixel;
        std::conditional_t<isConst, const ImageView &, ImageView &> mImgView;

      public:
        _iterator(const Vec2i &aPixel, std::conditional_t<isConst, const ImageView &, ImageView &> aImgView)
            : mPixel(aPixel), mImgView(aImgView)
        {
        }

        _iterator() = default;

        ~_iterator() = default;

        _iterator(const _iterator &)     = default;
        _iterator(_iterator &&) noexcept = default;

        _iterator &operator=(const _iterator &)     = default;
        _iterator &operator=(_iterator &&) noexcept = default;

        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = std::conditional_t<isConst, const T *, T *>;
        // By setting the reference type to the iterator itself, foreach-loops allow us the access to the iterator
        // and we have the information of the current pixel coordinate. To access the actual pixel value, we use the
        // Value() method.
        using reference = ImageView<T>::_iterator<isConst> &;

        _iterator &operator++()
        {
            mPixel.x++;
            if (mPixel.x > mImgView.mRoi.LastX())
            {
                mPixel.x = mImgView.mRoi.FirstX();
                mPixel.y++;
            }
            return *this;
        }
        _iterator &operator--()
        {
            mPixel.x--;
            if (mPixel.x < mImgView.mRoi.FirstX())
            {
                mPixel.x = mImgView.mRoi.LastX();
                mPixel.y--;
            }
            return *this;
        }

        _iterator operator++(int) &
        {
            _iterator ret = *this;
            operator++();
            return ret;
        }
        _iterator operator--(int) &
        {
            _iterator ret = *this;
            operator--();
            return ret;
        }

        [[nodiscard]] bool operator==(_iterator const &aOther) const
        {
            return std::addressof(mImgView) == std::addressof(aOther.mImgView) && mPixel == aOther.mPixel;
        }

        [[nodiscard]] bool operator!=(_iterator const &aOther) const
        {
            return std::addressof(mImgView) != std::addressof(aOther.mImgView) || mPixel != aOther.mPixel;
        }

        reference operator*()
        {
            return *this;
        }

        pointer operator->()
        {
            return gotoPtr(mImgView.mPtr, mImgView.mPitch, mPixel.x, mPixel.y);
        }

        [[nodiscard]] reference operator[](difference_type aRhs) const
        {
            difference_type diffY = aRhs / mImgView.mRoi.width;
            difference_type diffX = aRhs - (diffY * mImgView.mRoi.width);
            int x                 = mPixel.x + to_int(diffX);
            int y                 = mPixel.y + to_int(diffY);

            if (x > mImgView.mRoi.LastX())
            {
                x -= mImgView.mRoi.width;
                y++;
            }
            if (x < mImgView.mRoi.FirstX())
            {
                x += mImgView.mRoi.width;
                y--;
            }
            _iterator ret = *this;
            ret.mPixel.x  = x;
            ret.mPixel.y  = y;
            return ret;
        }

        [[nodiscard]] difference_type operator-(const _iterator &aRhs) const
        {
            return difference_type(mPixel.y - aRhs.mPixel.y) * difference_type(mImgView.mRoi.width) +
                   difference_type(mPixel.x - aRhs.mPixel.x);
        }

        [[nodiscard]] _iterator &operator+=(difference_type aRhs)
        {
            difference_type diffY = aRhs / mImgView.mRoi.width;
            difference_type diffX = aRhs - (diffY * mImgView.mRoi.width);
            mPixel.x += to_int(diffX);
            mPixel.y += to_int(diffY);

            if (mPixel.x > mImgView.mRoi.LastX())
            {
                mPixel.x -= mImgView.mRoi.width;
                mPixel.y++;
            }
            if (mPixel.x < mImgView.mRoi.FirstX())
            {
                mPixel.x += mImgView.mRoi.width;
                mPixel.y--;
            }
            return *this;
        }
        [[nodiscard]] _iterator &operator-=(difference_type aRhs)
        {
            difference_type diffY = aRhs / mImgView.mRoi.width;
            difference_type diffX = aRhs - (diffY * mImgView.mRoi.width);
            mPixel.x -= to_int(diffX);
            mPixel.y -= to_int(diffY);

            if (mPixel.x > mImgView.mRoi.LastX())
            {
                mPixel.x -= mImgView.mRoi.width;
                mPixel.y++;
            }
            if (mPixel.x < mImgView.mRoi.FirstX())
            {
                mPixel.x += mImgView.mRoi.width;
                mPixel.y--;
            }
            return *this;
        }

        [[nodiscard]] _iterator operator+(difference_type aRhs) const
        {
            _iterator ret(*this);
            difference_type diffY = aRhs / mImgView.mRoi.width;
            difference_type diffX = aRhs - (diffY * mImgView.mRoi.width);
            ret.mPixel.x += to_int(diffX);
            ret.mPixel.y += to_int(diffY);

            if (ret.mPixel.x > ret.mImgView.mRoi.LastX())
            {
                ret.mPixel.x -= ret.mImgView.mRoi.width;
                ret.mPixel.y++;
            }
            if (ret.mPixel.x < ret.mImgView.mRoi.FirstX())
            {
                ret.mPixel.x += ret.mImgView.mRoi.width;
                ret.mPixel.y--;
            }
            return ret;
        }
        [[nodiscard]] _iterator operator-(difference_type aRhs) const
        {
            _iterator ret(*this);
            difference_type diffY = aRhs / mImgView.mRoi.width;
            difference_type diffX = aRhs - (diffY * mImgView.mRoi.width);
            ret.mPixel.x -= to_int(diffX);
            ret.mPixel.y -= to_int(diffY);

            if (ret.mPixel.x > ret.mImgView.mRoi.LastX())
            {
                ret.mPixel.x -= ret.mImgView.mRoi.width;
                ret.mPixel.y++;
            }
            if (ret.mPixel.x < ret.mImgView.mRoi.FirstX())
            {
                ret.mPixel.x += ret.mImgView.mRoi.width;
                ret.mPixel.y--;
            }
            return ret;
        }
        friend _iterator operator+(difference_type aLhs, const _iterator &aRhs)
        {
            _iterator ret(aRhs);
            difference_type diffY = aLhs / ret.mImgView.mRoi.width;
            difference_type diffX = aLhs - (diffY * ret.mImgView.mRoi.width);
            ret.mPixel.x += to_int(diffX);
            ret.mPixel.y += to_int(diffY);

            if (ret.mPixel.x > ret.mImgView.mRoi.LastX())
            {
                ret.mPixel.x -= ret.mImgView.mRoi.width;
                ret.mPixel.y++;
            }
            if (ret.mPixel.x < ret.mImgView.mRoi.FirstX())
            {
                ret.mPixel.x += ret.mImgView.mRoi.width;
                ret.mPixel.y--;
            }
            return ret;
        }
        friend _iterator operator-(difference_type aLhs, const _iterator &aRhs)
        {
            _iterator ret(aRhs);
            difference_type diffY = aLhs / ret.mImgView.mRoi.width;
            difference_type diffX = aLhs - (diffY * ret.mImgView.mRoi.width);
            ret.mPixel.x -= to_int(diffX);
            ret.mPixel.y -= to_int(diffY);

            if (ret.mPixel.x > ret.mImgView.mRoi.LastX())
            {
                ret.mPixel.x -= ret.mImgView.mRoi.width;
                ret.mPixel.y++;
            }
            if (ret.mPixel.x < ret.mImgView.mRoi.FirstX())
            {
                ret.mPixel.x += ret.mImgView.mRoi.width;
                ret.mPixel.y--;
            }
            return ret;
        }

        [[nodiscard]] bool operator>(const _iterator &aRhs) const
        {
            if (mPixel.y > aRhs.mPixel.y)
            {
                return true;
            }
            if (mPixel.y < aRhs.mPixel.y)
            {
                return false;
            }
            return mPixel.x > aRhs.mPixel.x;
        }
        [[nodiscard]] bool operator<(const _iterator &aRhs) const
        {
            if (mPixel.y < aRhs.mPixel.y)
            {
                return true;
            }
            if (mPixel.y > aRhs.mPixel.y)
            {
                return false;
            }
            return mPixel.x < aRhs.mPixel.x;
        }
        [[nodiscard]] bool operator>=(const _iterator &aRhs) const
        {
            if (mPixel.y > aRhs.mPixel.y)
            {
                return true;
            }
            if (mPixel.y < aRhs.mPixel.y)
            {
                return false;
            }
            return mPixel.x >= aRhs.mPixel.x;
        }
        [[nodiscard]] bool operator<=(const _iterator &aRhs) const
        {
            if (mPixel.y < aRhs.mPixel.y)
            {
                return true;
            }
            if (mPixel.y > aRhs.mPixel.y)
            {
                return false;
            }
            return mPixel.x <= aRhs.mPixel.x;
        }

        const Vec2i &Pixel() const
        {
            return mPixel;
        }

        const T &Value() const
        {
            return *gotoPtr(mImgView.mPtr, mImgView.mPitch, mPixel.x, mPixel.y);
        }

        T &Value()
            requires(!isConst)
        {
            return *gotoPtr(mImgView.mPtr, mImgView.mPitch, mPixel.x, mPixel.y);
        }
    };

    using iterator       = _iterator<false>;
    using const_iterator = _iterator<true>;

    friend iterator;
    friend const_iterator;

  public:
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
    static constexpr size_t ChannelCount = to_size_t(channel_count<T>::value);

  private:
    /// <summary>
    /// Base pointer to image data.
    /// </summary>
    T *mPtr;

    /// <summary>
    /// Width in bytes of one image line + alignment bytes.
    /// </summary>
    size_t mPitch;

    /// <summary>
    /// Base pointer moved to actual ROI.
    /// </summary>
    T *mPtrRoi;

    /// <summary>
    /// Size of the allocated image buffer (full ROI).
    /// </summary>
    Size2D mSizeAlloc;

    /// <summary>
    /// ROI.
    /// </summary>
    Roi mRoi;

  protected:
    ImageView() = default;

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

    ImageView(const Size2D &aSize) : mPtr(nullptr), mPitch(0), mPtrRoi(nullptr), mSizeAlloc(aSize), mRoi(0, 0, aSize)
    {
    }

  public:
    ImageView(T *aBasePointer, const SizePitched &aSizeAlloc)
        : mPtr(aBasePointer), mPitch(aSizeAlloc.Pitch()), mPtrRoi(aBasePointer), mSizeAlloc(aSizeAlloc.Size()),
          mRoi(0, 0, aSizeAlloc.Size())
    {
    }
    ImageView(T *aBasePointer, const SizePitched &aSizeAlloc, const Roi &aRoi)
        : mPtr(aBasePointer), mPitch(aSizeAlloc.Pitch()),
          mPtrRoi(gotoPtr(aBasePointer, aSizeAlloc.Pitch(), aRoi.x, aRoi.y)), mSizeAlloc(aSizeAlloc.Size()), mRoi(aRoi)
    {
        checkRoiIsInRoi(aRoi, Roi(0, 0, mSizeAlloc));
    }
    ~ImageView() = default;

    ImageView(const ImageView &)     = default;
    ImageView(ImageView &&) noexcept = default;

    ImageView &operator=(const ImageView &)     = default;
    ImageView &operator=(ImageView &&) noexcept = default;

    [[nodiscard]] T &operator()(int aPixelX, int aPixelY)
    {
        return *gotoPtr(mPtrRoi, mPitch, aPixelX, aPixelY);
    }

    [[nodiscard]] const T &operator()(int aPixelX, int aPixelY) const
    {
        return *gotoPtr(mPtrRoi, mPitch, aPixelX, aPixelY);
    }

    /// <summary>
    /// Base pointer to image data.
    /// </summary>
    [[nodiscard]] T *Pointer() const
    {
        return mPtr;
    }
    /// <summary>
    /// Base pointer moved to actual ROI.
    /// </summary>
    [[nodiscard]] T *PointerRoi() const
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
    /// Returns a new ImageView with the new ROI
    /// </summary>
    ImageView GetView(const Roi &aRoi)
    {
        return ImageView(mPtr, SizePitched(mSizeAlloc, mPitch), aRoi);
    }

    /// <summary>
    /// Returns a new ImageView with the current ROI adapted by aBorder
    /// </summary>
    ImageView GetView(const Border &aBorder = Border())
    {
        const Roi newRoi = mRoi + aBorder;
        checkRoiIsInRoi(newRoi, Roi(0, 0, mSizeAlloc));
        return ImageView(mPtr, SizePitched(mSizeAlloc, mPitch), newRoi);
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

    iterator begin()
    {
        return iterator(mRoi.FirstPixel(), *this);
    }

    iterator end()
    {
        return iterator({mRoi.FirstX(), mRoi.LastY() + 1}, *this);
    }

    const_iterator begin() const
    {
        return cbegin();
    }

    const_iterator end() const
    {
        return cend();
    }

    const_iterator cbegin() const
    {
        return const_iterator(mRoi.FirstPixel(), *this);
    }

    const_iterator cend() const
    {
        return const_iterator({mRoi.FirstX(), mRoi.LastY() + 1}, *this);
    }

    /// <summary>
    /// Copy from host to device memory
    /// </summary>
    /// <param name="aDeviceDst">Destination</param>
    void CopyToDevice(cuda::ImageView<T> &aDeviceDst) const
    {
        if (mSizeAlloc != aDeviceDst.SizeAlloc())
        {
            throw ROIEXCEPTION("The source image does not have the same size as the destination image. Source size "
                               << mSizeAlloc << ", Destination size: " << aDeviceDst.SizeAlloc());
        }

        aDeviceDst.CopyToDevice(mPtr, mPitch);
    }

    /// <summary>
    /// Copy from device to host memory
    /// </summary>
    /// <param name="aDeviceSrc">Source</param>
    void CopyToHost(const cuda::ImageView<T> &aDeviceSrc)
    {
        if (mSizeAlloc != aDeviceSrc.SizeAlloc())
        {
            throw ROIEXCEPTION("The source image does not have the same size as the destination image. Source size "
                               << aDeviceSrc.SizeAlloc() << ", Destination size: " << mSizeAlloc);
        }

        aDeviceSrc.CopyToHost(mPtr, mPitch);
    }

    /// <summary>
    /// Copy from host to device memory
    /// </summary>
    void operator>>(cuda::ImageView<T> &aDest) const
    {
        CopyToDevice(aDest);
    }

    /// <summary>
    /// Copy from device to host memory
    /// </summary>
    void operator<<(const cuda::ImageView<T> &aDeviceSrc)
    {
        CopyToHost(aDeviceSrc);
    }

    /// <summary>
    /// Copy data from host to device memory only in ROI
    /// </summary>
    /// <param name="aDeviceDst">Device destination view</param>
    void CopyToDeviceRoi(cuda::ImageView<T> &aDeviceDst) const
    {
        // the callee will move the ptr to the first roi pixel
        aDeviceDst.CopyToDeviceRoi(mPtr, mPitch, mRoi);
    }

    /// <summary>
    /// Copy data from device to device memory
    /// </summary>
    /// <param name="aDeviceSrc">Device source view</param>
    void CopyToHostRoi(const cuda::ImageView<T> &aDeviceSrc)
    {
        // the callee will move the ptr to the first roi pixel
        aDeviceSrc.CopyToHostRoi(mPtr, mPitch, mRoi);
    }

    /// <summary>
    /// Returns true, if size and pixel content is identical (inside the ROI). Returns false if ROI size differs.
    /// </summary>
    bool IsIdentical(const ImageView<T> &aOther) const
    {
        if (aOther.SizeRoi() != SizeRoi())
        {
            return false;
        }
        auto iterSrc1 = cbegin();
        for (const auto &elemSrc2 : aOther)
        {
            if (elemSrc2.Value() != iterSrc1.Value())
            {
                return false;
            }
            ++iterSrc1;
        }

        return true;
    }

    /// <summary>
    /// Returns true, if size is equal and pixel content is identical up to provided limit (inside the ROI). Returns
    /// false if ROI size differs.
    /// </summary>
    bool IsSimilar(const ImageView<T> &aOther, remove_vector_t<T> aMaxDiff) const
        requires RealOrComplexFloatingVector<T>
    {
        if (aOther.SizeRoi() != SizeRoi())
        {
            return false;
        }

        T limit(aMaxDiff);

        auto iterSrc1 = cbegin();
        for (const auto &elemSrc2 : aOther)
        {
            T diff = T::Abs(elemSrc2.Value() - iterSrc1.Value());
            if (!(diff < limit))
            {
                return false;
            }
            ++iterSrc1;
        }
        return true;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aDst.ROI());

        using ComputeT = default_compute_type_for_t<T>;

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoudingMode::None>;

        opp::Add<ComputeT> op;
        addSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

        forEachPixel(aDst, functor);
        return aDst;
    }
};
} // namespace opp::image::cpuSimple