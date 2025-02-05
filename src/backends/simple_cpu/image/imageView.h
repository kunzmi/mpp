#pragma once
#include <backends/cuda/image/imageView.h>
#include <common/arithmetic/binary_operators.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{

template <PixelType T> class ImageView
{
#pragma region Iterator
  public:
    // With this iterator we can use a simple foreach-loop over the imageView to iterate through the pixels
    template <bool isConst> struct _iterator
    {
      private:
        Vec2i mPixel{0};
        std::conditional_t<isConst, const ImageView &, ImageView &>
            mImgView; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

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

        _iterator operator++(int) & // NOLINT(cert-dcl21-cpp)
        {
            _iterator ret = *this;
            operator++();
            return ret;
        }
        _iterator operator--(int) & // NOLINT(cert-dcl21-cpp)
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

        [[nodiscard]] _iterator operator[](difference_type aRhs) const
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

        [[nodiscard]] const Vec2i &Pixel() const
        {
            return mPixel;
        }

        [[nodiscard]] const T &Value() const
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

#ifdef _MSC_VER
    friend iterator operator+(iterator::difference_type aLhs, const iterator &aRhs);
    friend iterator operator-(iterator::difference_type aLhs, const iterator &aRhs);
    friend const_iterator operator+(const_iterator::difference_type aLhs, const const_iterator &aRhs);
    friend const_iterator operator-(const_iterator::difference_type aLhs, const const_iterator &aRhs);
#elif __clang__
    friend iterator operator+(iterator::difference_type aLhs, const iterator &aRhs);
    friend iterator operator-(iterator::difference_type aLhs, const iterator &aRhs);
    friend const_iterator operator+(const_iterator::difference_type aLhs, const const_iterator &aRhs);
    friend const_iterator operator-(const_iterator::difference_type aLhs, const const_iterator &aRhs);
#else
    // thanks GCC... explicit template specialisation should be allowed in non-namespace scope, and visual C++ doesn't
    // compile with that version, hence the ifdef
    template <bool _isConst>
    friend _iterator<_isConst> operator+(_iterator<_isConst>::difference_type aLhs, const _iterator<_isConst> &aRhs);
    template <bool _isConst>
    friend _iterator<_isConst> operator-(_iterator<_isConst>::difference_type aLhs, const _iterator<_isConst> &aRhs);

#endif

#pragma endregion

#pragma region Constructors
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

  private:
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

    explicit ImageView(const Size2D &aSize) : mSizeAlloc(aSize), mRoi(0, 0, aSize)
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
#pragma endregion

#pragma region Basics and Copy to device/host
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

    [[nodiscard]] iterator begin()
    {
        return iterator(mRoi.FirstPixel(), *this);
    }

    [[nodiscard]] iterator end()
    {
        return iterator({mRoi.FirstX(), mRoi.LastY() + 1}, *this);
    }

    [[nodiscard]] const_iterator begin() const
    {
        return cbegin();
    }

    [[nodiscard]] const_iterator end() const
    {
        return cend();
    }

    [[nodiscard]] const_iterator cbegin() const
    {
        return const_iterator(mRoi.FirstPixel(), *this);
    }

    [[nodiscard]] const_iterator cend() const
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
    [[nodiscard]] bool IsIdentical(const ImageView<T> &aOther) const
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
    [[nodiscard]] bool IsSimilar(const ImageView<T> &aOther, remove_vector_t<T> aMaxDiff) const
        requires RealFloatingVector<T>
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
#pragma endregion

#pragma region Data initialisation
#pragma region Convert
    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounding using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst)
        requires(!std::same_as<T, TTo>) &&
                (RealOrComplexIntVector<T> || (std::same_as<complex_basetype_t<remove_vector_t<T>>, float> &&
                                               (std::same_as<complex_basetype_t<remove_vector_t<TTo>>, BFloat16> ||
                                                std::same_as<complex_basetype_t<remove_vector_t<TTo>>, HalfFp16>)));

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16: RoundingMode::NearestTiesAwayFromZero is NOT supported, for float32 to BFloat, on
    /// host only RoundingMode::NearestTiesToEven and RoundingMode::TowardZero are supported.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode)
        requires(!std::same_as<T, TTo>) && RealOrComplexFloatingVector<T>;

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -> float -> scale -> DstT
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor)
        requires(!std::same_as<T, TTo>) && (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) &&
                (!std::same_as<TTo, Complex<float>>) && (!std::same_as<TTo, Complex<double>>);
#pragma endregion
#pragma region Copy
    /// <summary>
    /// Copy image.
    /// </summary>
    ImageView<T> &Copy(ImageView<T> &aDst);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    ImageView<T> &Copy(ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst, Channel aDstChannel)
        requires(vector_size_v<T> > 1) &&   //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy this single channel image to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(ImageView<TTo> &aDst, Channel aDstChannel)
        requires(vector_size_v<T> == 1) &&  //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst)
        requires(vector_size_v<T> > 1) &&    //
                (vector_size_v<TTo> == 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2)
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3)
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel4)
        requires(FourChannelNoAlpha<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2, ImageView<T> &aDst)
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3, ImageView<T> &aDst)
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel4, ImageView<T> &aDst)
        requires(FourChannelNoAlpha<T>);
#pragma endregion
#pragma region Dup
    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Dup(ImageView<TTo> &aDst)
        requires(vector_size_v<T> == 1) &&
                (vector_size_v<TTo> > 1) && std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;
#pragma endregion
#pragma region Scale
    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst)
        requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T> && RealOrComplexIntVector<TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax)
        requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax)
        requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax)
        requires(!std::same_as<T, TTo>);

#pragma endregion
#pragma region Set
    ImageView<T> &Set(const T &aConst);

    ImageView<T> &Set(const T &aConst, const ImageView<Pixel8uC1> &aMask);
#pragma endregion
#pragma region Swap Channel
    /// <summary>
    /// Swap channels
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels)
        requires((vector_active_size_v<TTo> <= vector_active_size_v<T>)) && //
                (vector_size_v<T> >= 3) &&                                  //
                (vector_size_v<TTo> >= 3) &&                                //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Swap channels (3-channel to 4-channel with additional value). If aDstChannels[i] == 3, channel i of aDst is set
    /// to aValue, if aDstChannels[i] > 3, channel i of aDst is kept unchanged.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                remove_vector_t<T> aValue)
        requires(vector_size_v<T> == 3) &&          //
                (vector_active_size_v<TTo> == 4) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;
#pragma endregion

#pragma region FillRandom
    ImageView<T> &FillRandom();
#pragma endregion
#pragma endregion

#pragma region Arithmetic functions
#pragma region Add
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const T &aConst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Sub
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const T &aConst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubInv(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubInv(const ImageView<T> &aSrc2, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubInv(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubInv(const T &aConst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Mul
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const T &aConst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region MulScale
    ImageView<T> &MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const T &aConst, ImageView<T> &aDst)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const ImageView<T> &aSrc2)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const T &aConst)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;
#pragma endregion
#pragma region Div
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const T &aConst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInv(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInv(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                         RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInv(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInv(const T &aConst, int aScaleFactor = 0,
                         RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                         RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                         RoundingMode aRoundingMode = RoundingMode::NearestTiesAwayFromZero)
        requires RealOrComplexIntVector<T>;
#pragma endregion

#pragma region Abs
    ImageView<T> &Abs(ImageView<T> &aDst)
        requires RealSignedVector<T>;

    ImageView<T> &Abs()
        requires RealSignedVector<T>;
#pragma endregion
#pragma region AbsDiff
    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealUnsignedVector<T>;

    ImageView<T> &AbsDiff(const T &aConst, ImageView<T> &aDst)
        requires RealUnsignedVector<T>;

    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2)
        requires RealUnsignedVector<T>;

    ImageView<T> &AbsDiff(const T &aConst)
        requires RealUnsignedVector<T>;
#pragma endregion
#pragma region And
    ImageView<T> &And(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &And(const T &aConst, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &And(const ImageView<T> &aSrc2)
        requires RealIntVector<T>;

    ImageView<T> &And(const T &aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region Not
    ImageView<T> &Not(ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &Not()
        requires RealIntVector<T>;
#pragma endregion
#pragma region Exp
    ImageView<T> &Exp(ImageView<T> &aDst)
        requires RealOrComplexVector<T>;

    ImageView<T> &Exp()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Ln
    ImageView<T> &Ln(ImageView<T> &aDst)
        requires RealOrComplexVector<T>;

    ImageView<T> &Ln()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region LShift
    ImageView<T> &LShift(uint aConst, ImageView<T> &aDst)
        requires RealIntVector<T>;
    ImageView<T> &LShift(uint aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region Or
    ImageView<T> &Or(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &Or(const T &aConst, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &Or(const ImageView<T> &aSrc2)
        requires RealIntVector<T>;

    ImageView<T> &Or(const T &aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region RShift
    ImageView<T> &RShift(uint aConst, ImageView<T> &aDst)
        requires RealIntVector<T>;
    ImageView<T> &RShift(uint aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region Sqr
    ImageView<T> &Sqr(ImageView<T> &aDst)
        requires RealOrComplexVector<T>;

    ImageView<T> &Sqr()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Sqrt
    ImageView<T> &Sqrt(ImageView<T> &aDst)
        requires RealOrComplexVector<T>;

    ImageView<T> &Sqrt()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Xor
    ImageView<T> &Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &Xor(const T &aConst, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &Xor(const ImageView<T> &aSrc2)
        requires RealIntVector<T>;

    ImageView<T> &Xor(const T &aConst)
        requires RealIntVector<T>;
#pragma endregion

#pragma region AlphaPremul
    /// <summary>
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesAwayFromZero (round()) which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(ImageView<T> &aDst)
        requires FourChannelNoAlpha<T>;

    ImageView<T> &AlphaPremul()
        requires FourChannelNoAlpha<T>;

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst)
        requires RealFloatingVector<T>;

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst)
        requires RealIntVector<T>;

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha)
        requires RealFloatingVector<T>;

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha)
        requires RealIntVector<T>;
#pragma endregion

#pragma region AlphaComp
    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp)
        requires(!FourChannelAlpha<T>) && RealVector<T>;

    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                            remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp)
        requires RealVector<T>;
#pragma endregion

#pragma region Complex
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires ComplexVector<T>;

    ImageView<T> &ConjMul(const ImageView<T> &aSrc2)
        requires ComplexVector<T>;

    ImageView<T> &Conj(ImageView<T> &aDst)
        requires ComplexVector<T>;

    ImageView<T> &Conj()
        requires ComplexVector<T>;

    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Magnitude(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &MagnitudeSqr(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Angle(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Real(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
        requires ComplexVector<T>;

    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Imag(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
        requires ComplexVector<T>;

    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst)
        requires RealSignedVector<T> && (!FourChannelAlpha<T>);

    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        const ImageView<T> &aSrcImag,
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst)
        requires RealSignedVector<T> && (!FourChannelAlpha<T>);
#pragma endregion
#pragma endregion

#pragma region Statistics
#pragma region MinEvery
    ImageView<T> &MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealVector<T>;

    ImageView<T> &MinEvery(const ImageView<T> &aSrc2)
        requires RealVector<T>;
#pragma endregion
#pragma region MaxEvery
    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst)
        requires RealVector<T>;

    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2)
        requires RealVector<T>;
#pragma endregion
#pragma endregion

#pragma region Threshold and Compare
#pragma region Compare
    ImageView<Pixel8uC1> &Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst);

    ImageView<Pixel8uC1> &Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst);

    ImageView<Pixel8uC1> &CompareEqEps(const ImageView<T> &aSrc2, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                       ImageView<Pixel8uC1> &aDst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<Pixel8uC1> &CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                       ImageView<Pixel8uC1> &aDst)
        requires RealOrComplexFloatingVector<T>;
#pragma endregion
#pragma region Threshold
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare)
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold)
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold)
        requires RealVector<T>;

    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare)
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue)
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue)
        requires RealVector<T>;
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT,
                                ImageView<T> &aDst)
        requires RealVector<T>;
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT)
        requires RealVector<T>;
#pragma endregion
#pragma endregion
};
} // namespace opp::image::cpuSimple