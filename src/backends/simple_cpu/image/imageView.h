#pragma once
#include "../dllexport_simplecpu.h"
#include <backends/simple_cpu/image/addSquareProductWeightedOutputType.h>
#include <backends/simple_cpu/image/conversionRelations.h>
#include <backends/simple_cpu/image/histogramLevelsTypes.h>
#include <backends/simple_cpu/image/morphologyComputeT.h>
#include <backends/simple_cpu/image/scaleRelations.h>
#include <common/arithmetic/binary_operators.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/filterArea.h>
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
#include <common/image/imageViewBase.h>
#include <common/image/matrix.h>
#include <common/image/matrix3x4.h>
#include <common/image/matrix4x4.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/statistics/indexMinMax.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace mpp::image::cpuSimple
{

template <PixelType T> class MPPEXPORT_SIMPLECPU ImageView : public ImageViewBase<T>
{
#pragma region Iterator
  public:
    // With this iterator we can use a simple foreach-loop over the imageView to iterate through the pixels
    template <bool isConst> struct MPPEXPORT_SIMPLECPU _iterator
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
        friend MPPEXPORT_SIMPLECPU _iterator operator+(difference_type aLhs, const _iterator &aRhs)
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
        friend MPPEXPORT_SIMPLECPU _iterator operator-(difference_type aLhs, const _iterator &aRhs)
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

#if defined(_MSC_VER) || defined(__clang__)
    friend MPPEXPORT_SIMPLECPU iterator operator+(iterator::difference_type aLhs, const iterator &aRhs);
    friend MPPEXPORT_SIMPLECPU iterator operator-(iterator::difference_type aLhs, const iterator &aRhs);
    friend MPPEXPORT_SIMPLECPU const_iterator operator+(const_iterator::difference_type aLhs,
                                                        const const_iterator &aRhs);
    friend MPPEXPORT_SIMPLECPU const_iterator operator-(const_iterator::difference_type aLhs,
                                                        const const_iterator &aRhs);
#else
    // thanks GCC... explicit template specialisation should be allowed in non-namespace scope, and visual C++ doesn't
    // compile with that version, hence the ifdef
    template <bool _isConst>
    friend _iterator<_isConst> MPPEXPORT_SIMPLECPU operator+(_iterator<_isConst>::difference_type aLhs,
                                                             const _iterator<_isConst> &aRhs);
    template <bool _isConst>
    friend _iterator<_isConst> MPPEXPORT_SIMPLECPU operator-(_iterator<_isConst>::difference_type aLhs,
                                                             const _iterator<_isConst> &aRhs);
#endif

#pragma endregion

#pragma region Constructors

  protected:
    ImageView() = default;

    T *&PointerRef()
    {
        return ImageViewBase<T>::mPtr;
    }
    size_t &PitchRef()
    {
        return ImageViewBase<T>::mPitch;
    }
    T *&PointerRoiRef()
    {
        return ImageViewBase<T>::mPtrRoi;
    }
    Size2D &SizeAllocRef()
    {
        return ImageViewBase<T>::mSizeAlloc;
    }
    Roi &ROIRef()
    {
        return ImageViewBase<T>::mRoi;
    }

    explicit ImageView(const Size2D &aSize) : ImageViewBase<T>(aSize)
    {
    }

    [[nodiscard]] ImageViewBase<T>::MemoryType_enum MemoryType() const override
    {
        return ImageViewBase<T>::MemoryType_enum::HostDefault;
    }

    using ImageViewBase<T>::mPtr;
    using ImageViewBase<T>::mPtrRoi;
    using ImageViewBase<T>::mSizeAlloc;
    using ImageViewBase<T>::mPitch;
    using ImageViewBase<T>::mRoi;

  public:
    using ImageViewBase<T>::Pointer;
    using ImageViewBase<T>::PointerRoi;
    using ImageViewBase<T>::SizeAlloc;
    using ImageViewBase<T>::Pitch;
    using ImageViewBase<T>::ROI;
    using ImageViewBase<T>::Width;
    using ImageViewBase<T>::WidthInBytes;
    using ImageViewBase<T>::Height;
    using ImageViewBase<T>::SizeRoi;
    using ImageViewBase<T>::WidthRoi;
    using ImageViewBase<T>::WidthRoiInBytes;
    using ImageViewBase<T>::HeightRoi;
    using ImageViewBase<T>::ChannelCount;
    using ImageViewBase<T>::TypeSize;

    ImageView(T *aBasePointer, const SizePitched &aSizeAlloc) noexcept : ImageViewBase<T>(aBasePointer, aSizeAlloc)
    {
    }
    ImageView(T *aBasePointer, const SizePitched &aSizeAlloc, const Roi &aRoi)
        : ImageViewBase<T>(aBasePointer, aSizeAlloc, aRoi)
    {
    }
    ~ImageView() override = default;

    ImageView(const ImageView &)     = default;
    ImageView(ImageView &&) noexcept = default;

    ImageView &operator=(const ImageView &)     = default;
    ImageView &operator=(ImageView &&) noexcept = default;

    operator ImageView<Vector4<remove_vector_t<T>>>() // NOLINT(hicpp-explicit-conversions)
        requires FourChannelAlpha<T>
    {
        return ImageView<Vector4<remove_vector_t<T>>>(reinterpret_cast<Vector4<remove_vector_t<T>> *>(mPtr),
                                                      SizePitched(mSizeAlloc, mPitch), mRoi);
    }

    operator ImageView<Vector4A<remove_vector_t<T>>>() // NOLINT(hicpp-explicit-conversions)
        requires FourChannelNoAlpha<T>
    {
        return ImageView<Vector4A<remove_vector_t<T>>>(reinterpret_cast<Vector4A<remove_vector_t<T>> *>(mPtr),
                                                       SizePitched(mSizeAlloc, mPitch), mRoi);
    }
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
    /// Copy from this view to other view
    /// </summary>
    /// <param name="aDstView">Destination view</param>
    void CopyTo(ImageViewBase<T> &aDstView) const override
    {
        if (SizeAlloc() != aDstView.SizeAlloc())
        {
            throw ROIEXCEPTION("The source image does not have the same size as the destination image. Source size "
                               << SizeAlloc() << ", Destination size: " << aDstView.SizeAlloc());
        }

        if (aDstView.MemoryType() == ImageViewBase<T>::MemoryType_enum::CudaDefault)
        {
            aDstView.CopyFrom(*this);
        }
        else
        {
            throw INVALIDARGUMENT(
                aDstView,
                "Unknown memory location for destination ImageView. Cannot copy data from CUDA device to there.");
        }
    }

    /// <summary>
    /// Copy from other view to this view
    /// </summary>
    /// <param name="aSrcView">Source view</param>
    void CopyFrom(const ImageViewBase<T> &aSrcView) override
    {
        if (SizeAlloc() != aSrcView.SizeAlloc())
        {
            throw ROIEXCEPTION("The source image does not have the same size as the destination image. Source size "
                               << aSrcView.SizeAlloc() << ", Destination size: " << SizeAlloc());
        }

        if (aSrcView.MemoryType() == ImageViewBase<T>::MemoryType_enum::CudaDefault)
        {
            aSrcView.CopyTo(*this);
        }
        else
        {
            throw INVALIDARGUMENT(
                aSrcView, "Unknown memory location for source ImageView. Cannot copy data from there to CUDA device.");
        }
    }

    /// <summary>
    /// Copy from this view to other view
    /// </summary>
    void operator>>(ImageViewBase<T> &aDstView) const
    {
        CopyTo(aDstView);
    }

    /// <summary>
    /// Copy from other view to this view
    /// </summary>
    void operator<<(const ImageViewBase<T> &aSrcView)
    {
        CopyFrom(aSrcView);
    }

    /// <summary>
    /// Copy from this view to other view only in ROI
    /// </summary>
    /// <param name="aDstView">Destination view</param>
    void CopyToRoi(ImageViewBase<T> &aDstView) const override
    {
        if (SizeRoi() != aDstView.SizeRoi())
        {
            throw ROIEXCEPTION(
                "The source image ROI does not have the same size as the destination image ROI. Source ROI size "
                << SizeRoi() << ", Destination size: " << aDstView.SizeRoi());
        }

        if (aDstView.MemoryType() == ImageViewBase<T>::MemoryType_enum::CudaDefault)
        {
            aDstView.CopyFromRoi(*this);
        }
        else
        {
            throw INVALIDARGUMENT(
                aDstView, "Unknown memory location for destination ImageView. Cannot copy data from host to there.");
        }
    }

    /// <summary>
    /// Copy from other view to this view only in ROI
    /// </summary>
    /// <param name="aSrcView">Source view</param>
    void CopyFromRoi(const ImageViewBase<T> &aSrcView) override
    {
        if (ImageViewBase<T>::SizeRoi() != aSrcView.SizeRoi())
        {
            throw ROIEXCEPTION(
                "The source image ROI does not have the same size as the destination image ROI. Source ROI size "
                << aSrcView.SizeRoi() << ", Destination size: " << ImageViewBase<T>::SizeRoi());
        }

        if (aSrcView.MemoryType() == ImageViewBase<T>::MemoryType_enum::CudaDefault)
        {
            aSrcView.CopyToRoi(*this);
        }
        else
        {
            throw INVALIDARGUMENT(aSrcView,
                                  "Unknown memory location for source ImageView. Cannot copy data from there to host.");
        }
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
        requires RealSignedVector<T>
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
            if (!(diff <= limit))
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
        requires RealUnsignedVector<T> && RealIntVector<T>
    {
        if (aOther.SizeRoi() != SizeRoi())
        {
            return false;
        }

        T limit(aMaxDiff);

        auto iterSrc1 = cbegin();
        for (const auto &elemSrc2 : aOther)
        {
            T diff{0};

            if (elemSrc2.Value().x > iterSrc1.Value().x)
            {
                diff.x = elemSrc2.Value().x - iterSrc1.Value().x;
            }
            if (elemSrc2.Value().x < iterSrc1.Value().x)
            {
                diff.x = iterSrc1.Value().x - elemSrc2.Value().x;
            }

            if constexpr (vector_active_size_v<T> > 1)
            {
                if (elemSrc2.Value().y > iterSrc1.Value().y)
                {
                    diff.y = elemSrc2.Value().y - iterSrc1.Value().y;
                }
                if (elemSrc2.Value().y < iterSrc1.Value().y)
                {
                    diff.y = iterSrc1.Value().y - elemSrc2.Value().y;
                }
            }
            if constexpr (vector_active_size_v<T> > 2)
            {
                if (elemSrc2.Value().z > iterSrc1.Value().z)
                {
                    diff.z = elemSrc2.Value().z - iterSrc1.Value().z;
                }
                if (elemSrc2.Value().z < iterSrc1.Value().z)
                {
                    diff.z = iterSrc1.Value().z - elemSrc2.Value().z;
                }
            }
            if constexpr (vector_active_size_v<T> > 3)
            {
                if (elemSrc2.Value().w > iterSrc1.Value().w)
                {
                    diff.w = elemSrc2.Value().w - iterSrc1.Value().w;
                }
                if (elemSrc2.Value().w < iterSrc1.Value().w)
                {
                    diff.w = iterSrc1.Value().w - elemSrc2.Value().w;
                }
            }

            if (!(diff <= limit))
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
    [[nodiscard]] bool IsSimilarIgnoringNAN(const ImageView<T> &aOther, remove_vector_t<T> aMaxDiff) const
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
            T src1 = iterSrc1.Value();
            T src2 = elemSrc2.Value();

            MakeNANandINFValid(src1.x, src2.x);
            if constexpr (vector_active_size_v<T> > 1)
            {
                MakeNANandINFValid(src1.y, src2.y);
            }
            if constexpr (vector_active_size_v<T> > 2)
            {
                MakeNANandINFValid(src1.z, src2.z);
            }
            if constexpr (vector_active_size_v<T> > 3)
            {
                MakeNANandINFValid(src1.w, src2.w);
            }

            T diff = T::Abs(src2 - src1);
            if (!(diff < limit))
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
    [[nodiscard]] bool IsSimilar(const ImageView<T> &aOther, complex_basetype_t<remove_vector_t<T>> aMaxDiff) const
        requires ComplexVector<T>
    {
        if (aOther.SizeRoi() != SizeRoi())
        {
            return false;
        }

        using pixelT           = complex_basetype_t<remove_vector_t<T>>;
        constexpr int channels = vector_active_size_v<T>;

        auto iterSrc1 = cbegin();
        for (const auto &elemSrc2 : aOther)
        {
            for (int channel = 0; channel < channels; channel++)
            {
                const pixelT real1 = elemSrc2.Value()[Channel(channel)].real;
                const pixelT imag1 = elemSrc2.Value()[Channel(channel)].imag;
                const pixelT real2 = iterSrc1.Value()[Channel(channel)].real;
                const pixelT imag2 = iterSrc1.Value()[Channel(channel)].imag;

                const pixelT diffreal = std::abs(real1 - real2);
                const pixelT diffimag = std::abs(imag1 - imag2);

                if (diffreal > aMaxDiff || diffimag > aMaxDiff)
                {
                    return false;
                }
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
    ImageView<TTo> &Convert(ImageView<TTo> &aDst) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && ConversionImplemented<T, TTo>;

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16: RoundingMode::NearestTiesAwayFromZero is NOT supported, for float32 to BFloat, on
    /// host only RoundingMode::NearestTiesToEven and RoundingMode::TowardZero are supported.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) &&
                ConversionRoundImplemented<T, TTo> && RealOrComplexFloatingVector<T>;

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -> float -> scale -> DstT
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && ConversionRoundScaleImplemented<T, TTo> &&
                (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) && (!std::same_as<TTo, Complex<float>>) &&
                (!std::same_as<TTo, Complex<double>>);
#pragma endregion
#pragma region Copy
    /// <summary>
    /// Copy image.
    /// </summary>
    ImageView<T> &Copy(ImageView<T> &aDst) const;

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    ImageView<T> &CopyMasked(ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const;

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst, Channel aDstChannel) const
        requires(vector_size_v<T> > 1) &&   //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy this single channel image to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(ImageView<TTo> &aDst, Channel aDstChannel) const
        requires(vector_size_v<T> == 1) &&  //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst) const
        requires(vector_size_v<T> > 1) &&    //
                (vector_size_v<TTo> == 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2) const
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3) const
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel4) const
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

    // the following are implemented in imageView_geometryTransforms_impl.h:
    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    ImageView<T> &Copy(ImageView<T> &aDst, const Vector2<int> &aLowerBorderSize, BorderType aBorder,
                       const T &aConstant = {0}) const;

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    ImageView<T> &Copy(ImageView<T> &aDst, const Pixel32fC2 &aDelta, InterpolationMode aInterpolation) const;

    /// <summary>
    /// Shift zero-frequency component to center of spectrum - same as fftshift operation in Matlab/Octave
    /// </summary>
    /// <param name="aDst">Destination image</param>
    ImageView<T> &FFTShift(ImageView<T> &aDst) const;

    /// <summary>
    /// Inverse of FFTShift - same as ifftshift operation in Matlab/Octave
    /// </summary>
    /// <param name="aDst">Destination image</param>
    ImageView<T> &IFFTShift(ImageView<T> &aDst) const;

#pragma endregion
#pragma region Dup
    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Dup(ImageView<TTo> &aDst) const
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
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, RoundingMode aRoundingMode) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
                RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
                RealOrComplexFloatingVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                          RoundingMode aRoundingMode) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
                RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          RoundingMode aRoundingMode) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexFloatingVector<TTo> && ScaleImplemented<T, TTo>
    ;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax, RoundingMode aRoundingMode) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

#pragma endregion
#pragma region Set
    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(const T &aConst);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    ImageView<T> &SetMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(remove_vector_t<T> aConst, Channel aChannel);
#pragma endregion
#pragma region Swap Channel
    /// <summary>
    /// Swap channels
    /// </summary>
    ImageView<T> &SwapChannel(ImageView<T> &aDst) const
        requires(vector_size_v<T> == 2);

    /// <summary>
    /// Swap channels (inplace)
    /// </summary>
    ImageView<T> &SwapChannel()
        requires(vector_size_v<T> == 2);

    /// <summary>
    /// Swap channels
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels) const
        requires((vector_active_size_v<TTo> <= vector_active_size_v<T>)) && //
                (vector_size_v<T> >= 3) &&                                  //
                (vector_size_v<TTo> >= 3) &&                                //
                (!has_alpha_channel_v<TTo>) &&                              //
                (!has_alpha_channel_v<T>) &&                                //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Swap channels (inplace)
    /// </summary>
    ImageView<T> &SwapChannel(const ChannelList<vector_active_size_v<T>> &aDstChannels)
        requires(vector_size_v<T> >= 3) && (!has_alpha_channel_v<T>);

    /// <summary>
    /// Swap channels (3-channel to 4-channel with additional value). If aDstChannels[i] == 3, channel i of aDst is set
    /// to aValue, if aDstChannels[i] > 3, channel i of aDst is kept unchanged.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                remove_vector_t<T> aValue) const
        requires(vector_size_v<T> == 3) &&          //
                (vector_active_size_v<TTo> == 4) && //
                (!has_alpha_channel_v<TTo>) &&      //
                (!has_alpha_channel_v<T>) &&        //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;
#pragma endregion

#pragma region FillRandom
    ImageView<T> &FillRandom();

    ImageView<T> &FillRandom(uint aSeed);

    ImageView<T> &FillRandomNormal(uint aSeed, double aMean, double aStd);
#pragma endregion

#pragma region Transpose
    /// <summary>
    /// Transpose image.
    /// </summary>
    ImageView<T> &Transpose(ImageView<T> &aDst) const
        requires NoAlpha<T>;
#pragma endregion
#pragma endregion

#pragma region Arithmetic functions
#pragma region Add
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const ImageView<T> &aSrc2, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Add(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Add(const T &aConst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Sub
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0) const
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

    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Mul
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const ImageView<T> &aSrc2, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Mul(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Mul(const T &aConst, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region MulScale
    ImageView<T> &MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const T &aConst, ImageView<T> &aDst) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const ImageView<T> &aSrc2)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScale(const T &aConst)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScaleMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScaleMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScaleMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    ImageView<T> &MulScaleMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;
#pragma endregion
#pragma region Div
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &Div(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &Div(const T &aConst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInv(const ImageView<T> &aSrc2)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInv(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                         RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInv(const T &aConst)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInv(const T &aConst, int aScaleFactor = 0,
                         RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven) const
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                               RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;

    ImageView<T> &DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
        requires RealOrComplexFloatingVector<T>;

    ImageView<T> &DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                               RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven)
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region AddSquare
    /// <summary>
    /// SrcDst += this^2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddSquare(ImageView<add_spw_output_for_t<T>> &aSrcDst) const;

    /// <summary>
    /// SrcDst += this^2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddSquareMasked(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                        const ImageView<Pixel8uC1> &aMask) const;
#pragma endregion
#pragma region AddProduct
    /// <summary>
    /// SrcDst += this * Src2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddProduct(const ImageView<T> &aSrc2,
                                                   ImageView<add_spw_output_for_t<T>> &aSrcDst) const;

    /// <summary>
    /// SrcDst += this * Src2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddProductMasked(const ImageView<T> &aSrc2,
                                                         ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                         const ImageView<Pixel8uC1> &aMask) const;
#pragma endregion
#pragma region AddWeighted
    /// <summary>
    /// Dst = this * alpha + Src2 * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aDst,
                                                    remove_vector_t<add_spw_output_for_t<T>> aAlpha) const;

    /// <summary>
    /// Dst = this * alpha + Src2 * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeightedMasked(const ImageView<T> &aSrc2,
                                                          ImageView<add_spw_output_for_t<T>> &aDst,
                                                          remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                          const ImageView<Pixel8uC1> &aMask) const;
    /// <summary>
    /// SrcDst = this * alpha + SrcDst * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                    remove_vector_t<add_spw_output_for_t<T>> aAlpha) const;

    /// <summary>
    /// SrcDst = this * alpha + SrcDst * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeightedMasked(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                          remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                          const ImageView<Pixel8uC1> &aMask) const;
#pragma endregion

#pragma region Abs
    ImageView<T> &Abs(ImageView<T> &aDst) const
        requires RealSignedVector<T>;

    ImageView<T> &Abs()
        requires RealSignedVector<T>;
#pragma endregion
#pragma region AbsDiff
    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealUnsignedVector<T>;

    ImageView<T> &AbsDiff(const T &aConst, ImageView<T> &aDst) const
        requires RealUnsignedVector<T>;

    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2)
        requires RealUnsignedVector<T>;

    ImageView<T> &AbsDiff(const T &aConst)
        requires RealUnsignedVector<T>;
#pragma endregion
#pragma region And
    ImageView<T> &And(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealIntVector<T>;

    ImageView<T> &And(const T &aConst, ImageView<T> &aDst) const
        requires RealIntVector<T>;

    ImageView<T> &And(const ImageView<T> &aSrc2)
        requires RealIntVector<T>;

    ImageView<T> &And(const T &aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region Not
    ImageView<T> &Not(ImageView<T> &aDst) const
        requires RealIntVector<T>;

    ImageView<T> &Not()
        requires RealIntVector<T>;
#pragma endregion
#pragma region Exp
    ImageView<T> &Exp(ImageView<T> &aDst) const
        requires RealOrComplexVector<T>;

    ImageView<T> &Exp()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Ln
    ImageView<T> &Ln(ImageView<T> &aDst) const
        requires RealOrComplexVector<T>;

    ImageView<T> &Ln()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region LShift
    ImageView<T> &LShift(uint aConst, ImageView<T> &aDst) const
        requires RealIntVector<T>;
    ImageView<T> &LShift(uint aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region Or
    ImageView<T> &Or(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealIntVector<T>;

    ImageView<T> &Or(const T &aConst, ImageView<T> &aDst) const
        requires RealIntVector<T>;

    ImageView<T> &Or(const ImageView<T> &aSrc2)
        requires RealIntVector<T>;

    ImageView<T> &Or(const T &aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region RShift
    ImageView<T> &RShift(uint aConst, ImageView<T> &aDst) const
        requires RealIntVector<T>;
    ImageView<T> &RShift(uint aConst)
        requires RealIntVector<T>;
#pragma endregion
#pragma region Sqr
    ImageView<T> &Sqr(ImageView<T> &aDst) const
        requires RealOrComplexVector<T>;

    ImageView<T> &Sqr()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Sqrt
    ImageView<T> &Sqrt(ImageView<T> &aDst) const
        requires RealOrComplexVector<T>;

    ImageView<T> &Sqrt()
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Xor
    ImageView<T> &Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealIntVector<T>;

    ImageView<T> &Xor(const T &aConst, ImageView<T> &aDst) const
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
    ImageView<T> &AlphaPremul(ImageView<T> &aDst) const
        requires FourChannelNoAlpha<T> && RealVector<T>;

    ImageView<T> &AlphaPremul()
        requires FourChannelNoAlpha<T> && RealVector<T>;

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst) const
        requires RealFloatingVector<T> && (!FourChannelAlpha<T>);

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst) const
        requires RealIntVector<T> && (!FourChannelAlpha<T>);

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha)
        requires RealFloatingVector<T> && (!FourChannelAlpha<T>);

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha)
        requires RealIntVector<T> && (!FourChannelAlpha<T>);

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst) const
        requires FourChannelAlpha<T>;

    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha)
        requires FourChannelAlpha<T>;
#pragma endregion

#pragma region AlphaComp
    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp) const
        requires(!FourChannelAlpha<T>) && RealVector<T>;

    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                            remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp) const
        requires RealVector<T>;
#pragma endregion

#pragma region Complex
    /// <summary>
    /// aDst = this * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires ComplexVector<T>;

    /// <summary>
    /// this = this * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2)
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = conj(this) (complex conjugate)
    /// </summary>
    ImageView<T> &Conj(ImageView<T> &aDst) const
        requires ComplexVector<T>;

    /// <summary>
    /// this = conj(this) (complex conjugate)
    /// </summary>
    ImageView<T> &Conj()
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = abs(this) (complex magnitude)<para/>
    /// If the source image ROI is of size (widthDestination / 2 + 1, heightDestination), the size of a R2C FFT with
    /// FFTW or CUFFT, then the result corresponds to the full spectrum with 0 frequency in the center.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Magnitude(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = abs(this)^2 (complex magnitude squared)<para/>
    /// If the source image ROI is of size (widthDestination / 2 + 1, heightDestination), the size of a R2C FFT with
    /// FFTW or CUFFT, then the result corresponds to the full spectrum with 0 frequency in the center.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &MagnitudeSqr(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = log(abs(this) + aOffset) (logarithm of complex magnitude)<para/>
    /// To avoid -INF in case of 0+0i, an offset is added to the magnitude value.<para/>
    /// If the source image ROI is of size (widthDestination / 2 + 1, heightDestination), the size of a R2C FFT with
    /// FFTW or CUFFT, then the result corresponds to the full spectrum with 0 frequency in the center.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &MagnitudeLog(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        complex_basetype_t<remove_vector_t<T>> aOffset) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = angle(this) (complex angle, atan2(imag, real))
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Angle(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = this.real (real component of complex value)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Real(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = this.imag (imaginary component of complex value)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Imag(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
        requires ComplexVector<T>;

    /// <summary>
    /// aDst.real = this, aDst.imag = 0 (converts real valued image to complex with imaginary part = 0)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst) const
        requires RealSignedVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// aDst.real = this, aDst.imag = aSrcImag (converts two real valued images to one complex image)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        const ImageView<T> &aSrcImag,
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst) const
        requires RealSignedVector<T> && (!FourChannelAlpha<T>);
#pragma endregion
#pragma endregion

#pragma region Filtering
#pragma region Fixed Filter
    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder, const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, BorderType aBorder,
                              const Roi &aAllowedReadRoi) const;

    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder) const;
    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
                              BorderType aBorder) const;

    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(has_alternative_filter_output_type_for_v<T>);
    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(has_alternative_filter_output_type_for_v<T>);

    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        const T &aConstant, BorderType aBorder) const
        requires(has_alternative_filter_output_type_for_v<T>);
    /// <summary>
    /// Applies a mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        BorderType aBorder) const
        requires(has_alternative_filter_output_type_for_v<T>);
#pragma endregion

#pragma region Separable Filter
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                  int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                                  const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                  int aFilterSize, int aFilterCenter, BorderType aBorder,
                                  const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                  int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder) const;
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                  int aFilterSize, int aFilterCenter, BorderType aBorder) const;
#pragma endregion
#pragma region Column Filter

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                               int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder) const;
    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                               int aFilterSize, int aFilterCenter, BorderType aBorder) const;
    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                               int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                               const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                               int aFilterSize, int aFilterCenter, BorderType aBorder,
                               const Roi &aAllowedReadRoi) const;

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &ColumnWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &ColumnWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder) const
        requires RealVector<T>;

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &ColumnWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &ColumnWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
#pragma endregion
#pragma region Row Filter
    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                            int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder) const;
    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                            int aFilterSize, int aFilterCenter, BorderType aBorder) const;

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                            int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                            int aFilterSize, int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi) const;

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &RowWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &RowWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder) const
        requires RealVector<T>;

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &RowWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &RowWindowSum(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
#pragma endregion
#pragma region Box Filter
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                            BorderType aBorder) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, const T &aConstant,
        BorderType aBorder) const
        requires RealIntVector<T>;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea,
        BorderType aBorder) const
        requires RealIntVector<T>;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, const T &aConstant,
        BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealIntVector<T>;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
        const Roi &aAllowedReadRoi) const
        requires RealIntVector<T>;

    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                 const T &aConstant, BorderType aBorder) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);
    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                 BorderType aBorder) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);

    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                 const T &aConstant, BorderType aBorder,
                                                 const Roi &aAllowedReadRoi) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);
    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                 BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);
#pragma endregion
#pragma region Min/Max Filter
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                            BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                            BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

#pragma endregion
#pragma region Median Filter
    /// <summary>
    /// Median filter
    /// </summary>
    ImageView<T> &MedianFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Median filter
    /// </summary>
    ImageView<T> &MedianFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                               const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

#pragma endregion
#pragma region Wiener Filter
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, const T &aConstant, BorderType aBorder,
                               const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, BorderType aBorder,
                               const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

#pragma endregion
#pragma region Threshold Adaptive Box Filter
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the souce pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                             const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                             const T &aValLE, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the souce pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                             const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                             const T &aValLE, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the souce pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                             const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                             const T &aValLE, const T &aConstant, BorderType aBorder,
                                             const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the souce pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                             const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                             const T &aValLE, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

#pragma endregion
#pragma region Filter
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const;
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                         const FilterArea &aFilterArea, BorderType aBorder) const;
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                         const Roi &aAllowedReadRoi) const;
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                         const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const;

#pragma endregion
#pragma region Bilateral Gauss Filter
    /// <summary>
    /// This function pre-computes the geometrical distance coefficients for bilateral Gauss filtering. The result of
    /// this function can be passed to BilateralGaussFilter.
    /// </summary>
    /// <param name="aPreCompGeomDistCoeff"></param>
    /// <param name="aFilterArea"></param>
    /// <param name="aPosSquareSigma"></param>
    /// <param name="aStreamCtx"></param>
    void PrecomputeBilateralGaussFilter(float *aPreCompGeomDistCoeff, const FilterArea &aFilterArea,
                                        float aPosSquareSigma) const
        requires RealVector<T> && (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, const T &aConstant,
                                       BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                       BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, const T &aConstant,
                                       BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, BorderType aBorder,
                                       const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
                                       const T &aConstant, BorderType aBorder) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
                                       BorderType aBorder) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
                                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                       const float *aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
                                       BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
#pragma endregion
#pragma region Gradient Vector
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY, ImageView<Pixel16sC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder,
                             const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY, ImageView<Pixel16sC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY, ImageView<Pixel32fC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder,
                             const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY, ImageView<Pixel32fC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                              ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                              ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize, BorderType aBorder,
                              const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                              ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                              ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize, BorderType aBorder,
                              const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                               ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                               ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                               ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                               ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY, ImageView<Pixel16sC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY, ImageView<Pixel16sC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY, ImageView<Pixel32fC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Sobel filters. Output images are only computed if the provided
    /// pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::SobelVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY, ImageView<Pixel32fC1> &aDstMag,
                             ImageView<Pixel32fC1> &aDstAngle, ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm,
                             MaskSize aMaskSize, BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                              ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                              ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                              BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                              ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Computes the gradients for each pixel using fixed Scharr filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: In contrast to Sobel and Prewitt variants, the Scharr-definition for the X (vertical) gradient filter
    /// kernel is identical compared to the definition in FixedFilter::ScharrVert in order to obtain identical results
    /// as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                              ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                              ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                              BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                               ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               const T &aConstant, BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                               ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                               ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               const T &aConstant, BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Computes the gradients for each pixel using fixed Prewitt filters. Output images are only computed if the
    /// provided pointer is not nullptr. If an output is set to nullptr, the result is skipped.<para/>
    /// Note: The definition for the X (vertical) gradient filter kernel is mirrored compared to the definition in
    /// FixedFilter::PrewittVert in order to obtain identical results as in NPP!
    /// </summary>
    /// <param name="aDstX">the X (vertical) gradient</param>
    /// <param name="aDstY">the Y (horizontal) gradient</param>
    /// <param name="aDstMag">the gradient magnitude</param>
    /// <param name="aDstAngle">the orientation computed using atan2</param>
    /// <param name="aDstCovariance">the covariance matrix stored in a Vector4 structure for
    /// convenience (.x is the x^2 gradient, .y is the y^2 gradient, .z and .w are x*y gradient).</param>
    /// <param name="aNorm">The norm used to compute aDstMag</param>
    /// <param name="aMaskSize">Mask size for the fixed filter</param>
    void GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                               ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                               ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                               BorderType aBorder) const
        requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
                 std::same_as<remove_vector_t<T>, float>);
#pragma endregion
#pragma region Unsharp Filter
    /// <summary>
    /// Smoothes the orginal images using the user defined filter aFilter (coefficients should sum up to 1) and then
    /// subtracts the result from the original to obtain a high-pass filtered image. After thresholding and weighting,
    /// the result is added to the original image using the following pseudo-formula:<para/>
    /// HighPass = Image - Filter(Image)<para/>
    /// Result = Image + nWeight * HighPass * (| HighPass | &gt;= nThreshold) <para/>
    /// where nWeight is the amount, nThreshold is the threshold, and &gt;= indicates a Boolean operation, 1 if true, or
    /// 0 otherwise.
    /// </summary>
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Smoothes the orginal images using the user defined filter aFilter (coefficients should sum up to 1) and then
    /// subtracts the result from the original to obtain a high-pass filtered image. After thresholding and weighting,
    /// the result is added to the original image using the following pseudo-formula:<para/>
    /// HighPass = Image - Filter(Image)<para/>
    /// Result = Image + nWeight * HighPass * (| HighPass | &gt;= nThreshold) <para/>
    /// where nWeight is the amount, nThreshold is the threshold, and &gt;= indicates a Boolean operation, 1 if true, or
    /// 0 otherwise.
    /// </summary>
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Smoothes the orginal images using the user defined filter aFilter (coefficients should sum up to 1) and then
    /// subtracts the result from the original to obtain a high-pass filtered image. After thresholding and weighting,
    /// the result is added to the original image using the following pseudo-formula:<para/>
    /// HighPass = Image - Filter(Image)<para/>
    /// Result = Image + nWeight * HighPass * (| HighPass | &gt;= nThreshold) <para/>
    /// where nWeight is the amount, nThreshold is the threshold, and &gt;= indicates a Boolean operation, 1 if true, or
    /// 0 otherwise.
    /// </summary>
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Smoothes the orginal images using the user defined filter aFilter (coefficients should sum up to 1) and then
    /// subtracts the result from the original to obtain a high-pass filtered image. After thresholding and weighting,
    /// the result is added to the original image using the following pseudo-formula:<para/>
    /// HighPass = Image - Filter(Image)<para/>
    /// Result = Image + nWeight * HighPass * (| HighPass | &gt;= nThreshold) <para/>
    /// where nWeight is the amount, nThreshold is the threshold, and &gt;= indicates a Boolean operation, 1 if true, or
    /// 0 otherwise.
    /// </summary>
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
#pragma endregion
#pragma region Harris Corner Response
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK,
                                                float aScale, const T &aConstant, BorderType aBorder) const
        requires std::same_as<T, Pixel32fC4>;
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK,
                                                float aScale, BorderType aBorder) const
        requires std::same_as<T, Pixel32fC4>;
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK,
                                                float aScale, const T &aConstant, BorderType aBorder,
                                                const Roi &aAllowedReadRoi) const
        requires std::same_as<T, Pixel32fC4>;
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK,
                                                float aScale, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires std::same_as<T, Pixel32fC4>;

#pragma endregion
#pragma region Canny edge
    /// <summary>
    /// For an gradient magnitude image and an gradient orientation image obtained from one of the gradient vector
    /// functions, this function performs canny edge detection.
    /// </summary>
    ImageView<Pixel8uC1> &CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                    ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold) const
        requires std::same_as<T, Pixel16sC1> || std::same_as<T, Pixel32fC1>;
    /// <summary>
    /// For an gradient magnitude image and an gradient orientation image obtained from one of the gradient vector
    /// functions, this function performs canny edge detection.
    /// </summary>
    ImageView<Pixel8uC1> &CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                    ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold,
                                    const Roi &aAllowedReadRoi) const
        requires std::same_as<T, Pixel16sC1> || std::same_as<T, Pixel32fC1>;

#pragma endregion
#pragma endregion

#pragma region Geometric Transforms
#pragma region Affine
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                             InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                             Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                             InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                           const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                           const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                           const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                           BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst3, const AffineTransformation<double> &aAffine,
                           InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                           Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst3, const AffineTransformation<double> &aAffine,
                           InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                           const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                           const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from source image to destination image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                           const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                           BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpAffineBack(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                 InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                 Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to
    /// use must be provided.
    /// </summary>
    ImageView<T> &WarpAffineBack(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                 InterpolationMode aInterpolation, BorderType aBorder,
                                 Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                               const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                               const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to
    /// use must be provided.
    /// </summary>
    static void WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                               const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                               BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                               const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                               const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to
    /// use must be provided.
    /// </summary>
    static void WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                               const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                               BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpAffineBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
        ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
        ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
        BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
    /// <summary>
    /// WarpAffine, the transformation aAffine defines the mapping from destination image to source image.<para/>
    /// Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to
    /// use must be provided.
    /// </summary>
    static void WarpAffineBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
        ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
        ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
        Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
#pragma endregion

#pragma region Perspective
    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                  InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                  Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                  InterpolationMode aInterpolation, BorderType aBorder,
                                  Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                                const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                                BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                                const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                                BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspective(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
        ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
        ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
        BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from source image to destination
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspective(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
        ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
        ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
        Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpPerspectiveBack(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                      InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                      Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &WarpPerspectiveBack(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                      InterpolationMode aInterpolation, BorderType aBorder,
                                      Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                    const PerspectiveTransformation<double> &aPerspective,
                                    InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                    Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                    const PerspectiveTransformation<double> &aPerspective,
                                    InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspectiveBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
        BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                    const PerspectiveTransformation<double> &aPerspective,
                                    InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspectiveBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
        ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
        ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
        BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// WarpPerspective, the transformation aPerspective defines the mapping from destination image to source
    /// image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi differs:
    /// For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and remain as
    /// is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching outside the
    /// roi.<para/>
    /// For all other BorderType, the pixels outside the source image roi are filled (and interpolated) according to the
    /// chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>
    /// For BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void WarpPerspectiveBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
        ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
        ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
        Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
#pragma endregion

#pragma region Rotate
    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                         InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;
    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, double aAngleInDeg, const Vector2<double> &aShift,
                       InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                       Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;
    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, double aAngleInDeg, const Vector2<double> &aShift,
                       InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                       double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
    /// <summary>
    /// Rotate, the transformation defines the mapping from source image to destination image with a counter-clock
    /// rotation around pixel(0,0) and a shift after rotation.<para/> Depending on BorderType, the behavior for pixels
    /// that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP: pixels
    /// outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                       double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
#pragma endregion

#pragma region Resize
    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    ImageView<T> &Resize(ImageView<T> &aDst, InterpolationMode aInterpolation) const;

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       InterpolationMode aInterpolation)
        requires TwoChannel<T>;

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, InterpolationMode aInterpolation)
        requires ThreeChannel<T>;

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                       InterpolationMode aInterpolation)
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// Returns a shift to be used in MPP resize method that matches to the result given by the NPP Resize-function.
    /// </summary>
    Vec2d ResizeGetNPPShift(ImageView<T> &aDst) const;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                         InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       const Vector2<double> &aScale, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       const Vector2<double> &aScale, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Vector2<double> &aScale,
                       const Vector2<double> &aShift, InterpolationMode aInterpolation, const T &aConstant,
                       BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Vector2<double> &aScale,
                       const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                       Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                       const Vector2<double> &aScale, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// Resize.<para/>As in ResizeSqrPixel in NPP. When mapping integer pixel coordinates from integer to floating
    /// point, in MPP the definition is as following: The integer pixel coordinate corresponds to the center of the
    /// pixel surface that thus has an extent for a pixel i from [i-0.5 .. i+0.5[ (excluding the right border). The
    /// entire valid image area then ranges from [-0.5 to width-0.5[ <para/>
    /// When rescaling, an additional shift is applied, so that the area from source image [-0.5 .. srcWidth-0.5[
    /// exactly matches
    /// [-0.5 .. dstWidth-0.5[.<para/> This shift is given by (as in NPP):<para/> InvScaleFactor = 1 / aScale;<para/>
    /// AdjustedShift  = aShift * InvScaleFactor + ((1 - InvScaleFactor) * 0.5);<para/>
    /// The output pixel with integer coordinate (X,Y) is then mapped to the source pixel:<para/>
    /// SrcX = InvScaleFactor.x * X - AdjustedShift.x;<para/>
    /// SrcY = InvScaleFactor.y * Y - AdjustedShift.y;<para/>
    /// Depending on BorderType, the behavior for
    /// pixels that fall outside the source image roi differs: For BorderType::None, the behavior is similiar to NPP:
    /// pixels outside the roi are not written to and remain as is, though at the image border, BorderType::Replicate is
    /// applied for interpolation kernels reaching outside the roi.<para/> For all other BorderType, the pixels outside
    /// the source image roi are filled (and interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    static void Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                       const Vector2<double> &aScale, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                       BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
#pragma endregion

#pragma region Mirror
    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    ImageView<T> &Mirror(ImageView<T> &aDst, MirrorAxis aAxis) const;

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    ImageView<T> &Mirror(MirrorAxis aAxis);
#pragma endregion

#pragma region Remap
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    ImageView<T> &Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                        InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                        Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    ImageView<T> &Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                        InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    ImageView<T> &Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                        const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                        const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi()) const;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    ImageView<T> &Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                        const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                        BorderType aBorder, Roi aAllowedReadRoi = Roi()) const;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation, const T &aConstant,
                      BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation, BorderType aBorder,
                      Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                      InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                      Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                      InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires TwoChannel<T>;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC2> &aCoordinateMap,
                      InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                      Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC2> &aCoordinateMap,
                      InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC1> &aCoordinateMapX,
                      const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                      const T &aConstant, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC1> &aCoordinateMapX,
                      const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                      BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires ThreeChannel<T>;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                      const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation, const T &aConstant,
                      BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                      const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation, BorderType aBorder,
                      Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/> For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                      const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                      InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                      Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
    /// <summary>
    /// Remap, for each destination image pixel, the coordinate map contains its mapped floating point coordinate in the
    /// source image.<para/> Depending on BorderType, the behavior for pixels that fall outside the source image roi
    /// differs: For BorderType::None, the behavior is similiar to NPP: pixels outside the roi are not written to and
    /// remain as is, though at the image border, BorderType::Replicate is applied for interpolation kernels reaching
    /// outside the roi.<para/> For all other BorderType, the pixels outside the source image roi are filled (and
    /// interpolated) according to the chosen BorderType.<para/>
    /// For BorderType::Mirror, BorderType::MirrorReplicate and BorderType::Wrap, only pixels once the width or
    /// height of the source image roi on each side is allowed for pixels outside the original roi. For transforms that
    /// fall outside this expanded area, the pixel value is not defined. <para/>For BorderType::Constant, the constant
    /// value to use must be provided.
    /// </summary>
    static void Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                      const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                      ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                      const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                      InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi = Roi())
        requires FourChannelNoAlpha<T>;
#pragma endregion
#pragma endregion

#pragma region Morphology
#pragma region No mask Erosion/Dilation
    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                           const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                          const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                          const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                           BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                          BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;

#pragma endregion
#pragma region Erosion
    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea, const T &aConstant,
                          BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea, BorderType aBorder,
                          const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                              const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                              const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                              const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea, const T &aConstant,
                          BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                          BorderType aBorder) const
        requires RealVector<T>;

    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                              const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                              const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma region Dilation
    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                           const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                           BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                               const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                               const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                               const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                           const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                           BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                               const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                               const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma region Open
    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                       BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                       const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                       BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma region Close
    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                        BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                        const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                        BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma region TopHat
    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                         const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                         BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                         const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                         BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma region BlackHat
    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                           const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                           const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                           const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                           const FilterArea &aFilterArea, BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma region Morphology Gradient
    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;
    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires RealVector<T>;

    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     const T &aConstant, BorderType aBorder) const
        requires RealVector<T>;
    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     BorderType aBorder) const
        requires RealVector<T>;
#pragma endregion
#pragma endregion

#pragma region Statistics
#pragma region AverageError
    void AverageError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                      double &aDstScalar) const
        requires(vector_active_size_v<T> > 1);

    void AverageErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                            double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> > 1);

    void AverageError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires(vector_active_size_v<T> == 1);

    void AverageErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                            const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion
#pragma region AverageRelativeError
    void AverageRelativeError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                              double &aDstScalar) const
        requires(vector_active_size_v<T> > 1);

    void AverageRelativeErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> > 1);

    void AverageRelativeError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires(vector_active_size_v<T> == 1);

    void AverageRelativeErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region DotProduct
    void DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                    double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                    c_double &aDstScalar) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                          c_double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);

    void DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                          const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MSE
    void MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
             c_double &aDstScalar) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                   c_double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);

    void MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                   const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                   const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MaximumError
    void MaximumError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                      double &aDstScalar) const
        requires(vector_active_size_v<T> > 1);

    void MaximumErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                            double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> > 1);

    void MaximumError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires(vector_active_size_v<T> == 1);

    void MaximumErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                            const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MaximumRelativeError
    void MaximumRelativeError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                              double &aDstScalar) const
        requires(vector_active_size_v<T> > 1);

    void MaximumRelativeErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> > 1);

    void MaximumRelativeError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires(vector_active_size_v<T> == 1);

    void MaximumRelativeErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region NormDiffInf
    void NormDiffInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                     double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormDiffInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                           double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormDiffInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormDiffInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                           const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormDiffL1
    void NormDiffL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                    double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormDiffL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormDiffL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormDiffL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormDiffL2
    void NormDiffL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                    double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormDiffL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormDiffL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormDiffL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region NormRelInf
    void NormRelInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                    double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormRelInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormRelInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormRelInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormRelL1
    void NormRelL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                   double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormRelL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                         double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormRelL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormRelL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                         const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormRelL2
    void NormRelL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                   double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormRelL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                         double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormRelL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormRelL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                         const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region PSNR
    void PSNR(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
              double aValueRange) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void PSNR(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst, double aValueRange) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region NormInf
    void NormInf(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormInfMasked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                       const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormInf(same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormInfMasked(same_vector_size_different_type_t<T, double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormL1
    void NormL1(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormL1Masked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                      const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormL1(same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormL1Masked(same_vector_size_different_type_t<T, double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormL2
    void NormL2(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormL2Masked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                      const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void NormL2(same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void NormL2Masked(same_vector_size_different_type_t<T, double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region Sum
    void Sum(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void Sum(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void SumMasked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void SumMasked(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void Sum(same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void Sum(same_vector_size_different_type_t<T, c_double> &aDst) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);

    void SumMasked(same_vector_size_different_type_t<T, double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void SumMasked(same_vector_size_different_type_t<T, c_double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region Mean
    void Mean(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void Mean(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void MeanMasked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                    const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MeanMasked(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar,
                    const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void Mean(same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void Mean(same_vector_size_different_type_t<T, c_double> &aDst) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);

    void MeanMasked(same_vector_size_different_type_t<T, double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MeanMasked(same_vector_size_different_type_t<T, c_double> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MeanStd
    void MeanStd(same_vector_size_different_type_t<T, double> &aMean,
                 same_vector_size_different_type_t<T, double> &aStd, double &aMeanScalar, double &aStdScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MeanStd(same_vector_size_different_type_t<T, c_double> &aMean,
                 same_vector_size_different_type_t<T, double> &aStd, c_double &aMeanScalar, double &aStdScalar) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void MeanStdMasked(same_vector_size_different_type_t<T, double> &aMean,
                       same_vector_size_different_type_t<T, double> &aStd, double &aMeanScalar, double &aStdScalar,
                       const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MeanStdMasked(same_vector_size_different_type_t<T, c_double> &aMean,
                       same_vector_size_different_type_t<T, double> &aStd, c_double &aMeanScalar, double &aStdScalar,
                       const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> > 1);

    void MeanStd(same_vector_size_different_type_t<T, double> &aMean,
                 same_vector_size_different_type_t<T, double> &aStd) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MeanStd(same_vector_size_different_type_t<T, c_double> &aMean,
                 same_vector_size_different_type_t<T, double> &aStd) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);

    void MeanStdMasked(same_vector_size_different_type_t<T, double> &aMean,
                       same_vector_size_different_type_t<T, double> &aStd, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MeanStdMasked(same_vector_size_different_type_t<T, c_double> &aMean,
                       same_vector_size_different_type_t<T, double> &aStd, const ImageView<Pixel8uC1> &aMask) const
        requires ComplexVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region CountInRange
    void CountInRange(const T &aLowerLimit, const T &aUpperLimit, same_vector_size_different_type_t<T, size_t> &aDst,
                      size_t &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                            same_vector_size_different_type_t<T, size_t> &aDst, size_t &aDstScalar,
                            const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                      same_vector_size_different_type_t<T, size_t> &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                            same_vector_size_different_type_t<T, size_t> &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region QualityIndex
    void QualityIndex(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
        requires RealVector<T>;
#pragma endregion

#pragma region SSIM
    void SSIM(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst, double aDynamicRange = 1.0,
              double aK1 = 0.01,       // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
              double aK2 = 0.03) const // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        requires RealVector<T>;
#pragma endregion

#pragma region MSSSIM

    /// <summary>
    /// Computes the Multi-Scale-SSIM of two images.
    /// Note: This implementation differs slightly from NPP as the exact parameters used are unknown. Here we follow the
    /// reference matlab implementation provided here: https://ece.uwaterloo.ca/~z70wang/research/ssim/. The only
    /// difference is in the filtering steps for image borders where MPP applies replication.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per channel result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aDynamicRange">The value range of the image. Typically this is 2^BitsPerPixel - 1.</param>
    /// <param name="aK1">Stabilisation constant 1, Default=0.01</param>
    /// <param name="aK2">Stabilisation constant 2, Default=0.03</param>
    /// <param name="aStreamCtx"></param>
    void MSSSIM(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                double aDynamicRange = 1.0, double aK1 = 0.01, double aK2 = 0.03) const
        requires RealVector<T>;
#pragma endregion

#pragma region Min
    void Min(T &aDst, remove_vector_t<T> &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void MinMasked(T &aDst, remove_vector_t<T> &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void Min(T &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void MinMasked(T &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region Max
    void Max(T &aDst, remove_vector_t<T> &aDstScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void MaxMasked(T &aDst, remove_vector_t<T> &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void Max(T &aDst) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    void MaxMasked(T &aDst, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MinMax
    void MinMax(T &aDstMin, T &aDstMax, remove_vector_t<T> &aDstMinScalar, remove_vector_t<T> &aDstMaxScalar) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MinMaxMasked(T &aDstMin, T &aDstMax, remove_vector_t<T> &aDstMinScalar, remove_vector_t<T> &aDstMaxScalar,
                      const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void MinMax(T &aDstMin, T &aDstMax) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MinMaxMasked(T &aDstMin, T &aDstMax, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MinIndex
    void MinIndex(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                  same_vector_size_different_type_t<T, int> &aDstIndexY, remove_vector_t<T> &aDstMinScalar,
                  Vector3<int> &aDstScalarIdx) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MinIndexMasked(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                        same_vector_size_different_type_t<T, int> &aDstIndexY, remove_vector_t<T> &aDstMinScalar,
                        Vector3<int> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void MinIndex(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                  same_vector_size_different_type_t<T, int> &aDstIndexY) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MinIndexMasked(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                        same_vector_size_different_type_t<T, int> &aDstIndexY, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MaxIndex
    void MaxIndex(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                  same_vector_size_different_type_t<T, int> &aDstIndexY, remove_vector_t<T> &aDstMaxScalar,
                  Vector3<int> &aDstScalarIdx) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MaxIndexMasked(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                        same_vector_size_different_type_t<T, int> &aDstIndexY, remove_vector_t<T> &aDstMaxScalar,
                        Vector3<int> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    void MaxIndex(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                  same_vector_size_different_type_t<T, int> &aDstIndexY) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MaxIndexMasked(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                        same_vector_size_different_type_t<T, int> &aDstIndexY, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MinMaxIndex
    void MinMaxIndex(T &aDstMin, T &aDstMax, IndexMinMax aDstIdx[vector_active_size_v<T>],
                     remove_vector_t<T> &aDstMinScalar, remove_vector_t<T> &aDstMaxScalar,
                     IndexMinMaxChannel &aDstScalarIdx) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MinMaxIndexMasked(T &aDstMin, T &aDstMax, IndexMinMax aDstIdx[vector_active_size_v<T>],
                           remove_vector_t<T> &aDstMinScalar, remove_vector_t<T> &aDstMaxScalar,
                           IndexMinMaxChannel &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    void MinMaxIndex(T &aDstMin, T &aDstMax, IndexMinMax &aDstIdx) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    void MinMaxIndexMasked(T &aDstMin, T &aDstMax, IndexMinMax &aDstIdx, const ImageView<Pixel8uC1> &aMask) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MinEvery
    ImageView<T> &MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealVector<T>;

    ImageView<T> &MinEvery(const ImageView<T> &aSrc2)
        requires RealVector<T>;
#pragma endregion
#pragma region MaxEvery
    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
        requires RealVector<T>;

    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2)
        requires RealVector<T>;
#pragma endregion

#pragma region Integral
    ImageView<same_vector_size_different_type_t<T, int>> &Integral(
        ImageView<same_vector_size_different_type_t<T, int>> &aDst,
        const same_vector_size_different_type_t<T, int> &aVal) const
        requires RealIntVector<T>;

    ImageView<same_vector_size_different_type_t<T, float>> &Integral(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        const same_vector_size_different_type_t<T, float> &aVal) const
        requires RealVector<T> && (!std::same_as<double, remove_vector_t<T>>);

    ImageView<same_vector_size_different_type_t<T, long64>> &Integral(
        ImageView<same_vector_size_different_type_t<T, long64>> &aDst,
        const same_vector_size_different_type_t<T, long64> &aVal) const
        requires RealIntVector<T>;

    ImageView<same_vector_size_different_type_t<T, double>> &Integral(
        ImageView<same_vector_size_different_type_t<T, double>> &aDst,
        const same_vector_size_different_type_t<T, double> &aVal) const
        requires RealVector<T>;

    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                     const same_vector_size_different_type_t<T, int> &aVal,
                     const same_vector_size_different_type_t<T, int> &aValSqr) const
        requires RealIntVector<T>;

    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                     const same_vector_size_different_type_t<T, int> &aVal,
                     const same_vector_size_different_type_t<T, long64> &aValSqr) const
        requires RealIntVector<T>;

    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                     const same_vector_size_different_type_t<T, float> &aVal,
                     const same_vector_size_different_type_t<T, double> &aValSqr) const
        requires RealVector<T> && (!std::same_as<double, remove_vector_t<T>>);

    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, double>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                     const same_vector_size_different_type_t<T, double> &aVal,
                     const same_vector_size_different_type_t<T, double> &aValSqr) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea) const
        requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea) const
        requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea) const
        requires(std::same_as<remove_vector_t<T>, float>) && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, double>> &aDst, const FilterArea &aFilterArea) const
        requires(std::same_as<remove_vector_t<T>, double>) && NoAlpha<T>;
#pragma endregion

#pragma region Histogram
    /// <summary>
    /// Compute levels with even distribution
    /// </summary>
    /// <param name="aHPtrLevels">A host pointer to array which receives the levels being computed.
    /// The array needs to be of size aLevels.</ param>
    /// <param name="aLevels">The number of levels being computed. aLevels must be at least 2</param>
    /// <param name="aLowerLevel">Lower boundary value of the lowest level.</param>
    /// <param name="aUpperLevel">Upper boundary value of the greatest level.</param>
    void EvenLevels(hist_even_level_types_for_t<T> *aHPtrLevels, int aLevels,
                    const hist_even_level_types_for_t<T> &aLowerLevel,
                    const hist_even_level_types_for_t<T> &aUpperLevel)
        requires RealVector<T>;

    /// <summary>
    /// The aLowerLevel (inclusive) and aUpperLevel (exclusive) define the boundaries of the range,
    /// which are evenly segmented into aHist.Size() bins.
    /// </summary>
    void HistogramEven(same_vector_size_different_type_t<T, int> *aHist, int aHistBinCount,
                       const hist_even_level_types_for_t<T> &aLowerLevel,
                       const hist_even_level_types_for_t<T> &aUpperLevel)
        requires RealVector<T>;

    /// <summary>
    /// Computes the histogram of an image within specified ranges.
    /// </summary>
    void HistogramRange(same_vector_size_different_type_t<T, int> *aHist, int aHistBinCount,
                        hist_range_types_for_t<T> *aLevels)
        requires RealVector<T>;
#pragma endregion

#pragma region Cross Correlation
    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                            const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);
    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                            BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      const T &aConstant, BorderType aBorder,
                                                      const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                    const T &aConstant, BorderType aBorder,
                                                    const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                    BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                       const T &aConstant, BorderType aBorder,
                                                       const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                       BorderType aBorder, const Roi &aAllowedReadRoi) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                            const T &aConstant, BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);
    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                            BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      const T &aConstant, BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                    const T &aConstant, BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                    BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                       const T &aConstant, BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                       BorderType aBorder) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

#pragma endregion

#pragma region RadialProfile

    /// <summary>
    /// Radial profile. Circular around provided center point, the center does not have to be on the image.
    /// </summary>
    /// <param name="aProfileCount">The number of pixels that have the distance of the array index to the center
    /// point.</param>
    /// <param name="aProfileSum">Per-channel sum, one element per radius. Must be of the same size as
    /// aProfileCount.</param>
    /// <param name="aProfileSumSqr">Per-channel squared sum, if provided the array must have the same size as
    /// aProfileSum, can be nullptr.</param>
    /// <param name="aCenter">The center of the circular average.</param>
    /// <param name="aStreamCtx"></param>
    void RadialProfile(int *aProfileCount, same_vector_size_different_type_t<T, float> *aProfileSum,
                       same_vector_size_different_type_t<T, float> *aProfileSumSqr, int aProfileSize,
                       const Vec2f &aCenter) const
        requires RealVector<T> && (!std::same_as<remove_vector_t<T>, double>);

    /// <summary>
    /// Radial profile. Circular around provided center point, the center does not have to be on the image.
    /// </summary>
    /// <param name="aProfileCount">The number of pixels that have the distance of the array index to the center
    /// point.</param>
    /// <param name="aProfileSum">Per-channel sum, one element per radius. Must be of the same size as
    /// aProfileCount.</param>
    /// <param name="aProfileSumSqr">Per-channel squared sum, if provided the array must have the same size as
    /// aProfileSum, can be nullptr.</param>
    /// <param name="aCenter">The center of the circular average.</param>
    /// <param name="aRadiusRatio">Ratio of orientation axis to orthogonal axis. (i.e. 1.5 means stretched 1.5x)</param>
    /// <param name="aAngleInRad">Orientation of the ellipse's orthogonal axis, clockwise in radians with 0.0 being
    /// vertical.</param>
    /// <param name="aStreamCtx"></param>
    void RadialProfile(int *aProfileCount, same_vector_size_different_type_t<T, float> *aProfileSum,
                       same_vector_size_different_type_t<T, float> *aProfileSumSqr, int aProfileSize,
                       const Vec2f &aCenter, float aRadiusRatio, float aAngleInRad) const
        requires RealVector<T> && (!std::same_as<remove_vector_t<T>, double>);
#pragma endregion
#pragma endregion

#pragma region Threshold and Compare
#pragma region Compare
    ImageView<Pixel8uC1> &Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst) const;

    ImageView<Pixel8uC1> &Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst) const;

    ImageView<Pixel8uC1> &Compare(CompareOp aCompare, ImageView<Pixel8uC1> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        const ImageView<T> &aSrc2, CompareOp aCompare,
        ImageView<same_vector_size_different_type_t<T, byte>> &aDst) const
        requires(vector_size_v<T> > 1);

    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        const T &aConst, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst) const
        requires(vector_size_v<T> > 1);

    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst) const
        requires RealOrComplexFloatingVector<T> && (vector_size_v<T> > 1);

    ImageView<Pixel8uC1> &CompareEqEps(const ImageView<T> &aSrc2, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                       ImageView<Pixel8uC1> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    ImageView<Pixel8uC1> &CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                       ImageView<Pixel8uC1> &aDst) const
        requires RealOrComplexFloatingVector<T>;
#pragma endregion
#pragma region Threshold
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare)
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold)
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold)
        requires RealVector<T>;

    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare)
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue)
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue)
        requires RealVector<T>;
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT,
                                ImageView<T> &aDst) const
        requires RealVector<T>;
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT)
        requires RealVector<T>;
#pragma endregion
#pragma region ReplaceIf

    /// <summary>
    /// aDst pixel is set to aValue if this and aSrc2 fulfill aCompare, this otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue, ImageView<T> &aDst) const;

    /// <summary>
    /// aDst pixel is set to aValue if this and aConst fulfill aCompare, this otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue, ImageView<T> &aDst) const;

    /// <summary>
    /// aDst pixel is set to aValue if this fulfills aCompare (for floating point checks, e.g. isinf()), this
    /// otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(CompareOp aCompare, const T &aValue, ImageView<T> &aDst) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// A pixel is set to aValue if this and aSrc2 fulfill aCompare, this otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue);

    /// <summary>
    /// A pixel is set to aValue if this and aConst fulfill aCompare, this otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue);

    /// <summary>
    /// A pixel is set to aValue if this fulfills aCompare (for floating point checks, e.g. isinf()), this
    /// otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(CompareOp aCompare, const T &aValue)
        requires RealOrComplexFloatingVector<T>;
#pragma endregion
#pragma endregion

#pragma region ColorConversion
#pragma region HLS
#pragma region RGBtoHLS
    /// <summary>
    /// Converts RGB to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoHLS(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts RGB to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to HLS (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoHLS(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region BGRtoHLS
    /// <summary>
    /// Converts BGR to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoHLS(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts BGR to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to HLS<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to HLS (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoHLS(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion

#pragma region HLStoRGB
    /// <summary>
    /// Converts HLS to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HLStoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HLS to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HLStoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HLS to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HLStoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HLS to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts HLS to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HLS to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HLS to RGB (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HLStoRGB(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region HLStoBGR
    /// <summary>
    /// Converts HLS to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HLStoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HLS to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HLStoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HLS to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HLStoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HLS to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts HLS to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HLS to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HLS to BGR (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HLStoBGR(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma endregion

#pragma region HSV
#pragma region RGBtoHSV
    /// <summary>
    /// Converts RGB to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoHSV(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts RGB to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to HSV (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoHSV(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region BGRtoHSV
    /// <summary>
    /// Converts BGR to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoHSV(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts BGR to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to HSV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to HSV (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoHSV(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion

#pragma region HSVtoRGB
    /// <summary>
    /// Converts HSV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HSVtoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HSV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HSVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HSV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HSVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HSV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts HSV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HSV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HSV to RGB (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HSVtoRGB(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region HSVtoBGR
    /// <summary>
    /// Converts HSV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HSVtoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HSV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HSVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HSV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void HSVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HSV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts HSV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts HSV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts HSV to BGR (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &HSVtoBGR(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma endregion

#pragma region Lab
#pragma region RGBtoLab
    /// <summary>
    /// Converts RGB to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoLab(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts RGB to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to Lab (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoLab(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region BGRtoLab
    /// <summary>
    /// Converts BGR to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoLab(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts BGR to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to Lab<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to Lab (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoLab(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion

#pragma region LabtoRGB
    /// <summary>
    /// Converts Lab to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LabtoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts Lab to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LabtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts Lab to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LabtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts Lab to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts Lab to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts Lab to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts Lab to RGB (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LabtoRGB(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region LabtoBGR
    /// <summary>
    /// Converts Lab to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LabtoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts Lab to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LabtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts Lab to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LabtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts Lab to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts Lab to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts Lab to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts Lab to BGR (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LabtoBGR(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma endregion

#pragma region LUV
#pragma region RGBtoLUV
    /// <summary>
    /// Converts RGB to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoLUV(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void RGBtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts RGB to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts RGB to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts RGB to LUV (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &RGBtoLUV(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region BGRtoLUV
    /// <summary>
    /// Converts BGR to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoLUV(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void BGRtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts BGR to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts BGR to LUV<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts BGR to LUV (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &BGRtoLUV(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion

#pragma region LUVtoRGB
    /// <summary>
    /// Converts LUV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LUVtoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts LUV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LUVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts LUV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LUVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts LUV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts LUV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts LUV to RGB<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts LUV to RGB (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LUVtoRGB(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma region LUVtoBGR
    /// <summary>
    /// Converts LUV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LUVtoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts LUV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LUVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts LUV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    void LUVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                  float aNormalizationFactor) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts LUV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static void LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
                 std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// Converts LUV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// Converts LUV to BGR<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    static ImageView<T> &LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                  float aNormalizationFactor)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
                 std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// Converts LUV to BGR (inplace)<para/>
    /// Internally the computation is done on normalized values ranging from [0..1], a normalization factor has to be
    /// provided to normalize the input values to [0..1]. For byte images this is usually 255.0f, for unsigned short
    /// either 1023.0f (10-bit value range), 4095.0f (12-bit value range), 16383.0f (14-bit value range) or 65535.0f
    /// (16-bit value range). For float images the factor is usually 1.0f.
    /// </summary>
    ImageView<T> &LUVtoBGR(float aNormalizationFactor)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
                 std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>;
#pragma endregion
#pragma endregion

#pragma region ColorTwist3x3
    /// <summary>
    /// ColorTwist with a 3x3 matrix
    /// </summary>
    ImageView<T> &ColorTwist(ImageView<T> &aDst, const Matrix<float> &aTwist) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix
    /// </summary>
    void ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix
    /// </summary>
    void ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                    const Matrix<float> &aTwist) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix
    /// </summary>
    static void ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                    const Matrix<float> &aTwist)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix. Alpha channel in destination image is set to aAlpha.
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                    remove_vector_t<T> aAlpha, const Matrix<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                    const Matrix<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix (inplace)
    /// </summary>
    ImageView<T> &ColorTwist(const Matrix<float> &aTwist)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 422 (YCbCr/YCrCb/CbYCr).
    /// </summary>
    /// <param name="aDstLumaChroma">Destination image with Luma/Chroma interleaved</param>
    /// <param name="aTwist"></param>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    /// <param name="aStreamCtx"></param>
    ImageView<Vector2<remove_vector_t<T>>> &ColorTwistTo422(ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma,
                                                            const Matrix<float> &aTwist,
                                                            ChromaSubsamplePos aChromaSubsamplePos,
                                                            bool aSwapLumaChroma) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 422 (YCbCr/YCrCb/CbYCr).
    /// </summary>
    /// <param name="aDstLumaChroma">Destination image with Luma/Chroma interleaved</param>
    /// <param name="aTwist"></param>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    /// <param name="aStreamCtx"></param>
    static void ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    ImageView<Vector3<remove_vector_t<T>>> &ColorTwistFrom422(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                              const Matrix<float> &aTwist, bool aSwapLumaChroma) const
        requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
                 std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    void ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                           bool aSwapLumaChroma) const
        requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
                 std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                           const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                           const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                           const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                           const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                           const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                           const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x3 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;
#pragma endregion

#pragma region ColorTwist3x4
    /// <summary>
    /// ColorTwist with a 3x4 matrix
    /// </summary>
    ImageView<T> &ColorTwist(ImageView<T> &aDst, const Matrix3x4<float> &aTwist) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix
    /// </summary>
    void ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix
    /// </summary>
    void ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                    const Matrix3x4<float> &aTwist) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix
    /// </summary>
    static void ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                    const Matrix3x4<float> &aTwist)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix. Alpha channel in destination image is set to aAlpha.
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                    remove_vector_t<T> aAlpha, const Matrix3x4<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                    const Matrix3x4<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix (inplace)
    /// </summary>
    ImageView<T> &ColorTwist(const Matrix3x4<float> &aTwist)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 422 (YCbCr/YCrCb/CbYCr).
    /// </summary>
    /// <param name="aDstLumaChroma">Destination image with Luma/Chroma interleaved</param>
    /// <param name="aTwist"></param>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    /// <param name="aStreamCtx"></param>
    ImageView<Vector2<remove_vector_t<T>>> &ColorTwistTo422(ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma,
                                                            const Matrix3x4<float> &aTwist,
                                                            ChromaSubsamplePos aChromaSubsamplePos,
                                                            bool aSwapLumaChroma) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    void ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                         ChromaSubsamplePos aChromaSubsamplePos) const
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 422 (YCbCr/YCrCb/CbYCr).
    /// </summary>
    /// <param name="aDstLumaChroma">Destination image with Luma/Chroma interleaved</param>
    /// <param name="aTwist"></param>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    /// <param name="aStreamCtx"></param>
    static void ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma downsampling to 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    static void ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                ChromaSubsamplePos aChromaSubsamplePos)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    ImageView<Vector3<remove_vector_t<T>>> &ColorTwistFrom422(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                              const Matrix3x4<float> &aTwist,
                                                              bool aSwapLumaChroma) const
        requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
                 std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    void ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                           bool aSwapLumaChroma) const
        requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
                 std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                           const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                           const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 422.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                           const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                           const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 420.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                           const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static ImageView<T> &ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                           ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                           const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                           InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
                 std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
                 std::same_as<Pixel32fC4A, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;

    /// <summary>
    /// ColorTwist with a 3x4 matrix and chroma upsampling from 411.
    /// </summary>
    /// <param name="aChromaSubsamplePos">Position of the chroma sample point relative to luma</param>
    /// <param name="aSwapLumaChroma">Cb/Cr can be swapped in the TwistMatrix, but Luma and Chroma can only swapped with
    /// this parameter. Set to true for CbYCr.</param>
    static void ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                  ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
        requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
                 std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>;
#pragma endregion

#pragma region ColorTwist4x4
    /// <summary>
    /// ColorTwist with a 4x4 matrix
    /// </summary>
    ImageView<T> &ColorTwist(ImageView<T> &aDst, const Matrix4x4<float> &aTwist) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix
    /// </summary>
    void ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                    const Matrix4x4<float> &aTwist) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix
    /// </summary>
    static void ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                           const Matrix4x4<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                    const Matrix4x4<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix (inplace)
    /// </summary>
    ImageView<T> &ColorTwist(const Matrix4x4<float> &aTwist)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

#pragma endregion

#pragma region ColorTwist4x4C
    /// <summary>
    /// ColorTwist with a 4x4 matrix and an additional constant vector addition
    /// </summary>
    ImageView<T> &ColorTwist(ImageView<T> &aDst, const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix and an additional constant vector addition
    /// </summary>
    void ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                    const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant) const
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix and an additional constant vector addition
    /// </summary>
    static void ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                           const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                           ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                           const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix and an additional constant vector addition
    /// </summary>
    static ImageView<T> &ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                    const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

    /// <summary>
    /// ColorTwist with a 4x4 matrix (inplace) and an additional constant vector addition
    /// </summary>
    ImageView<T> &ColorTwist(const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant)
        requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
                 std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>;

#pragma endregion

#pragma region GammaCorrBT709
    /// <summary>
    /// Conversion linear to gamma corrected using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaCorrBT709(ImageView<T> &aDst, float aNormFactor) const
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

    /// <summary>
    /// Conversion linear to gamma corrected using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaCorrBT709(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 2);

    /// <summary>
    /// Conversion linear to gamma corrected using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaCorrBT709(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

    /// <summary>
    /// Conversion linear to gamma corrected using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaCorrBT709(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
        ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 4);

    /// <summary>
    /// Conversion linear to gamma corrected using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaCorrBT709(float aNormFactor)
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

#pragma endregion

#pragma region GammaInvCorrBT709
    /// <summary>
    /// Conversion gamma corrected to linear using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaInvCorrBT709(ImageView<T> &aDst, float aNormFactor) const
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

    /// <summary>
    /// Conversion gamma corrected to linear using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaInvCorrBT709(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 2);

    /// <summary>
    /// Conversion gamma corrected to linear using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaInvCorrBT709(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

    /// <summary>
    /// Conversion gamma corrected to linear using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaInvCorrBT709(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
        ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 4);

    /// <summary>
    /// Conversion gamma corrected to linear using BT.709 curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaInvCorrBT709(float aNormFactor)
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

#pragma endregion

#pragma region GammaCorrsRGB
    /// <summary>
    /// Conversion linear to gamma corrected using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaCorrsRGB(ImageView<T> &aDst, float aNormFactor) const
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

    /// <summary>
    /// Conversion linear to gamma corrected using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaCorrsRGB(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 2);

    /// <summary>
    /// Conversion linear to gamma corrected using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaCorrsRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

    /// <summary>
    /// Conversion linear to gamma corrected using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaCorrsRGB(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
        ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 4);

    /// <summary>
    /// Conversion linear to gamma corrected using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaCorrsRGB(float aNormFactor)
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

#pragma endregion

#pragma region GammaInvCorrsRGB
    /// <summary>
    /// Conversion gamma corrected to linear using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaInvCorrsRGB(ImageView<T> &aDst, float aNormFactor) const
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

    /// <summary>
    /// Conversion gamma corrected to linear using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaInvCorrsRGB(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 2);

    /// <summary>
    /// Conversion gamma corrected to linear using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaInvCorrsRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                 const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                 const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

    /// <summary>
    /// Conversion gamma corrected to linear using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    static void GammaInvCorrsRGB(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
        ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 4);

    /// <summary>
    /// Conversion gamma corrected to linear using sRGB curve. The values are normalized to [0..1] before the
    /// operation using aNormFactor. For 8u images this is usually 255.0f.
    /// </summary>
    ImageView<T> &GammaInvCorrsRGB(float aNormFactor)
        requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>;

#pragma endregion

#pragma region GradientColorToGray
    /// <summary>
    /// Converts a color gradient image to grayscale.
    /// </summary>
    ImageView<Vector1<remove_vector_t<T>>> &GradientColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aDst,
                                                                Norm aNorm) const
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_size_v<T> > 1);

    /// <summary>
    /// Converts a color gradient image to grayscale.
    /// </summary>
    static void GradientColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 2);

    /// <summary>
    /// Converts a color gradient image to grayscale.
    /// </summary>
    static void GradientColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

    /// <summary>
    /// Converts a color gradient image to grayscale.
    /// </summary>
    static void GradientColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 4);

#pragma endregion

#pragma region ColorToGray
    /// <summary>
    /// Converts a multi-channel color image to a one channel gray scale image where each channel is weighted by the
    /// factor provided in aWeights.<para/> Common values to convert RGB to gray are nGray = 0.299 * R + 0.587 * G +
    /// 0.114 * B
    /// </summary>
    ImageView<Vector1<remove_vector_t<T>>> &ColorToGray(
        ImageView<Vector1<remove_vector_t<T>>> &aDst, const same_vector_size_different_type_t<T, float> &aWeights) const
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_size_v<T> > 1);

    /// <summary>
    /// Converts a multi-channel color image to a one channel gray scale image where each channel is weighted by the
    /// factor provided in aWeights.<para/> Common values to convert RGB to gray are nGray = 0.299 * R + 0.587 * G +
    /// 0.114 * B
    /// </summary>
    static void ColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            const same_vector_size_different_type_t<T, float> &aWeights)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 2);

    /// <summary>
    /// Converts a multi-channel color image to a one channel gray scale image where each channel is weighted by the
    /// factor provided in aWeights.<para/> Common values to convert RGB to gray are nGray = 0.299 * R + 0.587 * G +
    /// 0.114 * B
    /// </summary>
    static void ColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            const same_vector_size_different_type_t<T, float> &aWeights)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

    /// <summary>
    /// Converts a multi-channel color image to a one channel gray scale image where each channel is weighted by the
    /// factor provided in aWeights.<para/> Common values to convert RGB to gray are nGray = 0.299 * R + 0.587 * G +
    /// 0.114 * B
    /// </summary>
    static void ColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            const same_vector_size_different_type_t<T, float> &aWeights)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
                 std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 4);

#pragma endregion

#pragma region CFAToRGB
    /// <summary>
    /// Grayscale Color Filter Array to RGB Color Debayer conversion.<para/>
    /// Generates one RGB color pixel for every grayscale source pixel. Source and destination image ROIs must have even
    /// width and height. Missing pixel colors are generated using bilinear interpolation with chroma correlation of
    /// generated green values. aBayerGrid allows the user to specify the Bayer grid registration position at source
    /// image location oSrcROI.x, oSrcROI.y relative to pSrc.<para/> Possible registration positions are:<para/> BGGR,
    /// RGGB, GBRG, GRBG<para/> If it becomes necessary to access source pixels outside the aAllowedReadRoi then the
    /// source image borders are mirrored.
    /// </summary>
    ImageView<Vector3<remove_vector_t<T>>> &CFAToRGB(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                     BayerGridPosition aBayerGrid, Roi aAllowedReadRoi = Roi()) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
                 std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
                 std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>);

    /// <summary>
    /// Grayscale Color Filter Array to RGB Color Debayer conversion.<para/>
    /// Generates one RGB color pixel for every grayscale source pixel. Source and destination image ROIs must have even
    /// width and height. Missing pixel colors are generated using bilinear interpolation with chroma correlation of
    /// generated green values. aBayerGrid allows the user to specify the Bayer grid registration position at source
    /// image location oSrcROI.x, oSrcROI.y relative to pSrc.<para/> Possible registration positions are:<para/> BGGR,
    /// RGGB, GBRG, GRBG<para/> If it becomes necessary to access source pixels outside the aAllowedReadRoi then the
    /// source image borders are mirrored.<para/> The alpha channel (the last channel), is set to aAlpha.
    /// </summary>
    ImageView<Vector4<remove_vector_t<T>>> &CFAToRGB(ImageView<Vector4<remove_vector_t<T>>> &aDst,
                                                     remove_vector_t<T> aAlpha, BayerGridPosition aBayerGrid,
                                                     Roi aAllowedReadRoi = Roi()) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
                 std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
                 std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>);

    /// <summary>
    /// RGB Color to Grayscale Bayer Color Filter Array conversion.<para/>
    /// Depending on the chosen BayerGridPosition and pixel position, only either red, green or blue channel is stored
    /// in the destionation image.<para/> Image ROI must have even width and height.
    /// </summary>
    ImageView<Vector1<remove_vector_t<T>>> &RGBToCFA(ImageView<Vector1<remove_vector_t<T>>> &aDst,
                                                     BayerGridPosition aBayerGrid) const
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
                 std::same_as<uint, remove_vector_t<T>> || std::same_as<BFloat16, remove_vector_t<T>> ||
                 std::same_as<HalfFp16, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> == 3);

#pragma endregion

#pragma region LUTPalette
    /// <summary>
    /// Converts a LUT with levels aLevels and values aValues to a fixed palette of either 256 (8-bit types) or 65536
    /// (16-bit types) values. aPalette must be an array of at least 256 or 65536 elements, depending on datatype.
    /// </summary>
    static void LUTToPalette(const int *aLevels, const int *aValues, int aLUTSize,
                             Vector1<remove_vector_t<T>> *aPalette, InterpolationMode aInterpolationMode)
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
                 std::same_as<remove_vector_t<T>, ushort>);

    /// <summary>
    /// Palette look-up-table color conversion.<para/>
    /// aPalette must be an array of at least 2^aBitSize elements.<para/>
    /// aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<T> &LUTPalette(ImageView<T> &aDst, const T *aPalette, int aBitSize) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
                 std::same_as<remove_vector_t<T>, ushort>) &&
                (vector_size_v<T> == 1);

    /// <summary>
    /// Palette look-up-table color conversion - inplace.<para/>
    /// aPalette must be an array of at least 2^aBitSize elements.<para/>
    /// aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<T> &LUTPalette(const T *aPalette, int aBitSize)
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion.<para/>
    /// aPalette must be an array of at least 2^aBitSize elements.<para/>
    /// aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Vector3<remove_vector_t<T>>> &LUTPalette(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                       const Vector3<remove_vector_t<T>> *aPalette, int aBitSize) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion. One channel gray scale to 3 channel color. The palette is padded to 4
    /// channels where the last channel is not used.<para/> aPalette must be an array of at least 2^aBitSize
    /// elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Vector3<remove_vector_t<T>>> &LUTPalette(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                       const Vector4A<remove_vector_t<T>> *aPalette, int aBitSize) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion. One channel gray scale to 3 channel color.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Vector4A<remove_vector_t<T>>> &LUTPalette(ImageView<Vector4A<remove_vector_t<T>>> &aDst,
                                                        const Vector3<remove_vector_t<T>> *aPalette, int aBitSize) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion. One channel gray scale to 3 channel color.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements. The palette is padded to 4 channels where the last channel is not
    /// used.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Vector4A<remove_vector_t<T>>> &LUTPalette(ImageView<Vector4A<remove_vector_t<T>>> &aDst,
                                                        const Vector4A<remove_vector_t<T>> *aPalette,
                                                        int aBitSize) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion. One channel gray scale to 4 channel color.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Vector4<remove_vector_t<T>>> &LUTPalette(ImageView<Vector4<remove_vector_t<T>>> &aDst,
                                                       const Vector4<remove_vector_t<T>> *aPalette, int aBitSize) const
        requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion.<para/> aPalette is a pointer to number of color channels
    /// arrays of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<T> &LUTPalette(ImageView<T> &aDst,
                             const Vector1<remove_vector_t<T>> *const aPalette[vector_active_size_v<T>],
                             int aBitSize) const
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
                 std::same_as<remove_vector_t<T>, ushort>) &&
                (vector_active_size_v<T> >= 2);

    /// <summary>
    /// Palette look-up-table color conversion - inplace.<para/> aPalette is a pointer to number of color channels
    /// arrays of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<T> &LUTPalette(const Vector1<remove_vector_t<T>> *const aPalette[vector_active_size_v<T>], int aBitSize)
        requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
                 std::same_as<remove_vector_t<T>, ushort>) &&
                (vector_active_size_v<T> >= 2);

    /// <summary>
    /// Palette look-up-table color conversion with 16 bit to 8 bit conversion.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Pixel8uC1> &LUTPalette(ImageView<Pixel8uC1> &aDst, const Pixel8uC1 *aPalette, int aBitSize) const
        requires(std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion with 16 bit to 8 bit conversion.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Pixel8uC3> &LUTPalette(ImageView<Pixel8uC3> &aDst, const Pixel8uC3 *aPalette, int aBitSize) const
        requires(std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion with 16 bit to 8 bit conversion.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Pixel8uC4> &LUTPalette(ImageView<Pixel8uC4> &aDst, const Pixel8uC4 *aPalette, int aBitSize) const
        requires(std::same_as<Pixel16uC1, T>);

    /// <summary>
    /// Palette look-up-table color conversion with 16 bit to 8 bit conversion.<para/> aPalette must be an
    /// array of at least 2^aBitSize elements.<para/> aBitSize must be in range [1..(8*sizeof(pixel base type)].
    /// </summary>
    ImageView<Pixel8uC4A> &LUTPalette(ImageView<Pixel8uC4A> &aDst, const Pixel8uC4A *aPalette, int aBitSize) const
        requires(std::same_as<Pixel16uC1, T>);

#pragma endregion

#pragma region LUT
    /// <summary>
    /// Precomputes LUT indices in aAccelerator for an equispaced value range with aAcceleratorSize values.<para/>
    /// aAccelerator must be of at least aAcceleratorSize elements and can then be passed as an argument to LUT().
    /// </summary>
    static void LUTAccelerator(const Pixel32fC1 *aLevels, int *aAccelerator, int aLUTSize, int aAcceleratorSize)
        requires(RealFloatingPoint<remove_vector_t<T>>);

    /// <summary>
    /// Look-up-table color conversion.<para/>
    /// The look-up-table is derived from a set of user defined mapping points given in aLevels and aValues. The table
    /// is interpolated using the mode provided in aInterpolationMode, where only NearestNeighbor, Linear and
    /// CubicLagrange are supported.<para/> To speed up the table look-up, an accelerator array must be provided
    /// containing the LUT indices on an equispaced sampling. The method LUTAccelerator can be used to fill the array
    /// for given aLevels values. The accelerator array should be larger than aLevels.
    /// </summary>
    ImageView<T> &LUT(ImageView<T> &aDst, const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues, const int *aAccelerator,
                      int aLutSize, int aAcceleratorSize, InterpolationMode aInterpolationMode) const
        requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Look-up-table color conversion.<para/>
    /// The look-up-table is derived from a set of user defined mapping points given in aLevels and aValues. The table
    /// is interpolated using the mode provided in aInterpolationMode, where only NearestNeighbor, Linear and
    /// CubicLagrange are supported.<para/> To speed up the table look-up, an accelerator array must be provided
    /// containing the LUT indices on an equispaced sampling. The method LUTAccelerator can be used to fill the array
    /// for given aLevels values. The accelerator array should be larger than aLevels.
    /// </summary>
    ImageView<T> &LUT(ImageView<T> &aDst, const Pixel32fC1 *const aLevels[vector_active_size_v<T>],
                      const Pixel32fC1 *const aValues[vector_active_size_v<T>],
                      const int *const aAccelerator[vector_active_size_v<T>],
                      int const aLutSize[vector_active_size_v<T>], int const aAcceleratorSize[vector_active_size_v<T>],
                      InterpolationMode aInterpolationMode) const
        requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Look-up-table color conversion - inplace.<para/>
    /// The look-up-table is derived from a set of user defined mapping points given in aLevels and aValues. The table
    /// is interpolated using the mode provided in aInterpolationMode, where only NearestNeighbor, Linear and
    /// CubicLagrange are supported.<para/> To speed up the table look-up, an accelerator array must be provided
    /// containing the LUT indices on an equispaced sampling. The method LUTAccelerator can be used to fill the array
    /// for given aLevels values. The accelerator array should be larger than aLevels.
    /// </summary>
    ImageView<T> &LUT(const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues, const int *aAccelerator, int aLutSize,
                      int aAcceleratorSize, InterpolationMode aInterpolationMode)
        requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Look-up-table color conversion - inplace.<para/>
    /// The look-up-table is derived from a set of user defined mapping points given in aLevels and aValues. The table
    /// is interpolated using the mode provided in aInterpolationMode, where only NearestNeighbor, Linear and
    /// CubicLagrange are supported.<para/> To speed up the table look-up, an accelerator array must be provided
    /// containing the LUT indices on an equispaced sampling. The method LUTAccelerator can be used to fill the array
    /// for given aLevels values. The accelerator array should be larger than aLevels.
    /// </summary>
    ImageView<T> &LUT(const Pixel32fC1 *const aLevels[vector_active_size_v<T>],
                      const Pixel32fC1 *const aValues[vector_active_size_v<T>],
                      const int *const aAccelerator[vector_active_size_v<T>],
                      int const aLutSize[vector_active_size_v<T>], int const aAcceleratorSize[vector_active_size_v<T>],
                      InterpolationMode aInterpolationMode)
        requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> > 1);

#pragma endregion

#pragma region Lut3D
    /// <summary>
    /// Look-up-table color conversion using a 3D LUT and tri-linear interpolation.<para/>
    /// For C4 pixel types, the alpha channel is copied from source to destination image, for C4A the alpha channel in
    /// destination image remains untouched.<para/>
    /// aLut3D is an array of size aLutSize.x * aLutSize.y * aLutSize.z, where the blue channel axis is the fastest
    /// moving one. The LUT is provided with 3 elements per pixel (red, green, blue).
    /// </summary>
    ImageView<T> &LUTTrilinear(ImageView<T> &aDst, const Vector3<remove_vector_t<T>> *aLut3D,
                               const Vector3<remove_vector_t<T>> &aMinLevel,
                               const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize) const
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> >= 3);

    /// <summary>
    /// Look-up-table color conversion using a 3D LUT and tri-linear interpolation.<para/>
    /// For C4 pixel types, the alpha channel is copied from source to destination image, for C4A the alpha channel in
    /// destination image remains untouched.<para/>
    /// aLut3D is an array of size aLutSize.x * aLutSize.y * aLutSize.z, where the blue channel axis is the fastest
    /// moving one. The LUT is provided with 4 elements per pixel (red, green, blue), 4th channel is unused and is only
    /// for padding.
    /// </summary>
    ImageView<T> &LUTTrilinear(ImageView<T> &aDst, const Vector4A<remove_vector_t<T>> *aLut3D,
                               const Vector3<remove_vector_t<T>> &aMinLevel,
                               const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize) const
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> >= 3);

    /// <summary>
    /// Look-up-table color conversion using a 3D LUT and tri-linear interpolation - inline.<para/>
    /// For C4 pixel types, the alpha channel is copied from source to destination image, for C4A the alpha channel in
    /// destination image remains untouched.<para/>
    /// aLut3D is an array of size aLutSize.x * aLutSize.y * aLutSize.z, where the blue channel axis is the fastest
    /// moving one. The LUT is provided with 3 elements per pixel (red, green, blue).
    /// </summary>
    ImageView<T> &LUTTrilinear(const Vector3<remove_vector_t<T>> *aLut3D, const Vector3<remove_vector_t<T>> &aMinLevel,
                               const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> >= 3);

    /// <summary>
    /// Look-up-table color conversion using a 3D LUT and tri-linear interpolation - inline.<para/>
    /// For C4 pixel types, the alpha channel is copied from source to destination image, for C4A the alpha channel in
    /// destination image remains untouched.<para/>
    /// aLut3D is an array of size aLutSize.x * aLutSize.y * aLutSize.z, where the blue channel axis is the fastest
    /// moving one. The LUT is provided with 4 elements per pixel (red, green, blue), 4th channel is unused and is only
    /// for padding.
    /// </summary>
    ImageView<T> &LUTTrilinear(const Vector4A<remove_vector_t<T>> *aLut3D, const Vector3<remove_vector_t<T>> &aMinLevel,
                               const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize)
        requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
                 std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
                (vector_active_size_v<T> >= 3);

#pragma endregion

#pragma region CompColorKey
    /// <summary>
    /// All pixels of the this source image equal to the specified key color aColorKey are replaced with the
    /// corresponding pixel of the background image aSrc2 and stored in the destination image aDst.
    /// </summary>
    ImageView<T> &CompColorKey(const ImageView<T> &aSrc2, const T &aColorKey, ImageView<T> &aDst) const
        requires RealVector<T>;

    /// <summary>
    /// All pixels of the this source image equal to the specified key color aColorKey are replaced with the
    /// corresponding pixel of the background image aSrc2 and stored in the same source image (inplace operation).
    /// </summary>
    ImageView<T> &CompColorKey(const ImageView<T> &aSrc2, const T &aColorKey)
        requires RealVector<T>;
#pragma endregion

#pragma region ConvertSampling422
    /// <summary>
    /// Converts from packed C2 422 sampling to planar P2 422 sampling.
    /// </summary>
    /// <param name="aSwapLumaChroma">Set to false if input is in format YCbCr, true for CbYCr.</param>
    void ConvertSampling422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                            ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, bool aSwapLumaChroma) const
        requires RealVector<T> && (vector_size_v<T> == 2);

    /// <summary>
    /// Converts from packed C2 422 sampling to planar P3 422 sampling.
    /// </summary>
    /// <param name="aSwapLumaChroma">Set to false if input is in format YCbCr, true for CbYCr.</param>
    void ConvertSampling422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                            ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, bool aSwapLumaChroma) const
        requires RealVector<T> && (vector_size_v<T> == 2);

    /// <summary>
    /// Converts from planar P2 422 sampling to packed C2 422 sampling.
    /// </summary>
    /// <param name="aSwapLumaChroma">Set to false if output should be in format YCbCr, true for CbYCr.</param>
    static ImageView<Vector2<remove_vector_t<T>>> &ConvertSampling422(
        ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma, ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
        ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, bool aSwapLumaChroma)
        requires RealVector<T> && (vector_size_v<T> == 3);

    /// <summary>
    /// Converts from planar P3 422 sampling to packed C2 422 sampling.
    /// </summary>
    /// <param name="aSwapLumaChroma">Set to false if output should be in format YCbCr, true for CbYCr.</param>
    static ImageView<Vector2<remove_vector_t<T>>> &ConvertSampling422(
        ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma, ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
        ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma,
        bool aSwapLumaChroma)
        requires RealVector<T> && (vector_size_v<T> == 3);

#pragma endregion
#pragma endregion
};

} // namespace mpp::image::cpuSimple