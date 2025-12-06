#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "dataExchangeAndInit/conversionRelations.h"
#include "dataExchangeAndInit/scaleRelations.h"
#include "morphology/morphologyComputeT.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/arithmetic/addSquareProductWeightedOutputType.h>
#include <backends/cuda/image/arithmetic/arithmeticKernel.h>
#include <backends/cuda/image/dataExchangeAndInit/dataExchangeAndInitKernel.h>
#include <backends/cuda/image/filtering/windowSumResultType.h>
#include <backends/cuda/image/geometryTransforms/geometryTransformsKernel.h>
#include <backends/cuda/image/statistics/statisticsKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/filterArea.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/safeCast.h>
#include <common/statistics/indexMinMax.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>

namespace mpp::image::cuda
{

template <PixelType T> class ImageView
{
#pragma region Constructors
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
    static constexpr size_t ChannelCount = to_size_t(channel_count_v<T>);
    /// <summary>
    /// Default line-pitch alignment for manually allocated images [bytes]
    /// </summary>
    static constexpr size_t PitchAlignment = 256;

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

    static size_t PadImageWidthToPitch(size_t aWidthInBytes, size_t aPitchAlignment = PitchAlignment)
    {
        if (aWidthInBytes % aPitchAlignment != 0)
        {
            aWidthInBytes += aPitchAlignment - (aWidthInBytes % aPitchAlignment);
        }
        return aWidthInBytes;
    }

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
    ImageView(T *aBasePointer, const SizePitched &aSizeAlloc) noexcept
        : mPtr(aBasePointer), mPitch(aSizeAlloc.Pitch()), mPtrRoi(aBasePointer), mSizeAlloc(aSizeAlloc.Size()),
          mRoi(0, 0, aSizeAlloc.Size())
    {
    }
    ImageView(T *aBasePointer, const Size2D &aSize, size_t aPitch) noexcept
        : mPtr(aBasePointer), mPitch(aPitch), mPtrRoi(aBasePointer), mSizeAlloc(aSize), mRoi(0, 0, aSize)
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

    DEVICE_CODE operator ImageView<Vector4<remove_vector_t<T>>>() // NOLINT(hicpp-explicit-conversions)
        requires FourChannelAlpha<T>
    {
        return ImageView<Vector4<remove_vector_t<T>>>(reinterpret_cast<Vector4<remove_vector_t<T>> *>(mPtr),
                                                      SizePitched(mSizeAlloc, mPitch), mRoi);
    }

    DEVICE_CODE operator ImageView<Vector4A<remove_vector_t<T>>>() // NOLINT(hicpp-explicit-conversions)
        requires FourChannelNoAlpha<T>
    {
        return ImageView<Vector4A<remove_vector_t<T>>>(reinterpret_cast<Vector4A<remove_vector_t<T>> *>(mPtr),
                                                       SizePitched(mSizeAlloc, mPitch), mRoi);
    }

    /// <summary>
    /// Null can be used when a nullptr should be passed as an optional output argument. It is not made const as the
    /// optional output arguments need to be non-const...
    /// </summary>
    static ImageView<T> Null;
#pragma endregion

#pragma region Basics and Copy to device/host
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
    /// Base pointer moved to first pixel of actual ROI.
    /// </summary>
    [[nodiscard]] T *PointerRoi()
    {
        return mPtrRoi;
    }
    /// <summary>
    /// Base pointer moved to first pixel of actual ROI.
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

    /// <summary>
    /// Copy from Host to device memory (full image ignoring ROI)
    /// </summary>
    /// <param name="aHostSrc">Source</param>
    /// <param name="aHostStride">Size of one image line in bytes with padding (host image)</param>
    void CopyToDevice(const void *aHostSrc, size_t aHostStride = 0)
    {
        if (aHostStride == 0)
        {
            aHostStride = WidthInBytes();
        }
        cudaSafeCall(cudaMemcpy2D(mPtr, mPitch, aHostSrc, aHostStride, WidthInBytes(), to_size_t(Height()),
                                  cudaMemcpyHostToDevice));
    }

    /// <summary>
    /// Copy from device to device memory (full image ignoring ROI)
    /// </summary>
    /// <param name="aDeviceSrc">Source</param>
    void CopyDeviceToDevice(const ImageView &aDeviceSrc)
    {
        if (mSizeAlloc != aDeviceSrc.SizeAlloc())
        {
            throw ROIEXCEPTION("The source image does not have the same size as the destination image. Source size "
                               << aDeviceSrc.SizeAlloc() << ", Destination size: " << mSizeAlloc);
        }
        cudaSafeCall(cudaMemcpy2D(mPtr, mPitch, aDeviceSrc.Pointer(), aDeviceSrc.Pitch(), WidthInBytes(),
                                  to_size_t(Height()), cudaMemcpyDeviceToDevice));
    }

    /// <summary>
    /// Copy from device to device memory (full image ignoring ROI)
    /// </summary>
    /// <param name="aDeviceSrc">Source</param>
    /// <param name="aSourcePitch">Pitch of aDeviceSrc</param>
    void CopyDeviceToDevice(const T *aDeviceSrc, size_t aSourcePitch = 0)
    {
        if (aSourcePitch == 0)
        {
            aSourcePitch = WidthInBytes();
        }
        cudaSafeCall(cudaMemcpy2D(mPtr, mPitch, aDeviceSrc, aSourcePitch, WidthInBytes(), to_size_t(Height()),
                                  cudaMemcpyDeviceToDevice));
    }

    /// <summary>
    /// Copy data from device to host memory (full image ignoring ROI)
    /// </summary>
    /// <param name="aHostDest">void* to destination in host memory</param>
    /// <param name="aHostStride">Size of one image line in bytes with padding</param>
    void CopyToHost(void *aHostDest, size_t aHostStride = 0) const
    {
        if (aHostStride == 0)
        {
            aHostStride = WidthInBytes();
        }
        cudaSafeCall(cudaMemcpy2D(aHostDest, aHostStride, mPtr, mPitch, WidthInBytes(), to_size_t(Height()),
                                  cudaMemcpyDeviceToHost));
    }

    /// <summary>
    /// Copy from device to host memory (full image ignoring ROI)
    /// </summary>
    void operator>>(void *aDest) const
    {
        CopyToHost(aDest);
    }

    /// <summary>
    /// Copy from host to device memory (full image ignoring ROI)
    /// </summary>
    void operator<<(const void *aSource)
    {
        CopyToDevice(aSource);
    }

    /// <summary>
    /// Copy from device to host memory (full image ignoring ROI)
    /// </summary>
    void CopyToHost(std::vector<T> &aHostDest) const
    {
        CopyToHost(aHostDest.data());
    }

    /// <summary>
    /// Copy from host to device memory (full image ignoring ROI)
    /// </summary>
    void CopyToDevice(const std::vector<T> &aHostSrc)
    {
        CopyToDevice(aHostSrc.data());
    }

    /// <summary>
    /// Copy from device to host memory (full image ignoring ROI)
    /// </summary>
    void operator>>(std::vector<T> &aHostDest) const
    {
        CopyToHost(aHostDest.data());
    }

    /// <summary>
    /// Copy from host to device memory (full image ignoring ROI)
    /// </summary>
    void operator<<(const std::vector<T> &aHostSrc)
    {
        CopyToDevice(aHostSrc.data());
    }

    /// <summary>
    /// Copy data from host to device memory only in ROI
    /// </summary>
    /// <param name="aSource">Pointer to pixel (0,0) in source image</param>
    /// <param name="aSourcePitch">Pitch of source array</param>
    /// <param name="aRoiSource">ROI of source image, empty if full image</param>
    void CopyToDeviceRoi(const void *aSource, size_t aSourcePitch = 0, const Roi &aRoiSource = Roi())
    {
        Roi roiSrc = aRoiSource;
        if (roiSrc.width == 0 && roiSrc.height == 0)
        {
            roiSrc = Roi(0, 0, SizeRoi());
        }

        if (aSourcePitch == 0)
        {
            aSourcePitch = WidthRoiInBytes();
        }

        if (SizeRoi() != roiSrc.Size())
        {
            throw ROIEXCEPTION("Source and destination ROI must have same size. Source size is "
                               << roiSrc.Size() << " destination size is " << SizeRoi());
        }
        const T *src = gotoPtr(static_cast<const T *>(aSource), aSourcePitch, aRoiSource.x, aRoiSource.y);

        cudaSafeCall(cudaMemcpy2D(mPtrRoi, mPitch, src, aSourcePitch, WidthRoiInBytes(), to_size_t(HeightRoi()),
                                  cudaMemcpyHostToDevice));
    }

    /// <summary>
    /// Copy data from device to host memory only in ROI
    /// </summary>
    /// <param name="aDest">Pointer to pixel (0,0) in destination image</param>
    /// <param name="aDestPitch">Pitch of source array</param>
    /// <param name="aRoiDest">ROI of destination image, empty if full image</param>
    void CopyToHostRoi(void *aDest, size_t aDestPitch = 0, const Roi &aRoiDest = Roi()) const
    {
        Roi roiDest = aRoiDest;
        if (roiDest.width == 0 && roiDest.height == 0)
        {
            roiDest = Roi(0, 0, SizeRoi());
        }

        if (aDestPitch == 0)
        {
            aDestPitch = WidthRoiInBytes();
        }

        if (SizeRoi() != roiDest.Size())
        {
            throw ROIEXCEPTION("Source and destination ROI must have same size. Source size is "
                               << SizeRoi() << " destination size is " << roiDest.Size());
        }
        T *dest = gotoPtr(static_cast<T *>(aDest), aDestPitch, roiDest.x, roiDest.y);

        cudaSafeCall(cudaMemcpy2D(dest, aDestPitch, mPtrRoi, mPitch, WidthRoiInBytes(), to_size_t(HeightRoi()),
                                  cudaMemcpyDeviceToHost));
    }
#pragma endregion

#pragma region Data initialisation
#pragma region Convert
    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && ConversionImplemented<T, TTo>;

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && ConversionRoundImplemented<T, TTo>;

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && ConversionRoundScaleImplemented<T, TTo> &&
                (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) && (!std::same_as<TTo, Complex<float>>) &&
                (!std::same_as<TTo, Complex<double>>);
#pragma endregion
#pragma region Copy
    /// <summary>
    /// Copy image.
    /// </summary>
    ImageView<T> &Copy(ImageView<T> &aDst,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    ImageView<T> &CopyMasked(ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst, Channel aDstChannel,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> > 1) &&   //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy this single channel image to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(ImageView<TTo> &aDst, Channel aDstChannel,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> == 1) &&  //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> > 1) &&    //
                (vector_size_v<TTo> == 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel4,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(FourChannelNoAlpha<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel4, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires(FourChannelNoAlpha<T>);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    ImageView<T> &Copy(ImageView<T> &aDst, const Vector2<int> &aLowerBorderSize, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    ImageView<T> &Copy(ImageView<T> &aDst, const Vector2<int> &aLowerBorderSize, BorderType aBorder, const T &aConstant,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    ImageView<T> &Copy(ImageView<T> &aDst, const Pixel32fC2 &aDelta, InterpolationMode aInterpolation,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
#pragma endregion
#pragma region Dup
    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Dup(ImageView<TTo> &aDst,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> == 1) &&
                (vector_size_v<TTo> > 1) && std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;
#pragma endregion
#pragma region Scale
    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, RoundingMode aRoundingMode,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
                RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
                RealOrComplexFloatingVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                          RoundingMode aRoundingMode,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
                RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          RoundingMode aRoundingMode,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexFloatingVector<TTo> && ScaleImplemented<T, TTo>
    ;

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax, RoundingMode aRoundingMode,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!std::same_as<T, TTo>) &&
                (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>;

#pragma endregion
#pragma region Set
    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    ImageView<T> &SetMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    ImageView<T> &SetMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());
    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(remove_vector_t<T> aConst, Channel aChannel,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> > 1);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(const mpp::cuda::DevVarView<remove_vector_t<T>> &aConst, Channel aChannel,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> > 1);
#pragma endregion
#pragma region Swap Channel
    /// <summary>
    /// Swap channels
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires((vector_active_size_v<TTo> <= vector_active_size_v<T>)) && //
                (vector_size_v<T> >= 3) &&                                  //
                (vector_size_v<TTo> >= 3) &&                                //
                (!has_alpha_channel_v<TTo>) &&                              //
                (!has_alpha_channel_v<T>) &&                                //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Swap channels (inplace)
    /// </summary>
    ImageView<T> &SwapChannel(const ChannelList<vector_active_size_v<T>> &aDstChannels,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> >= 3) && (!has_alpha_channel_v<T>);

    /// <summary>
    /// Swap channels (3-channel to 4-channel with additional value). If aDstChannels[i] == 3, channel i of aDst is set
    /// to aValue, if aDstChannels[i] > 3, channel i of aDst is kept unchanged.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                remove_vector_t<T> aValue,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> == 3) &&          //
                (vector_active_size_v<TTo> == 4) && //
                (!has_alpha_channel_v<TTo>) &&      //
                (!has_alpha_channel_v<T>) &&        //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;
#pragma endregion
#pragma region Transpose
    /// <summary>
    /// Transpose image.
    /// </summary>
    ImageView<T> &Transpose(ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires NoAlpha<T>;
#pragma endregion
#pragma endregion

#pragma region Arithmetic functions
#pragma region Add
    /// <summary>
    /// aDst = this + aSrc2
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst
    /// </summary>
    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst
    /// </summary>
    ImageView<T> &Add(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aSrc2
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst
    /// </summary>
    ImageView<T> &Add(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const T &aConst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst
    /// </summary>
    ImageView<T> &Add(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aSrc2 for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &AddMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Sub
    /// <summary>
    /// aDst = this - aSrc2
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst
    /// </summary>
    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst
    /// </summary>
    ImageView<T> &Sub(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aSrc2
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst
    /// </summary>
    ImageView<T> &Sub(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const T &aConst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst
    /// </summary>
    ImageView<T> &Sub(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 - this
    /// </summary>
    ImageView<T> &SubInv(const ImageView<T> &aSrc2,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 - this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &SubInv(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this
    /// </summary>
    ImageView<T> &SubInv(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &SubInv(const T &aConst, int aScaleFactor = 0,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this
    /// </summary>
    ImageView<T> &SubInv(const mpp::cuda::DevVarView<T> &aConst,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &SubInv(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &SubMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 - this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 - this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                               int aScaleFactor                       = 0,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Mul
    /// <summary>
    /// aDst = this * aSrc2
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst
    /// </summary>
    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst
    /// </summary>
    ImageView<T> &Mul(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aSrc2
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst
    /// </summary>
    ImageView<T> &Mul(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const T &aConst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst
    /// </summary>
    ImageView<T> &Mul(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aSrc2 for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor                       = 0,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region MulScale
    /// <summary>
    /// aDst = this * aSrc2, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const T &aConst, ImageView<T> &aDst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// this = this * aSrc2, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const ImageView<T> &aSrc2,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// this = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const T &aConst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// this = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const mpp::cuda::DevVarView<T> &aConst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aSrc2, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScaleMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScaleMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScaleMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                 const ImageView<Pixel8uC1> &aMask,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// this = this * aSrc2, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScaleMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// this = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScaleMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// this = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScaleMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;
#pragma endregion
#pragma region Div
    /// <summary>
    /// aDst = this / aSrc2
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst
    /// </summary>
    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst
    /// </summary>
    ImageView<T> &Div(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aSrc2
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst
    /// </summary>
    ImageView<T> &Div(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const T &aConst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst
    /// </summary>
    ImageView<T> &Div(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 / this
    /// </summary>
    ImageView<T> &DivInv(const ImageView<T> &aSrc2,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 / this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &DivInv(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this
    /// </summary>
    ImageView<T> &DivInv(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &DivInv(const T &aConst, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this
    /// </summary>
    ImageView<T> &DivInv(const mpp::cuda::DevVarView<T> &aConst,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &DivInv(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                            const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                            RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &DivMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                            int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 / this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 / this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                               RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                               RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                               int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region AddSquare
    /// <summary>
    /// SrcDst += this^2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddSquare(
        ImageView<add_spw_output_for_t<T>> &aSrcDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// SrcDst += this^2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddSquareMasked(
        ImageView<add_spw_output_for_t<T>> &aSrcDst, const ImageView<Pixel8uC1> &aMask,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());
#pragma endregion
#pragma region AddProduct
    /// <summary>
    /// SrcDst += this * Src2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddProduct(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aSrcDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// SrcDst += this * Src2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddProductMasked(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aSrcDst, const ImageView<Pixel8uC1> &aMask,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());
#pragma endregion
#pragma region AddWeighted
    /// <summary>
    /// Dst = this * alpha + Src2 * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aDst,
        remove_vector_t<add_spw_output_for_t<T>> aAlpha,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Dst = this * alpha + Src2 * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeightedMasked(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aDst,
        remove_vector_t<add_spw_output_for_t<T>> aAlpha, const ImageView<Pixel8uC1> &aMask,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// SrcDst = this * alpha + SrcDst * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(
        ImageView<add_spw_output_for_t<T>> &aSrcDst, remove_vector_t<add_spw_output_for_t<T>> aAlpha,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// SrcDst = this * alpha + SrcDst * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeightedMasked(
        ImageView<add_spw_output_for_t<T>> &aSrcDst, remove_vector_t<add_spw_output_for_t<T>> aAlpha,
        const ImageView<Pixel8uC1> &aMask,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());
#pragma endregion

#pragma region Abs
    /// <summary>
    /// aDst = abs(this)
    /// </summary>
    ImageView<T> &Abs(ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealSignedVector<T>;

    /// <summary>
    /// this = abs(this)
    /// </summary>
    ImageView<T> &Abs(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealSignedVector<T>;
#pragma endregion
#pragma region AbsDiff
    /// <summary>
    /// aDst = abs(this - aSrc2)
    /// </summary>
    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealUnsignedVector<T>;

    /// <summary>
    /// aDst = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const T &aConst, ImageView<T> &aDst,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealUnsignedVector<T>;

    /// <summary>
    /// aDst = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealUnsignedVector<T>;

    /// <summary>
    /// this = abs(this - aSrc2)
    /// </summary>
    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// this = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const T &aConst,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// this = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const mpp::cuda::DevVarView<T> &aConst,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;
#pragma endregion
#pragma region And
    /// <summary>
    /// aDst = this & aSrc2 (bitwise AND)
    /// </summary>
    ImageView<T> &And(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const T &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// this = this & aSrc2 (bitwise AND)
    /// </summary>
    ImageView<T> &And(const ImageView<T> &aSrc2,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Not
    /// <summary>
    /// aDst = ~this (bitwise NOT)
    /// </summary>
    ImageView<T> &Not(ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// this = ~this (bitwise NOT)
    /// </summary>
    ImageView<T> &Not(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Exp
    /// <summary>
    /// aDst = exp(this) (exponential function)
    /// </summary>
    ImageView<T> &Exp(ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = exp(this) (exponential function)
    /// </summary>
    ImageView<T> &Exp(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Ln
    /// <summary>
    /// aDst = log(this) (natural logarithm)
    /// </summary>
    ImageView<T> &Ln(ImageView<T> &aDst,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = log(this) (natural logarithm)
    /// </summary>
    ImageView<T> &Ln(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region LShift
    /// <summary>
    /// aDst = this << aConst (left bitshift)
    /// </summary>
    ImageView<T> &LShift(uint aConst, ImageView<T> &aDst,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;
    /// <summary>
    /// this = this << aConst (left bitshift)
    /// </summary>
    ImageView<T> &LShift(uint aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Or
    /// <summary>
    /// aDst = this | aSrc2 (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const T &aConst, ImageView<T> &aDst,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// this = this | aSrc2 (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const ImageView<T> &aSrc2,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const mpp::cuda::DevVarView<T> &aConst,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region RShift
    /// <summary>
    /// aDst = this >> aConst (right bitshift)
    /// </summary>
    ImageView<T> &RShift(uint aConst, ImageView<T> &aDst,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;
    /// <summary>
    /// this = this >> aConst (right bitshift)
    /// </summary>
    ImageView<T> &RShift(uint aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Sqr
    /// <summary>
    /// aDst = this * this (this^2)
    /// </summary>
    ImageView<T> &Sqr(ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = this * this (this^2)
    /// </summary>
    ImageView<T> &Sqr(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Sqrt
    /// <summary>
    /// aDst = Sqrt(this) (square root function)
    /// </summary>
    ImageView<T> &Sqrt(ImageView<T> &aDst,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = Sqrt(this) (square root function)
    /// </summary>
    ImageView<T> &Sqrt(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Xor
    /// <summary>
    /// aDst = this ^ aSrc2 (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const T &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// this = this ^ aSrc2 (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const ImageView<T> &aSrc2,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const mpp::cuda::DevVarView<T> &aConst,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion

#pragma region AlphaPremul
    /// <summary>
    /// Premultiplies pixels of an image with alpha from fourth color channel.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires FourChannelNoAlpha<T> && RealVector<T>;

    /// <summary>
    /// Premultiplies pixels of an image with alpha from fourth color channel.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires FourChannelNoAlpha<T> && RealVector<T>;

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value. aAlpha is expected in value range 0..1
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealFloatingVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value. aAlpha is expected in value range 0..1
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealFloatingVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires FourChannelAlpha<T>;

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value. aAlpha is expected in value range 0..1
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires FourChannelAlpha<T>;
#pragma endregion
#pragma region AlphaComp
    /// <summary>
    /// Composite two images using alpha opacity values contained in each image. Last color channel is alpha channel, 1
    /// channel images are treated as alpha channel only.
    /// </summary>
    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!FourChannelAlpha<T>) && RealVector<T>;

    /// <summary>
    /// Composite two images using constant alpha values.
    /// </summary>
    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                            remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion

#pragma region Complex
    /// <summary>
    /// aDst = this * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T>;

    /// <summary>
    /// this = this * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = conj(this) (complex conjugate)
    /// </summary>
    ImageView<T> &Conj(ImageView<T> &aDst,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T>;

    /// <summary>
    /// this = conj(this) (complex conjugate)
    /// </summary>
    ImageView<T> &Conj(const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = abs(this) (complex magnitude)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Magnitude(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = abs(this)^2 (complex magnitude squared)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &MagnitudeSqr(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = angle(this) (complex angle, atan2(imag, real))
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Angle(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = this.real (real component of complex value)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Real(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = this.imag (imaginary component of complex value)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Imag(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires ComplexVector<T>;

    /// <summary>
    /// aDst.real = this, aDst.imag = 0 (converts real valued image to complex with imaginary part = 0)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealSignedVector<T> && (!FourChannelAlpha<T>) &&
                 (std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
                  std::same_as<float, remove_vector_t<T>>);

    /// <summary>
    /// aDst.real = this, aDst.imag = aSrcImag (converts two real valued images to one complex image)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        const ImageView<T> &aSrcImag,
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealSignedVector<T> && (!FourChannelAlpha<T>) &&
                 (std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
                  std::same_as<float, remove_vector_t<T>>);
#pragma endregion
#pragma endregion

#pragma region Filtering
#pragma region Fixed Filter
    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder, const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, BorderType aBorder,
                              const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, const T &aConstant,
                              BorderType aBorder,

                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<T> &FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize, BorderType aBorder,

                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);
    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);
    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &FixedFilter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);
#pragma endregion
#pragma region Separable Filter
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst,
                                  const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                  int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                                  const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst,
                                  const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                  int aFilterSize, int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst,
                                  const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                  int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &SeparableFilter(ImageView<T> &aDst,
                                  const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                  int aFilterSize, int aFilterCenter, BorderType aBorder,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
#pragma endregion
#pragma region Column Filter

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst,
                               const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                               int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst,
                               const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                               int aFilterSize, int aFilterCenter, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst,
                               const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                               int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                               const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &ColumnFilter(ImageView<T> &aDst,
                               const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                               int aFilterSize, int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &ColumnWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &ColumnWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &ColumnWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &ColumnWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
#pragma endregion
#pragma region Row Filter
    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst,
                            const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                            int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst,
                            const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                            int aFilterSize, int aFilterCenter, BorderType aBorder,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst,
                            const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                            int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    ImageView<T> &RowFilter(ImageView<T> &aDst,
                            const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                            int aFilterSize, int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &RowWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &RowWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &RowWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    ImageView<window_sum_result_type_t<T>> &RowWindowSum(
        ImageView<window_sum_result_type_t<T>> &aDst,
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize,
        int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
#pragma endregion
#pragma region Box Filter
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<T> &BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, const T &aConstant,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, const T &aConstant,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;
    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, float>> &BoxFilter(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T>;

    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(
        ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);
    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(
        ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);

    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(
        ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);
    /// <summary>
    /// A specialised box filter for one-channel images that returns in first channel result image the mean value under
    /// the box area and in the second channel the summed squared pixel values. The result can then be used in the
    /// CrossCorrelationCoefficient function.
    /// </summary>
    ImageView<Pixel32fC2> &BoxAndSumSquareFilter(
        ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8);
#pragma endregion
#pragma region Min/Max Filter
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the maximum pixel value.
    /// </summary>
    ImageView<T> &MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,

                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,

                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The filter finds in the neighborhood of each pixel defined in aFilterArea the minimum pixel value.
    /// </summary>
    ImageView<T> &MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                            const Roi &aAllowedReadRoi,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

#pragma endregion
#pragma region Wiener Filter
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, const T &aConstant, BorderType aBorder,

                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, BorderType aBorder,

                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, const T &aConstant, BorderType aBorder,
                               const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Applies Wiener filter to the image.
    /// </summary>
    ImageView<T> &WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                               const filter_compute_type_for_t<T> &aNoise, BorderType aBorder,
                               const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

#pragma endregion
#pragma region Threshold Adaptive Box Filter
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the source pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea, const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
        const T &aValLE, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the source pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea, const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
        const T &aValLE, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the source pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea, const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
        const T &aValLE, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Computes the average pixel values of the pixels under a mask.
    /// Once the neighborhood average around a source pixel is determined the source pixel is compared to the average
    /// aDelta and if the source pixel is greater than that average the corresponding destination pixel is set to
    /// aValGT, otherwise aValLE.
    /// </summary>
    ImageView<T> &ThresholdAdaptiveBoxFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea, const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
        const T &aValLE, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

#pragma endregion
#pragma region Filter
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst,
                         const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst,
                         const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                         const FilterArea &aFilterArea, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst,
                         const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                         const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<T> &Filter(ImageView<T> &aDst,
                         const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                         const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &Filter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst,
        const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
        const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &Filter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst,
        const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
        const FilterArea &aFilterArea, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &Filter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst,
        const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
        const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);
    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    ImageView<alternative_filter_output_type_for_t<T>> &Filter(
        ImageView<alternative_filter_output_type_for_t<T>> &aDst,
        const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
        const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(has_alternative_filter_output_type_for_v<T>);
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
    void PrecomputeBilateralGaussFilter(
        mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, const FilterArea &aFilterArea, float aPosSquareSigma,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, const T &aConstant,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, const T &aConstant,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);

    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> &&
                 (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
        const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(!SingleChannel<T>) && RealVector<T> &&
                (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>);
    /// <summary>
    /// Applies the bilateral Gauss filter to the image using pre-computed geometrical distance coefficients obtained
    /// from PrecomputeBilateralGaussFilter().
    /// </summary>
    ImageView<T> &BilateralGaussFilter(
        ImageView<T> &aDst, const FilterArea &aFilterArea,
        const mpp::cuda::DevVarView<Pixel32fC1> &aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, BorderType aBorder, const Roi &aAllowedReadRoi,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, BorderType aBorder, const Roi &aAllowedReadRoi,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              BorderType aBorder, const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              BorderType aBorder, const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, const T &aConstant, BorderType aBorder,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, BorderType aBorder,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             MaskSize aMaskSize, BorderType aBorder,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              BorderType aBorder,

                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                              BorderType aBorder,

                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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

                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               const T &aConstant, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               const T &aConstant, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                               BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst,
                                const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                const T &aConstant, BorderType aBorder,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst,
                                const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                BorderType aBorder,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst,
                                const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
    ImageView<T> &UnsharpFilter(ImageView<T> &aDst,
                                const mpp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                int aFilterSize, int aFilterCenter,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                BorderType aBorder, const Roi &aAllowedReadRoi,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region Harris Corner Response
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(
        ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK, float aScale, const T &aConstant,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<T, Pixel32fC4>;
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(
        ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK, float aScale, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<T, Pixel32fC4>;
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(
        ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK, float aScale, const T &aConstant,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<T, Pixel32fC4>;
    /// <summary>
    /// From a covariance matrix for each pixel obtained from one of the GradientVector functions, this function
    /// computes the Harris Corner response.
    /// </summary>
    ImageView<Pixel32fC1> &HarrisCornerResponse(
        ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize, float aK, float aScale, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<T, Pixel32fC4>;

#pragma endregion
#pragma region Canny edge
    /// <summary>
    /// For an gradient magnitude image and an gradient orientation image obtained from one of the gradient vector
    /// functions, this function performs canny edge detection.
    /// </summary>
    ImageView<Pixel8uC1> &CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                    ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires std::same_as<T, Pixel16sC1> || std::same_as<T, Pixel32fC1>;
    /// <summary>
    /// For an gradient magnitude image and an gradient orientation image obtained from one of the gradient vector
    /// functions, this function performs canny edge detection.
    /// </summary>
    ImageView<Pixel8uC1> &CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                    ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold,
                                    const Roi &aAllowedReadRoi,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
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
                             const Roi &aAllowedReadRoi,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                             InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                           const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                 const Roi &aAllowedReadRoi,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                                 InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                               const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                               BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                               const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    static void WarpAffineBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                             InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                             InterpolationMode aInterpolation, BorderType aBorder,
                             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                           const T &aConstant, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           InterpolationMode aInterpolation, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           const T &aConstant, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                           BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                               const T &aConstant, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                               BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    static void WarpAffineBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    static void WarpAffineBack(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                  const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                                  InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                                const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                BorderType aBorder, const Roi &aAllowedReadRoi,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &WarpPerspectiveBack(
        ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
    ImageView<T> &WarpPerspectiveBack(
        ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                                    const Roi &aAllowedReadRoi,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                    InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                  InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                                const T &aConstant, BorderType aBorder,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                BorderType aBorder,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    static void WarpPerspective(
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &WarpPerspectiveBack(
        ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
        const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
    ImageView<T> &WarpPerspectiveBack(
        ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                                    InterpolationMode aInterpolation, BorderType aBorder,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
        ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
        const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                         const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                         InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                         InterpolationMode aInterpolation, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                       const T &aConstant, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       InterpolationMode aInterpolation, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const T &aConstant, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &Resize(ImageView<T> &aDst, InterpolationMode aInterpolation,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                       InterpolationMode aInterpolation,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       ImageView<Vector1<remove_vector_t<T>>> &aDst3, InterpolationMode aInterpolation,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       InterpolationMode aInterpolation,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires FourChannelNoAlpha<T>;

    /// <summary>
    /// Returns a shift to be used in MPP resize method that matches to the result given by the NPP Resize-function.
    /// </summary>
    Vec2f ResizeGetNPPShift(ImageView<T> &aDst) const;

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
                         const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                         InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    /// fall outside this expanded area, the pixel value is not defined. <para/> For
    /// BorderType::Constant, the constant value to use must be provided.
    /// </summary>
    ImageView<T> &Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                         InterpolationMode aInterpolation, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                       const T &aConstant, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       const T &aConstant, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                       BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires FourChannelNoAlpha<T>;
#pragma endregion

#pragma region Mirror
    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    ImageView<T> &Mirror(ImageView<T> &aDst, MirrorAxis aAxis,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    ImageView<T> &Mirror(MirrorAxis aAxis,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());
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
                        const Roi &aAllowedReadRoi,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                        InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                        BorderType aBorder, const Roi &aAllowedReadRoi,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                      BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                        InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                        InterpolationMode aInterpolation, BorderType aBorder,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                        const T &aConstant, BorderType aBorder,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
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
                        BorderType aBorder,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

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
                      BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      InterpolationMode aInterpolation, BorderType aBorder,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      InterpolationMode aInterpolation, BorderType aBorder,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const T &aConstant, BorderType aBorder,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
                      InterpolationMode aInterpolation, BorderType aBorder,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires FourChannelNoAlpha<T>;
#pragma endregion
#pragma endregion

#pragma region Morphology
#pragma region No mask Erosion/Dilation
    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                           const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                          const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                          const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,

                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the entire mask area defined by aFilterArea (maximum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,

                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,

                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the entire mask area defined by aFilterArea (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,

                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

#pragma endregion
#pragma region Erosion
    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                          const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                          const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                          const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                              const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                              const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                              const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                          const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs erosion on the mask area defined by aFilterArea and where aMask is != 0 (minimum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Erosion(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                          const FilterArea &aFilterArea, BorderType aBorder,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                              const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-erosion on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &ErosionGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                              const FilterArea &aFilterArea, BorderType aBorder,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region Dilation
    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                               const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                               const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                               const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs dilation on the mask area defined by aFilterArea and where aMask is != 0 (maximum pixel in the
    /// neighborhood).
    /// </summary>
    ImageView<T> &Dilation(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                               const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Performs gray-scale-dilation on the mask area defined by aFilterArea. The value of aMask is added to the pixel
    /// value and clamped to pixel type value range before comparison (minimum pixel in the neighborhood).
    /// </summary>
    ImageView<T> &DilationGray(ImageView<T> &aDst, const mpp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                               const FilterArea &aFilterArea, BorderType aBorder,
                               const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region Open
    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                       const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                       const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                       const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                       const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// First applies erosion then dilation.
    /// </summary>
    ImageView<T> &Open(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                       const FilterArea &aFilterArea, BorderType aBorder,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region Close
    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                        const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                        const Roi &aAllowedReadRoi,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                        const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                        const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// First applies dilation then erosion.
    /// </summary>
    ImageView<T> &Close(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                        const FilterArea &aFilterArea, BorderType aBorder,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region TopHat
    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                         const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                         const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the original image minus the result from morphological opening.
    /// </summary>
    ImageView<T> &TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                         const FilterArea &aFilterArea, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region BlackHat
    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// The result is the result from morphological closing minus the original image.
    /// </summary>
    ImageView<T> &BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask,
                           const FilterArea &aFilterArea, BorderType aBorder,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma region Morphology Gradient
    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(
        ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(
        ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(
        ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
        const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Dilation minus erosion.
    /// </summary>
    ImageView<T> &MorphologyGradient(
        ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion
#pragma endregion

#pragma region Statistics
#pragma region AverageError
    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t AverageErrorBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t AverageErrorMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the average error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<remove_vector_t<averageError_types_for_rt<T>>> &aDstScalar,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the average error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageErrorMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                            mpp::cuda::DevVarView<remove_vector_t<averageError_types_for_rt<T>>> &aDstScalar,
                            const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the average error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the average error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageErrorMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                            const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion
#pragma region AverageRelativeError
    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t AverageRelativeErrorBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t AverageRelativeErrorMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the average relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/> For multi-channel images, the result is computed for each channel seperatly in
    /// aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute value is used
    /// for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageRelativeError(const ImageView<T> &aSrc2,
                              mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<averageRelativeError_types_for_rt<T>>> &aDstScalar,
                              mpp::cuda::DevVarView<byte> &aBuffer,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the average relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageRelativeErrorMasked(
        const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
        mpp::cuda::DevVarView<remove_vector_t<averageRelativeError_types_for_rt<T>>> &aDstScalar,
        const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the average relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageRelativeError(const ImageView<T> &aSrc2,
                              mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<byte> &aBuffer,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the average relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void AverageRelativeErrorMasked(const ImageView<T> &aSrc2,
                                    mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region DotProduct
    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t DotProductBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t DotProductMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the dot product of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the dot product is defined as: DotProduct = Sum(Src1(i,j) * Src2(i,j)) <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void DotProduct(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<remove_vector_t<dotProduct_types_for_rt<T>>> &aDstScalar,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the dot product of two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the dot product is defined as: DotProduct = Sum(Src1(i,j) * Src2(i,j)) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void DotProductMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<remove_vector_t<dotProduct_types_for_rt<T>>> &aDstScalar,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the dot product of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the dot product is defined as: DotProduct = Sum(Src1(i,j) * Src2(i,j))
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void DotProduct(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes dot product of two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the dot product is defined as: DotProduct = Sum(Src1(i,j) * Src2(i,j))
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void DotProductMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MSE
    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MSEBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MSEMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the Mean Square Error of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the MSE is defined as: MSE = Sum((Src1(i,j) - Src2(i,j))^2) / (W*H) <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MSE(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
             mpp::cuda::DevVarView<remove_vector_t<mse_types_for_rt<T>>> &aDstScalar,
             mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the Mean Square Error of two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the MSE is defined as: MSE = Sum((Src1(i,j) - Src2(i,j))^2) / (W*H) <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MSEMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                   mpp::cuda::DevVarView<remove_vector_t<mse_types_for_rt<T>>> &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the Mean Square Error of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the MSE is defined as: MSE = Sum((Src1(i,j) - Src2(i,j))^2) / (W*H)
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MSE(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
             mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the Mean Square Error of two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the MSE is defined as: MSE = Sum((Src1(i,j) - Src2(i,j))^2) / (W*H)
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MSEMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MaximumError
    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaximumErrorBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaximumErrorMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the maximum error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the maximum error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumErrorMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                            mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                            const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the maximum error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the maximum error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumErrorMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                            const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MaximumRelativeError
    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaximumRelativeErrorBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaximumRelativeErrorMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the maximum relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/> For multi-channel images, the result is computed for each channel
    /// seperatly in aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute
    /// value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumRelativeError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<maxRelativeError_types_for_rt<T>>> &aDstScalar,
                              mpp::cuda::DevVarView<byte> &aBuffer,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the maximum relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumRelativeErrorMasked(
        const ImageView<T> &aSrc2, mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
        mpp::cuda::DevVarView<remove_vector_t<maxRelativeError_types_for_rt<T>>> &aDstScalar,
        const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the maximum relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumRelativeError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<byte> &aBuffer,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the maximum relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaximumRelativeErrorMasked(const ImageView<T> &aSrc2,
                                    mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region NormDiffInf
    /// <summary>
    /// Returns the required temporary buffer size for NormDiffInf.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormDiffInfBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormDiffInfMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormDiffInfMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the infinity norm of difference of pixels between two images.<para/>
    /// The infinity norm of differences is defined as: NormDiffInf = max((|Src1(i,j) - Src2(i,j)|)
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                     mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                     mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the infinity norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/>
    /// The infinity norm of differences is defined as: NormDiffInf = max((|Src1(i,j) - Src2(i,j)|)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffInfMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                           mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                           const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes infinity norm of difference of pixels between two images.<para/>
    /// The infinity norm of differences is defined as: NormDiffInf = max((|Src1(i,j) - Src2(i,j)|)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                     mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes infinity norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/> The infinity norm of differences is defined as: NormDiffInf = max((|Src1(i,j) - Src2(i,j)|)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffInfMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                           const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormDiffL1
    /// <summary>
    /// Returns the required temporary buffer size for NormDiffL1.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormDiffL1BufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormDiffL1Masked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormDiffL1MaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the L1 norm of difference of pixels between two images.<para/>
    /// The L1 norm of differences is defined as: NormDiffL1 = sum((|Src1(i,j) - Src2(i,j)|)<para/> For multi-channel
    /// images, the result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<remove_vector_t<normDiffL1_types_for_rt<T>>> &aDstScalar,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the L1 norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/>
    /// The L1 norm of differences is defined as: NormDiffL1 = sum((|Src1(i,j) - Src2(i,j)|)<para/> For multi-channel
    /// images, the result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<remove_vector_t<normDiffL1_types_for_rt<T>>> &aDstScalar,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes L1 norm of difference of pixels between two images.<para/>
    /// The L1 norm of differences is defined as: NormDiffL1 = sum((|Src1(i,j) - Src2(i,j)|)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes L1 norm of difference of pixels between two images where only pixels with mask != 0 are used.<para/>
    /// The L1 norm of differences is defined as: NormDiffL1 = sum((|Src1(i,j) - Src2(i,j)|)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormDiffL2
    /// <summary>
    /// Returns the required temporary buffer size for NormDiffL2.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormDiffL2BufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormDiffL2Masked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormDiffL2MaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the L2 norm of difference of pixels between two images.<para/>
    /// The L2 norm of differences is defined as: NormDiffL2 = sqrt(sum(((Src1(i,j) - Src2(i,j))^2))<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<remove_vector_t<normDiffL2_types_for_rt<T>>> &aDstScalar,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the L2 norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/>
    /// The L2 norm of differences is defined as: NormDiffL2 = sqrt(sum(((Src1(i,j) - Src2(i,j))^2))<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<remove_vector_t<normDiffL2_types_for_rt<T>>> &aDstScalar,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes L2 norm of difference of pixels between two images.<para/>
    /// The L2 norm of differences is defined as: NormDiffL2 = sqrt(sum(((Src1(i,j) - Src2(i,j))^2))
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes L2 norm of difference of pixels between two images where only pixels with mask != 0 are used.<para/>
    /// The L2 norm of differences is defined as: NormDiffL2 = sqrt(sum(((Src1(i,j) - Src2(i,j))^2))
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormDiffL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region NormRelInf
    /// <summary>
    /// Returns the required temporary buffer size for NormRelInf.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormRelInfBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormRelInfMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormRelInfMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the relative infinity norm of difference of pixels between two images.<para/>
    /// The relative infinity norm of differences is defined as: NormRelInf = NormDiffInf(Src1, Src2) / NormInf(Src2)
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<remove_vector_t<normRelInf_types_for_rt<T>>> &aDstScalar,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the relative infinity norm of difference of pixels between two images where only pixels with mask != 0
    /// are used.<para/> The relative infinity norm of differences is defined as: NormRelInf = NormDiffInf(Src1, Src2) /
    /// NormInf(Src2)<para/> For multi-channel images, the result is computed for each channel seperatly in aDst, or for
    /// all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelInfMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<remove_vector_t<normRelInf_types_for_rt<T>>> &aDstScalar,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes relative infinity norm of difference of pixels between two images.<para/>
    /// The relative infinity norm of differences is defined as: NormRelInf = NormDiffInf(Src1, Src2) / NormInf(Src2)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes relative infinity norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/> The relative infinity norm of differences is defined as: NormRelInf = NormDiffInf(Src1, Src2) /
    /// NormInf(Src2)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelInfMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                          const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormRelL1
    /// <summary>
    /// Returns the required temporary buffer size for NormRelL1.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormRelL1BufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormRelL1Masked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormRelL1MaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the relative L1 norm of difference of pixels between two images.<para/>
    /// The relative L1 norm of differences is defined as: NormRelL1 = NormDiffL1(Src1, Src2) / NormL1(Src2)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                   mpp::cuda::DevVarView<remove_vector_t<normRelL1_types_for_rt<T>>> &aDstScalar,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the relative L1 norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/>
    /// The relative L1 norm of differences is defined as: NormRelL1 = NormDiffL1(Src1, Src2) / NormL1(Src2)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                         mpp::cuda::DevVarView<remove_vector_t<normRelL1_types_for_rt<T>>> &aDstScalar,
                         const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes relative L1 norm of difference of pixels between two images.<para/>
    /// The relative L1 norm of differences is defined as: NormRelL1 = NormDiffL1(Src1, Src2) / NormL1(Src2)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes L1 norm of difference of pixels between two images where only pixels with mask != 0 are used.<para/>
    /// The relative L1 norm of differences is defined as: NormRelL1 = NormDiffL1(Src1, Src2) / NormL1(Src2)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                         const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormRelL2
    /// <summary>
    /// Returns the required temporary buffer size for NormRelL2.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormRelL2BufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormRelL2Masked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormRelL2MaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the relative L2 norm of difference of pixels between two images.<para/>
    /// The relative L2 norm of differences is defined as: NormRelL2 = NormDiffL2(Src1, Src2) / NormL2(Src2)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                   mpp::cuda::DevVarView<remove_vector_t<normRelL2_types_for_rt<T>>> &aDstScalar,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the relative L2 norm of difference of pixels between two images where only pixels with mask != 0 are
    /// used.<para/>
    /// The relative L2 norm of differences is defined as: NormRelL2 = NormDiffL2(Src1, Src2) / NormL2(Src2)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                         mpp::cuda::DevVarView<remove_vector_t<normRelL2_types_for_rt<T>>> &aDstScalar,
                         const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes relative L2 norm of difference of pixels between two images.<para/>
    /// The relative L2 norm of differences is defined as: NormRelL2 = NormDiffL2(Src1, Src2) / NormL2(Src2)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes L2 norm of difference of pixels between two images where only pixels with mask != 0 are used.<para/>
    /// The relative L2 norm of differences is defined as: NormRelL2 = NormDiffL2(Src1, Src2) / NormL2(Src2)
    /// </summary>
    /// <param name="aSrc2">Second image to compare this image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormRelL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                         const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                         const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region PSNR
    /// <summary>
    /// Returns the required temporary buffer size for PSNR.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t PSNRBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the PSNR of two images. <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aValueRange">The maximum possible pixel value, eg. 255 for 8 bit unsigned int images, 4095 for
    /// 12-bit unsigned images, etc.</param> <param name="aBuffer">Temporary device memory buffer for
    /// computation.</param> <param name="aStreamCtx"></param>
    void PSNR(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
              mpp::cuda::DevVarView<remove_vector_t<mse_types_for_rt<T>>> &aDstScalar,
              remove_vector_t<mse_types_for_rt<T>> aValueRange, mpp::cuda::DevVarView<byte> &aBuffer,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the PSNR of two images.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aValueRange">The maximum possible pixel value, eg. 255 for 8 bit unsigned int images, 4095 for
    /// 12-bit unsigned images, etc.</param> <param name="aBuffer">Temporary device memory buffer for
    /// computation.</param> <param name="aStreamCtx"></param>
    void PSNR(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
              remove_vector_t<mse_types_for_rt<T>> aValueRange, mpp::cuda::DevVarView<byte> &aBuffer,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region NormInf
    /// <summary>
    /// Returns the required temporary buffer size for NormInf.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormInfBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormInfMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormInfMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the infinity norm.<para/>
    /// The infinity norm is defined as: NormInf = max(|Src1(i,j)|)
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormInf(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst,
                 mpp::cuda::DevVarView<remove_vector_t<normInf_types_for_rt<T>>> &aDstScalar,
                 mpp::cuda::DevVarView<byte> &aBuffer,
                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the infinity norm where only pixels with mask != 0 are
    /// used.<para/>
    /// The infinity norm is defined as: NormInf = max(|Src1(i,j)|)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormInfMasked(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst,
                       mpp::cuda::DevVarView<remove_vector_t<normInf_types_for_rt<T>>> &aDstScalar,
                       const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes infinity norm.<para/>
    /// The infinity norm is defined as: NormInf = max(|Src1(i,j)|)
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormInf(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes infinity norm where only pixels with mask != 0 are
    /// used.<para/> The infinity norm is defined as: NormInf = max(|Src1(i,j)|)
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormInfMasked(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                       mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormL1
    /// <summary>
    /// Returns the required temporary buffer size for NormL1.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormL1BufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormL1Masked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormL1MaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the L1 norm.<para/>
    /// The L1 norm is defined as: NormL1 = sum(|Src1(i,j)|)
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL1(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst,
                mpp::cuda::DevVarView<remove_vector_t<normL1_types_for_rt<T>>> &aDstScalar,
                mpp::cuda::DevVarView<byte> &aBuffer,
                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the L1 norm where only pixels with mask != 0 are
    /// used.<para/>
    /// The L1 norm is defined as: NormL1 = sum(|Src1(i,j)|)<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL1Masked(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<remove_vector_t<normL1_types_for_rt<T>>> &aDstScalar,
                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes L1 norm.<para/>
    /// The L1 norm is defined as: NormL1 = sum(|Src1(i,j)|)
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL1(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes L1 norm where only pixels with mask != 0 are
    /// used.<para/> The L1 norm is defined as: NormL1 = sum(|Src1(i,j)|)
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL1Masked(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region NormL2
    /// <summary>
    /// Returns the required temporary buffer size for NormL2.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormL2BufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for NormL2Masked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t NormL2MaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the L2 norm.<para/>
    /// The L2 norm is defined as: NormL2 = sqrt(sum(Src1(i,j)^2))
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL2(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst,
                mpp::cuda::DevVarView<remove_vector_t<normL2_types_for_rt<T>>> &aDstScalar,
                mpp::cuda::DevVarView<byte> &aBuffer,
                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the L2 norm where only pixels with mask != 0 are
    /// used.<para/>
    /// The L2 norm is defined as: NormL2 = sqrt(sum(Src1(i,j)^2))<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL2Masked(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<remove_vector_t<normL2_types_for_rt<T>>> &aDstScalar,
                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes L2 norm.<para/>
    /// The L2 norm is defined as: NormL2 = sqrt(sum(Src1(i,j)^2))
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL2(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes L2 norm where only pixels with mask != 0 are
    /// used.<para/> The L2 norm is defined as: NormL2 = sqrt(sum(Src1(i,j)^2))
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void NormL2Masked(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region Sum
    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    [[nodiscard]] size_t SumBufferSize(
        const mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    [[nodiscard]] size_t SumMaskedBufferSize(
        const mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    [[nodiscard]] size_t SumBufferSize(
        const mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    [[nodiscard]] size_t SumMaskedBufferSize(
        const mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
             mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 1>>> &aDstScalar,
             mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
             mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 2>>> &aDstScalar,
             mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
                   mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 1>>> &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
                   mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 2>>> &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the sum of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the sum of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst, const ImageView<Pixel8uC1> &aMask,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst, const ImageView<Pixel8uC1> &aMask,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexIntVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region Mean
    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MeanBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MeanMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Mean(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst,
              mpp::cuda::DevVarView<remove_vector_t<mean_types_for_rt<T>>> &aDstScalar,
              mpp::cuda::DevVarView<byte> &aBuffer,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MeanMasked(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst,
                    mpp::cuda::DevVarView<remove_vector_t<mean_types_for_rt<T>>> &aDstScalar,
                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the mean of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Mean(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MeanMasked(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                    mpp::cuda::DevVarView<byte> &aBuffer,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region MeanStd
    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MeanStdBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;
    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MeanStdMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar also nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MeanStd(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                 mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd,
                 mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt1<T>>> &aMeanScalar,
                 mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt2<T>>> &aStdScalar,
                 mpp::cuda::DevVarView<byte> &aBuffer,
                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values where only pixels with mask != 0 are used.<para/>For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar also nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MeanStdMasked(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                       mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd,
                       mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt1<T>>> &aMeanScalar,
                       mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt2<T>>> &aStdScalar,
                       const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.
    /// </summary>
    /// <param name="aMean">Mean value</param>
    /// <param name="aStd">Standard deviation</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MeanStd(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                 mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd, mpp::cuda::DevVarView<byte> &aBuffer,
                 const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aMean">Mean value</param>
    /// <param name="aStd">Standard deviation</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MeanStdMasked(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                       mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd, const ImageView<Pixel8uC1> &aMask,
                       mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_active_size_v<T> == 1);
#pragma endregion

#pragma region CountInRange
    /// <summary>
    /// Returns the required temporary buffer size for CountInRange.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t CountInRangeBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for CountInRangeMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t CountInRangeMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Counts the pixels in a given value range.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aLowerLimit">Lower bound of the specified range (inclusive).</param>
    /// <param name="aUpperLimit">Upper bound of the specified range (inclusive).</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                      mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                      mpp::cuda::DevVarView<size_t> &aDstScalar, mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Counts the pixels in a given value range where only pixels with mask != 0 are used.<para/>For multi-channel
    /// images, the result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aLowerLimit">Lower bound of the specified range (inclusive).</param>
    /// <param name="aUpperLimit">Upper bound of the specified range (inclusive).</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                            mpp::cuda::DevVarView<size_t> &aDstScalar, const ImageView<Pixel8uC1> &aMask,
                            mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Counts the pixels in a given value range.
    /// </summary>
    /// <param name="aLowerLimit">Lower bound of the specified range (inclusive).</param>
    /// <param name="aUpperLimit">Upper bound of the specified range (inclusive).</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                      mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Counts the pixels in a given value range where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aLowerLimit">Lower bound of the specified range (inclusive).</param>
    /// <param name="aUpperLimit">Upper bound of the specified range (inclusive).</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                            const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region QualityIndex
    /// <summary>
    /// Returns the required temporary buffer size for QualityIndex.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t QualityIndexBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the image quality index of two images. This implementation is identical to the one in NPP computing a
    /// global index without a sliding window.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per channel result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void QualityIndex(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<qualityIndex_types_for_rt<T>> &aDst,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for QualityIndexWindow.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t QualityIndexWindowBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the QualityIndex of two images. This function is implemented using a sliding window approach as is done
    /// in the original paper / code with a window size of 11x11 pixels.
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Per channel result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void QualityIndexWindow(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<qiw_types_for_rt<T>> &aDst,
                            mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion

#pragma region SSIM
    /// <summary>
    /// Returns the required temporary buffer size for SSIM.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t SSIMBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the SSIM of two images.
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
    void SSIM(
        const ImageView<T> &aSrc2, mpp::cuda::DevVarView<ssim_types_for_rt<T>> &aDst,
        mpp::cuda::DevVarView<byte> &aBuffer,
        remove_vector_t<ssim_types_for_rt<T>> aDynamicRange = static_cast<remove_vector_t<ssim_types_for_rt<T>>>(1.0),
        remove_vector_t<ssim_types_for_rt<T>> aK1           = static_cast<remove_vector_t<ssim_types_for_rt<T>>>(0.01),
        remove_vector_t<ssim_types_for_rt<T>> aK2           = static_cast<remove_vector_t<ssim_types_for_rt<T>>>(0.03),
        const mpp::cuda::StreamCtx &aStreamCtx              = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion

#pragma region MSSSIM
    /// <summary>
    /// Returns the required temporary buffer size for MSSSIM.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MSSSIMBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

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
    void MSSSIM(
        const ImageView<T> &aSrc2, mpp::cuda::DevVarView<ssim_types_for_rt<T>> &aDst,
        mpp::cuda::DevVarView<byte> &aBuffer,
        remove_vector_t<ssim_types_for_rt<T>> aDynamicRange = static_cast<remove_vector_t<ssim_types_for_rt<T>>>(1.0),
        remove_vector_t<ssim_types_for_rt<T>> aK1           = static_cast<remove_vector_t<ssim_types_for_rt<T>>>(0.01),
        remove_vector_t<ssim_types_for_rt<T>> aK2           = static_cast<remove_vector_t<ssim_types_for_rt<T>>>(0.03),
        const mpp::cuda::StreamCtx &aStreamCtx              = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
#pragma endregion

#pragma region Min
    /// <summary>
    /// Returns the required temporary buffer size for Min.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for MinMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Minimum pixel value.
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Min(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
             mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum pixel value where only pixels with mask != 0 are used.<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMasked(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum pixel value.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Min(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Minimum pixel value where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMasked(mpp::cuda::DevVarView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region Max
    /// <summary>
    /// Returns the required temporary buffer size for Max.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaxBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for MaxMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaxMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Maximum pixel value.
    /// <para/> For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Max(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
             mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Maximum pixel value where only pixels with mask != 0 are used.<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaxMasked(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Maximum pixel value.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void Max(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
             const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Maximum pixel value where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaxMasked(mpp::cuda::DevVarView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                   mpp::cuda::DevVarView<byte> &aBuffer,
                   const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MinMax
    /// <summary>
    /// Returns the required temporary buffer size for MinMax.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinMaxBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for MinMaxMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinMaxMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Minimum and maximum pixel value.<para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDstMin/aDstMax, or for all
    /// channels in aDstMinScalar/aDstMaxScalar.
    /// </summary>
    /// <param name="aDstMin">Per-channel minimum value, can be nullptr if aDstMax is also nullptr</param>
    /// <param name="aDstMax">Per-channel maximum value, can be nullptr if aDstMin is also nullptr</param>
    /// <param name="aDstMinScalar">Minimum value for all channels, can be nullptr if aDstMaxScalar also nullptr</param>
    /// <param name="aDstMaxScalar">Maximum value for all channels, can be nullptr if aDstMinScalar also nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMax(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar, mpp::cuda::DevVarView<byte> &aBuffer,
                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);
    /// <summary>
    /// Minimum and maximum pixel value where only pixels with mask != 0 are used.<para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDstMin/aDstMax, or for all
    /// channels in aDstMinScalar/aDstMaxScalar.
    /// </summary>
    /// <param name="aDstMin">Per-channel minimum value, can be nullptr if aDstMax is also nullptr</param>
    /// <param name="aDstMax">Per-channel maximum value, can be nullptr if aDstMin is also nullptr</param>
    /// <param name="aDstMinScalar">Minimum value for all channels, can be nullptr if aDstMaxScalar also nullptr</param>
    /// <param name="aDstMaxScalar">Maximum value for all channels, can be nullptr if aDstMinScalar also nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMaxMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                      mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                      mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar, const ImageView<Pixel8uC1> &aMask,
                      mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum and maximum pixel value.
    /// </summary>
    /// <param name="aDstMin">Minimum value</param>
    /// <param name="aDstMax">Maximum value</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMax(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                mpp::cuda::DevVarView<byte> &aBuffer,
                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Minimum and maximum pixel value where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDstMin">Minimum value</param>
    /// <param name="aDstMax">Maximum value</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMaxMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                      const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MinIndex
    /// <summary>
    /// Returns the required temporary buffer size for MinIndex.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinIndexBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for MinIndexMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinIndexMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Minimum pixel value and its pixel index. For multiple occurences of the same value the index with the lowest
    /// flattened index (y * width + x) is returned.<para/> For multi-channel images, the result is computed for each
    /// channel seperatly in aDstMin/aDstIndexX/aDstIndexY, or for all channels in aDstMinScalar/aDstScalarIdx.
    /// </summary>
    /// <param name="aDstMin">Per-channel minimum value, can be nullptr if aDstIndexX and aDstIndexY are also
    /// nullptr</param>
    /// <param name="aDstIndexX">Per-channel X pixel index, can be nullptr if aDstMin and aDstIndexY are
    /// also nullptr</param>
    /// <param name="aDstIndexY">Per-channel Y pixel index, can be nullptr if aDstMin and
    /// aDstIndexX are also nullptr</param>
    /// <param name="aDstMinScalar">Minimum value for all channels, can be nullptr
    /// if aDstScalarIdx also nullptr</param>
    /// <param name="aDstScalarIdx">Pixel index of the minimum value, the .z
    /// component gives the image channel of the value. Can be nullptr if aDstMinScalar also nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinIndex(mpp::cuda::DevVarView<T> &aDstMin,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                  mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                  mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum pixel value and its pixel index where only pixels with mask != 0 are used. For multiple occurences of
    /// the same value the index with the lowest flattened index (y * width + x) is returned.<para/> For multi-channel
    /// images, the result is computed for each channel seperatly in aDstMin/aDstIndexX/aDstIndexY, or for all channels
    /// in aDstMinScalar/aDstScalarIdx.
    /// </summary>
    /// <param name="aDstMin">Per-channel minimum value, can be nullptr if aDstIndexX and aDstIndexY are also
    /// nullptr</param>
    /// <param name="aDstIndexX">Per-channel X pixel index, can be nullptr if aDstMin and aDstIndexY are
    /// also nullptr</param>
    /// <param name="aDstIndexY">Per-channel Y pixel index, can be nullptr if aDstMin and
    /// aDstIndexX are also nullptr</param>
    /// <param name="aDstMinScalar">Minimum value for all channels, can be nullptr
    /// if aDstScalarIdx also nullptr</param>
    /// <param name="aDstScalarIdx">Pixel index of the minimum value, the .z
    /// component gives the image channel of the value. Can be nullptr if aDstMinScalar also nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinIndexMasked(mpp::cuda::DevVarView<T> &aDstMin,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                        mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                        mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask,
                        mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum pixel value and its pixel index. For multiple occurences of the same value the index with the lowest
    /// flattened index (y * width + x) is returned.
    /// </summary>
    /// <param name="aDstMin">Minimum value</param>
    /// <param name="aDstIndexX">X pixel index</param>
    /// <param name="aDstIndexY">Y pixel index</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinIndex(mpp::cuda::DevVarView<T> &aDstMin,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                  mpp::cuda::DevVarView<byte> &aBuffer,
                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Minimum pixel value and its pixel index where only pixels with mask != 0 are used. For multiple occurences of
    /// the same value the index with the lowest flattened index (y * width + x) is returned.
    /// </summary>
    /// <param name="aDstMin">Minimum value</param>
    /// <param name="aDstIndexX">X pixel index</param>
    /// <param name="aDstIndexY">Y pixel index</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinIndexMasked(mpp::cuda::DevVarView<T> &aDstMin,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                        const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MaxIndex
    /// <summary>
    /// Returns the required temporary buffer size for MinIndex.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaxIndexBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for MinIndexMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MaxIndexMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Maximum pixel value and its pixel index. For multiple occurences of the same value the index with the lowest
    /// flattened index (y * width + x) is returned.<para/> For multi-channel images, the result is computed for each
    /// channel seperatly in aDstMin/aDstIndexX/aDstIndexY, or for all channels in aDstMinScalar/aDstScalarIdx.
    /// </summary>
    /// <param name="aDstMin">Per-channel maximum value, can be nullptr if aDstIndexX and aDstIndexY are also
    /// nullptr</param>
    /// <param name="aDstIndexX">Per-channel X pixel index, can be nullptr if aDstMin and aDstIndexY are
    /// also nullptr</param>
    /// <param name="aDstIndexY">Per-channel Y pixel index, can be nullptr if aDstMin and
    /// aDstIndexX are also nullptr</param>
    /// <param name="aDstMinScalar">Maximum value for all channels, can be nullptr
    /// if aDstScalarIdx also nullptr</param>
    /// <param name="aDstScalarIdx">Pixel index of the maximum value, the .z
    /// component gives the image channel of the value. Can be nullptr if aDstMinScalar also nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaxIndex(mpp::cuda::DevVarView<T> &aDstMax,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                  mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                  mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Maximum pixel value and its pixel index where only pixels with mask != 0 are used. For multiple occurences of
    /// the same value the index with the lowest flattened index (y * width + x) is returned.<para/> For multi-channel
    /// images, the result is computed for each channel seperatly in aDstMin/aDstIndexX/aDstIndexY, or for all channels
    /// in aDstMinScalar/aDstScalarIdx.
    /// </summary>
    /// <param name="aDstMin">Per-channel maximum value, can be nullptr if aDstIndexX and aDstIndexY are also
    /// nullptr</param>
    /// <param name="aDstIndexX">Per-channel X pixel index, can be nullptr if aDstMin and aDstIndexY are
    /// also nullptr</param>
    /// <param name="aDstIndexY">Per-channel Y pixel index, can be nullptr if aDstMin and
    /// aDstIndexX are also nullptr</param>
    /// <param name="aDstMinScalar">Maximum value for all channels, can be nullptr
    /// if aDstScalarIdx also nullptr</param>
    /// <param name="aDstScalarIdx">Pixel index of the maximum value, the .z
    /// component gives the image channel of the value. Can be nullptr if aDstMinScalar also nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMax,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                        mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                        mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask,
                        mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Maximum pixel value and its pixel index. For multiple occurences of the same value the index with the lowest
    /// flattened index (y * width + x) is returned.
    /// </summary>
    /// <param name="aDstMin">Maximum value</param>
    /// <param name="aDstIndexX">X pixel index</param>
    /// <param name="aDstIndexY">Y pixel index</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaxIndex(mpp::cuda::DevVarView<T> &aDstMax,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                  mpp::cuda::DevVarView<byte> &aBuffer,
                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Maximum pixel value and its pixel index where only pixels with mask != 0 are used. For multiple occurences of
    /// the same value the index with the lowest flattened index (y * width + x) is returned.
    /// </summary>
    /// <param name="aDstMin">Maximum value</param>
    /// <param name="aDstIndexX">X pixel index</param>
    /// <param name="aDstIndexY">Y pixel index</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMax,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                        mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                        const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion
#pragma region MinMaxIndex
    /// <summary>
    /// Returns the required temporary buffer size for MinMaxIndex.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinMaxIndexBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// Returns the required temporary buffer size for MinMaxIndexMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t MinMaxIndexMaskedBufferSize(
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Minimum and maximum pixel value and their pixel indices. For multiple occurences of the same value the index
    /// with the lowest flattened index (y * width + x) is returned.<para/> For multi-channel images, the result is
    /// computed for each channel seperatly in aDstMin/aDstMax/aDstIdx, or for all channels in
    /// aDstMinScalar/aDstMaxScalar/aDstScalarIdx.
    /// </summary>
    /// <param name="aDstMin">Minimum value per channel (array with size of active channels).
    /// Can be nullptr if aDstMax and aDstIdx are also nullptr.</param>
    /// <param name="aDstMax">Maximum value per channel (array with size of active channels).
    /// Can be nullptr if aDstMin and aDstIdx are also nullptr.</param>
    /// <param name="aDstIdx">Pixel index for min and max value per channel (array with size of active channels).
    /// Can be nullptr if aDstMin and aDstMax are also nullptr.</param>
    /// <param name="aDstMinScalar">Minimum value for all channels (array with size of 1).
    /// Can be nullptr if aDstMaxScalar and aDstScalarIdx are also nullptr.</param>
    /// <param name="aDstMaxScalar">Maximum value for all channels (array with size of 1).
    /// Can be nullptr if aDstMinScalar and aDstScalarIdx are also nullptr.</param>
    /// <param name="aDstScalarIdx">Pixel index for min and max value for all channels (array with size of 1).
    /// Can be nullptr if aDstMinScalar and aDstMaxScalar are also nullptr.</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMaxIndex(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                     mpp::cuda::DevVarView<IndexMinMax> &aDstIdx,
                     mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                     mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                     mpp::cuda::DevVarView<IndexMinMaxChannel> &aDstScalarIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum and maximum pixel value and their pixel indices where only pixels with mask != 0 are used. For multiple
    /// occurences of the same value the index with the lowest flattened index (y * width + x) is returned.<para/> For
    /// multi-channel images, the result is computed for each channel seperatly in aDstMin/aDstMax/aDstIdx, or for all
    /// channels in aDstMinScalar/aDstMaxScalar/aDstScalarIdx.
    /// </summary>
    /// <param name="aDstMin">Minimum value per channel (array with size of active channels).
    /// Can be nullptr if aDstMax and aDstIdx are also nullptr.</param>
    /// <param name="aDstMax">Maximum value per channel (array with size of active channels).
    /// Can be nullptr if aDstMin and aDstIdx are also nullptr.</param>
    /// <param name="aDstIdx">Pixel index for min and max value per channel (array with size of active channels).
    /// Can be nullptr if aDstMin and aDstMax are also nullptr.</param>
    /// <param name="aDstMinScalar">Minimum value for all channels (array with size of 1).
    /// Can be nullptr if aDstMaxScalar and aDstScalarIdx are also nullptr.</param>
    /// <param name="aDstMaxScalar">Maximum value for all channels (array with size of 1).
    /// Can be nullptr if aDstMinScalar and aDstScalarIdx are also nullptr.</param>
    /// <param name="aDstScalarIdx">Pixel index for min and max value for all channels (array with size of 1).
    /// Can be nullptr if aDstMinScalar and aDstMaxScalar are also nullptr.</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                           mpp::cuda::DevVarView<IndexMinMax> &aDstIdx,
                           mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                           mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                           mpp::cuda::DevVarView<IndexMinMaxChannel> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask,
                           mpp::cuda::DevVarView<byte> &aBuffer,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Minimum and maximum pixel value and their pixel indices. For multiple occurences of the same value the index
    /// with the lowest flattened index (y * width + x) is returned.
    /// </summary>
    /// <param name="aDstMin">Minimum value</param>
    /// <param name="aDstMax">Maximum value</param>
    /// <param name="aDstIdx">Pixel indices for min/max value</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMaxIndex(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                     mpp::cuda::DevVarView<IndexMinMax> &aDstIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
    /// <summary>
    /// Minimum and maximum pixel value and their pixel indices where only pixels with mask != 0 are used. For multiple
    /// occurences of the same value the index with the lowest flattened index (y * width + x) is returned.
    /// </summary>
    /// <param name="aDstMin">Minimum value</param>
    /// <param name="aDstMax">Maximum value</param>
    /// <param name="aDstIdx">Pixel indices for min/max value</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void MinMaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                           mpp::cuda::DevVarView<IndexMinMax> &aDstIdx, const ImageView<Pixel8uC1> &aMask,
                           mpp::cuda::DevVarView<byte> &aBuffer,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region Integral
    /// <summary>
    /// Returns the required temporary buffer size for Integral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t IntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, int>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;
    /// <summary>
    /// Returns the required temporary buffer size for Integral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t IntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>);
    /// <summary>
    /// Returns the required temporary buffer size for Integral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t IntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, long64>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;
    /// <summary>
    /// Returns the required temporary buffer size for Integral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t IntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, double>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T>;
    /// <summary>
    /// Returns the required temporary buffer size for SqrIntegral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t SqrIntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, int>> &aDst,
        ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;
    /// <summary>
    /// Returns the required temporary buffer size for SqrIntegral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t SqrIntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, int>> &aDst,
        ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;
    /// <summary>
    /// Returns the required temporary buffer size for SqrIntegral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t SqrIntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>);
    /// <summary>
    /// Returns the required temporary buffer size for SqrIntegral.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    [[nodiscard]] size_t SqrIntegralBufferSize(
        ImageView<same_vector_size_different_type_t<T, double>> &aDst,
        ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    /// <returns></returns>
    ImageView<same_vector_size_different_type_t<T, int>> &Integral(
        ImageView<same_vector_size_different_type_t<T, int>> &aDst,
        const same_vector_size_different_type_t<T, int> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    /// <returns></returns>
    ImageView<same_vector_size_different_type_t<T, float>> &Integral(
        ImageView<same_vector_size_different_type_t<T, float>> &aDst,
        const same_vector_size_different_type_t<T, float> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>);

    /// <summary>
    /// Computes the integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    /// <returns></returns>
    ImageView<same_vector_size_different_type_t<T, long64>> &Integral(
        ImageView<same_vector_size_different_type_t<T, long64>> &aDst,
        const same_vector_size_different_type_t<T, long64> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    /// <returns></returns>
    ImageView<same_vector_size_different_type_t<T, double>> &Integral(
        ImageView<same_vector_size_different_type_t<T, double>> &aDst,
        const same_vector_size_different_type_t<T, double> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the integral image and the squared integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aSqr">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels.</param>
    /// <param name="aValSqr">The value to add to aSqr image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                     const same_vector_size_different_type_t<T, int> &aVal,
                     const same_vector_size_different_type_t<T, int> &aValSqr, mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the integral image and the squared integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aSqr">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels.</param>
    /// <param name="aValSqr">The value to add to aSqr image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                     const same_vector_size_different_type_t<T, int> &aVal,
                     const same_vector_size_different_type_t<T, long64> &aValSqr, mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealIntVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the integral image and the squared integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aSqr">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels.</param>
    /// <param name="aValSqr">The value to add to aSqr image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                     const same_vector_size_different_type_t<T, float> &aVal,
                     const same_vector_size_different_type_t<T, double> &aValSqr, mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>);

    /// <summary>
    /// Computes the integral image and the squared integral image.
    /// </summary>
    /// <param name="aDst">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aSqr">ROI of destination image must be 1 pixel larger in width and height than source ROI.</param>
    /// <param name="aVal">The value to add to aDst image pixels.</param>
    /// <param name="aValSqr">The value to add to aSqr image pixels</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    void SqrIntegral(ImageView<same_vector_size_different_type_t<T, double>> &aDst,
                     ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                     const same_vector_size_different_type_t<T, double> &aVal,
                     const same_vector_size_different_type_t<T, double> &aValSqr, mpp::cuda::DevVarView<byte> &aBuffer,
                     const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T> && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(std::same_as<remove_vector_t<T>, float>) && NoAlpha<T>;

    /// <summary>
    /// Computes the standard deviation from integral square images.
    /// </summary>
    void RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                    ImageView<same_vector_size_different_type_t<T, double>> &aDst, const FilterArea &aFilterArea,
                    const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(std::same_as<remove_vector_t<T>, double>) && NoAlpha<T>;
#pragma endregion
#pragma region MinEvery
    /// <summary>
    /// aDst = min(this, aSrc2) (minimum per pixel, per channel)
    /// </summary>
    ImageView<T> &MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// this = min(this, aSrc2) (minimum per pixel, per channel)
    /// </summary>
    ImageView<T> &MinEvery(const ImageView<T> &aSrc2,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
#pragma endregion
#pragma region MaxEvery
    /// <summary>
    /// aDst = max(this, aSrc2) (maximum per pixel, per channel)
    /// </summary>
    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// this = max(this, aSrc2) (maximum per pixel, per channel)
    /// </summary>
    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2,
                           const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
#pragma endregion

#pragma region Histogram
    /// <summary>
    /// Compute levels with even distribution, depending on aHistorgamEvenMode, this function tries to give identical
    /// results as the same function in NPP or as the methods used in the CUB backend used by MPP for histogram
    /// computation.
    /// </summary>
    /// <param name="aHPtrLevels">A host pointer to array which receives the levels being computed.
    /// The array needs to be of size aLevels.</ param>
    /// <param name="aLevels">The number of levels being computed. aLevels must be at least 2</param>
    /// <param name="aLowerLevel">Lower boundary value of the lowest level.</param>
    /// <param name="aUpperLevel">Upper boundary value of the greatest level.</param>
    /// <param name="aHistorgamEvenMode">Switch compatibility mode: CUB (default) or NPP.</param>
    void EvenLevels(int *aHPtrLevels, int aLevels, int aLowerLevel, int aUpperLevel,
                    HistorgamEvenMode aHistorgamEvenMode = HistorgamEvenMode::Default);

    /// <summary>
    /// Returns the required temporary buffer size for HistogramEven.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aLevels">aLevels - 1 = number of histogram bins, per channel</param>
    [[nodiscard]] size_t HistogramEvenBufferSize(
        const same_vector_size_different_type_t<T, int> &aLevels,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// The aLowerLevel (inclusive) and aUpperLevel (exclusive) define the boundaries of the range,
    /// which are evenly segmented into aHist.Size() bins.
    /// </summary>
    /// <param name="aHist">host array of device memory pointers to the computed histograms
    /// (one for each active channel). The size of aHist[channel] gives the number of bins used.</param>
    /// <param name="aLowerLevel">lower level (inclusive, per channel)</param>
    /// <param name="aUpperLevel">upper level (exclusive, per channel)</param>
    /// <param name="aBuffer"></param>
    /// <param name="aStreamCtx"></param>
    void HistogramEven(mpp::cuda::DevVarView<int> aHist[vector_active_size_v<T>],
                       const hist_even_level_types_for_t<T> &aLowerLevel,
                       const hist_even_level_types_for_t<T> &aUpperLevel, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// The aLowerLevel (inclusive) and aUpperLevel (exclusive) define the boundaries of the range,
    /// which are evenly segmented into aHist.Size() bins.
    /// </summary>
    /// <param name="aHist">device memory pointer to the computed histogram.
    /// The size of aHist gives the number of bins used.</param>
    /// <param name="aLowerLevel">lower level (inclusive)</param>
    /// <param name="aUpperLevel">upper level (exclusive)</param>
    /// <param name="aBuffer"></param>
    /// <param name="aStreamCtx"></param>
    void HistogramEven(mpp::cuda::DevVarView<int> &aHist, const hist_even_level_types_for_t<T> &aLowerLevel,
                       const hist_even_level_types_for_t<T> &aUpperLevel, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T> && (vector_active_size_v<T> == 1);

    /// <summary>
    /// Returns the required temporary buffer size for HistogramRange.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aNumLevels">aNumLevels - 1 = number of histogram bins, per channel</param>
    [[nodiscard]] size_t HistogramRangeBufferSize(
        const same_vector_size_different_type_t<T, int> &aNumLevels,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;

    /// <summary>
    /// Computes the histogram of an image within specified ranges.
    /// </summary>
    /// <param name="aHist">host array of device memory pointers to the computed histograms
    /// (one for each active channel). The size of aHist[channel] gives the number of bins used.</param>
    /// <param name="aLevels">host array of device memory pointers to the array
    /// with the range defintions, one array per channel. The levels array must be one element
    /// larger than the histogram array, as number of levels = number of bins + 1.</param>
    /// <param name="aBuffer"></param>
    /// <param name="aStreamCtx"></param>
    void HistogramRange(mpp::cuda::DevVarView<int> aHist[vector_active_size_v<T>],
                        mpp::cuda::DevVarView<hist_range_types_for_t<T>> aLevels[vector_active_size_v<T>],
                        mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T> && (vector_active_size_v<T> > 1);

    /// <summary>
    /// Computes the histogram of an image within specified ranges.
    /// </summary>
    /// <param name="aHist">device memory pointer to the computed histogram.
    /// The size of aHist gives the number of bins used.</param>
    /// <param name="aLevels">device memory pointer to the array with the range defintion.
    /// The levels array must be one element larger than the histogram array, as number of
    /// levels = number of bins + 1.</param>
    /// <param name="aBuffer"></param>
    /// <param name="aStreamCtx"></param>
    void HistogramRange(mpp::cuda::DevVarView<int> &aHist,
                        const mpp::cuda::DevVarView<hist_range_types_for_t<T>> &aLevels,
                        mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T> && (vector_active_size_v<T> == 1);
#pragma endregion

#pragma region Cross Correlation
    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);
    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(
        const ImageView<Pixel32fC2> &aSrcBoxFiltered, const ImageView<T> &aTemplate,
        const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant,
        BorderType aBorder, const Roi &aAllowedReadRoi,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(
        const ImageView<Pixel32fC2> &aSrcBoxFiltered, const ImageView<T> &aTemplate,
        const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder,
        const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);
    /// <summary>
    /// Computes the un-normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelation(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized cross-correlation.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the normalized squared distance.<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &SquareDistanceNormalized(
        const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(
        const ImageView<Pixel32fC2> &aSrcBoxFiltered, const ImageView<T> &aTemplate,
        const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate, ImageView<Pixel32fC1> &aDst, const T &aConstant,
        BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

    /// <summary>
    /// Computes the cross-correlation coefficient (CrossCorr_NormLevel in NPP).<para/>
    /// Note: in order to compute the common "full" or "same" variant as e.g. in NPP, set the input and output ROIs
    /// accordingly and use BorderType::Constant with aConstant = 0.
    /// </summary>
    ImageView<Pixel32fC1> &CrossCorrelationCoefficient(
        const ImageView<Pixel32fC2> &aSrcBoxFiltered, const ImageView<T> &aTemplate,
        const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate, ImageView<Pixel32fC1> &aDst, BorderType aBorder,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8);

#pragma endregion
#pragma endregion

#pragma region Threshold and Compare
#pragma region Compare
    /// <summary>
    /// aDst pixel is set to 255 if this and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// aDst pixel is set to 255 if this and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// aDst pixel is set to 255 if this and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare,
                                  ImageView<Pixel8uC1> &aDst,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// aDst pixel is set to 255 if this fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel
    /// images:<para/> CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/> CompareOp::Eq |
    /// CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                  const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst pixel is set to 255 if this and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> > 1);

    /// <summary>
    /// aDst pixel is set to 255 if this and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        const T &aConst, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> > 1);

    /// <summary>
    /// aDst pixel is set to 255 if this and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare,
        ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires(vector_size_v<T> > 1);

    /// <summary>
    /// aDst pixel is set to 255 if this fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The comparison is performed for each channel individually and the flag CompareOp::PerChannel
    /// must be set for aCompare.
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, byte>> &Compare(
        CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T> && (vector_size_v<T> > 1);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(this - aSrc2) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &CompareEqEps(
        const ImageView<T> &aSrc2, complex_basetype_t<remove_vector_t<T>> aEpsilon, ImageView<Pixel8uC1> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(this - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &CompareEqEps(
        const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon, ImageView<Pixel8uC1> &aDst,
        const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(this - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &CompareEqEps(
        const mpp::cuda::DevVarView<T> &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
        ImageView<Pixel8uC1> &aDst, const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;
#pragma endregion
#pragma region Threshold
    /// <summary>
    /// If for a comparison operation aCompare the predicate (sourcePixel aCompare nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation aCompare the predicate (sourcePixel aCompare nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &Threshold(const mpp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare, ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation less than the predicate (sourcePixel &lt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdLT(const T &aThreshold, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation less than the predicate (sourcePixel &lt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdLT(const mpp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation greater than the predicate (sourcePixel &gt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdGT(const T &aThreshold, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation greater than the predicate (sourcePixel &gt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdGT(const mpp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation aCompare the predicate (sourcePixel aCompare nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation aCompare the predicate (sourcePixel aCompare nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &Threshold(const mpp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation less than the predicate (sourcePixel &lt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdLT(const T &aThreshold,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation less than the predicate (sourcePixel &lt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdLT(const mpp::cuda::DevVarView<T> &aThreshold,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation greater than the predicate (sourcePixel &gt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdGT(const T &aThreshold,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation greater than the predicate (sourcePixel &gt; nThreshold) is true, the pixel is set
    /// to aThreshold, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdGT(const mpp::cuda::DevVarView<T> &aThreshold,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;

    /// <summary>
    /// If for a comparison operation aCompare the predicate (sourcePixel aCompare nThreshold) is true, the pixel is set
    /// to aValue, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation less than the predicate (sourcePixel &lt; nThreshold) is true, the pixel is set
    /// to aValue, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation greater than the predicate (sourcePixel &gt; nThreshold) is true, the pixel is set
    /// to aValue, otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation aCompare the predicate (sourcePixel aCompare nThreshold) is true, the pixel is set
    /// to aValue, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation less than the predicate (sourcePixel &lt; nThreshold) is true, the pixel is set
    /// to aValue, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation greater than the predicate (sourcePixel &gt; nThreshold) is true, the pixel is set
    /// to aValue, otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue,
                              const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation sourcePixel is less than aThresholdLT is true, the pixel is set
    /// to aValueLT, else if sourcePixel is greater than aThresholdGT the pixel is set to aValueGT,
    /// otherwise it is set to sourcePixel.
    /// </summary>
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT,
                                ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealVector<T>;
    /// <summary>
    /// If for a comparison operation sourcePixel is less than aThresholdLT is true, the pixel is set
    /// to aValueLT, else if sourcePixel is greater than aThresholdGT the pixel is set to aValueGT,
    /// otherwise it is set to sourcePixel. (Inplace operation)
    /// </summary>
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT,
                                const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
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
    ImageView<T> &ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue, ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// aDst pixel is set to aValue if this and aConst fulfill aCompare, this otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue, ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// aDst pixel is set to aValue if this and aConst fulfill aCompare, this otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare, const T &aValue,
                            ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const;

    /// <summary>
    /// aDst pixel is set to aValue if this fulfills aCompare (for floating point checks, e.g. isinf()), this
    /// otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(CompareOp aCompare, const T &aValue, ImageView<T> &aDst,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get()) const
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// A pixel is set to aValue if this and aSrc2 fulfill aCompare, this otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// A pixel is set to aValue if this and aConst fulfill aCompare, this otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// A pixel is set to aValue if this and aConst fulfill aCompare, this otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare, const T &aValue,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// A pixel is set to aValue if this fulfills aCompare (for floating point checks, e.g. isinf()), this
    /// otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    ImageView<T> &ReplaceIf(CompareOp aCompare, const T &aValue,
                            const mpp::cuda::StreamCtx &aStreamCtx = mpp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;
#pragma endregion
#pragma endregion
};
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND