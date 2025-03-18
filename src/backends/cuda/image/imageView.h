#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/arithmetic/addSquareProductWeightedOutputType.h>
#include <backends/cuda/image/arithmetic/arithmeticKernel.h>
#include <backends/cuda/image/dataExchangeAndInit/dataExchangeAndInitKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>

namespace opp::image::cuda
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
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>) &&
                (RealOrComplexIntVector<T> || (std::same_as<complex_basetype_t<remove_vector_t<T>>, float> &&
                                               (std::same_as<complex_basetype_t<remove_vector_t<TTo>>, BFloat16> ||
                                                std::same_as<complex_basetype_t<remove_vector_t<TTo>>, HalfFp16>)));

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>) && RealOrComplexFloatingVector<T>;

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>) && (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) &&
                (!std::same_as<TTo, Complex<float>>) && (!std::same_as<TTo, Complex<double>>);
#pragma endregion
#pragma region Copy
    /// <summary>
    /// Copy image.
    /// </summary>
    ImageView<T> &Copy(ImageView<T> &aDst,
                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    ImageView<T> &Copy(ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst, Channel aDstChannel,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> > 1) &&   //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy this single channel image to channel aDstChannel of aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(ImageView<TTo> &aDst, Channel aDstChannel,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> == 1) &&  //
                (vector_size_v<TTo> > 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Copy(Channel aSrcChannel, ImageView<TTo> &aDst,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> > 1) &&    //
                (vector_size_v<TTo> == 1) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    void Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
              ImageView<Vector1<remove_vector_t<T>>> &aDstChannel4,
              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(FourChannelNoAlpha<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(TwoChannel<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(ThreeChannel<T>);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    static ImageView<T> &Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3,
                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel4, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(FourChannelNoAlpha<T>);
#pragma endregion
#pragma region Dup
    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Dup(ImageView<TTo> &aDst,
                        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
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
    ImageView<TTo> &Scale(ImageView<TTo> &aDst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T> && RealOrComplexIntVector<TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<TTo>;

    /// <summary>
    /// Convert witch scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                          scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!std::same_as<T, TTo>);

#pragma endregion
#pragma region Set
    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    ImageView<T> &Set(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    ImageView<T> &Set(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    ImageView<T> &Set(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());
#pragma endregion
#pragma region Swap Channel
    /// <summary>
    /// Swap channels
    /// </summary>
    template <PixelType TTo>
    ImageView<TTo> &SwapChannel(ImageView<TTo> &aDst, const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
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
                                remove_vector_t<T> aValue,
                                const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(vector_size_v<T> == 3) &&          //
                (vector_active_size_v<TTo> == 4) && //
                std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>;
#pragma endregion
#pragma endregion

#pragma region Arithmetic functions
#pragma region Add
    /// <summary>
    /// aDst = this + aSrc2
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst
    /// </summary>
    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aSrc2
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst
    /// </summary>
    ImageView<T> &Add(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const T &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aSrc2 for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this + aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Add(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Add(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this += aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Sub
    /// <summary>
    /// aDst = this - aSrc2
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst
    /// </summary>
    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aSrc2
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst
    /// </summary>
    ImageView<T> &Sub(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const T &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 - this
    /// </summary>
    ImageView<T> &SubInv(const ImageView<T> &aSrc2,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 - this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &SubInv(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this
    /// </summary>
    ImageView<T> &SubInv(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &SubInv(const T &aConst, int aScaleFactor = 0,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this
    /// </summary>
    ImageView<T> &SubInv(const opp::cuda::DevVarView<T> &aConst,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &SubInv(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this - aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this -= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Sub(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 - this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 - this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst - this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &SubInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst - this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &SubInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                         int aScaleFactor                       = 0,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region Mul
    /// <summary>
    /// aDst = this * aSrc2
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst
    /// </summary>
    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aSrc2
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst
    /// </summary>
    ImageView<T> &Mul(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const T &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aSrc2 for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this * aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this *= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Mul(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region MulScale
    /// <summary>
    /// aDst = this * aSrc2, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const T &aConst, ImageView<T> &aDst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aSrc2, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const ImageView<T> &aSrc2,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const T &aConst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width
    /// </summary>
    ImageView<T> &MulScale(const opp::cuda::DevVarView<T> &aConst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aSrc2, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScale(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScale(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                           const ImageView<Pixel8uC1> &aMask,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aSrc2, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScale(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScale(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;

    /// <summary>
    /// aDst = this * aConst, then scales the result by the maximum value for the data bit width, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &MulScale(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>;
#pragma endregion
#pragma region Div
    /// <summary>
    /// aDst = this / aSrc2
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst
    /// </summary>
    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aSrc2
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst
    /// </summary>
    ImageView<T> &Div(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const T &aConst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 / this
    /// </summary>
    ImageView<T> &DivInv(const ImageView<T> &aSrc2,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 / this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &DivInv(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this
    /// </summary>
    ImageView<T> &DivInv(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &DivInv(const T &aConst, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this
    /// </summary>
    ImageView<T> &DivInv(const opp::cuda::DevVarView<T> &aConst,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    ImageView<T> &DivInv(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// aDst = this / aConst for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst = this / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aSrc2, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Div(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Div(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this /= aConst, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    ImageView<T> &Div(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aSrc2 / this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aSrc2 / this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                         RoundingMode aRoundingMode             = RoundingMode::NearestTiesToEven,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;

    /// <summary>
    /// this = aConst / this, for all pixels where aMask != 0
    /// </summary>
    ImageView<T> &DivInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// this = aConst / this, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    ImageView<T> &DivInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                         int aScaleFactor = 0, RoundingMode aRoundingMode = RoundingMode::NearestTiesToEven,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>;
#pragma endregion
#pragma region AddSquare
    /// <summary>
    /// SrcDst += this^2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddSquare(
        ImageView<add_spw_output_for_t<T>> &aSrcDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// SrcDst += this^2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddSquare(
        ImageView<add_spw_output_for_t<T>> &aSrcDst, const ImageView<Pixel8uC1> &aMask,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());
#pragma endregion
#pragma region AddProduct
    /// <summary>
    /// SrcDst += this * Src2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddProduct(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aSrcDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// SrcDst += this * Src2
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddProduct(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aSrcDst, const ImageView<Pixel8uC1> &aMask,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());
#pragma endregion
#pragma region AddWeighted
    /// <summary>
    /// Dst = this * alpha + Src2 * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aDst,
        remove_vector_t<add_spw_output_for_t<T>> aAlpha,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// Dst = this * alpha + Src2 * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(
        const ImageView<T> &aSrc2, ImageView<add_spw_output_for_t<T>> &aDst,
        remove_vector_t<add_spw_output_for_t<T>> aAlpha, const ImageView<Pixel8uC1> &aMask,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());
    /// <summary>
    /// SrcDst = this * alpha + SrcDst * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(
        ImageView<add_spw_output_for_t<T>> &aSrcDst, remove_vector_t<add_spw_output_for_t<T>> aAlpha,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// SrcDst = this * alpha + SrcDst * (1 - alpha)
    /// </summary>
    ImageView<add_spw_output_for_t<T>> &AddWeighted(
        ImageView<add_spw_output_for_t<T>> &aSrcDst, remove_vector_t<add_spw_output_for_t<T>> aAlpha,
        const ImageView<Pixel8uC1> &aMask,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());
#pragma endregion

#pragma region Abs
    /// <summary>
    /// aDst = abs(this)
    /// </summary>
    ImageView<T> &Abs(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealSignedVector<T>;

    /// <summary>
    /// this = abs(this)
    /// </summary>
    ImageView<T> &Abs(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealSignedVector<T>;
#pragma endregion
#pragma region AbsDiff
    /// <summary>
    /// aDst = abs(this - aSrc2)
    /// </summary>
    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// aDst = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const T &aConst, ImageView<T> &aDst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// aDst = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// this = abs(this - aSrc2)
    /// </summary>
    ImageView<T> &AbsDiff(const ImageView<T> &aSrc2,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// this = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const T &aConst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;

    /// <summary>
    /// this = abs(this - aConst)
    /// </summary>
    ImageView<T> &AbsDiff(const opp::cuda::DevVarView<T> &aConst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealUnsignedVector<T>;
#pragma endregion
#pragma region And
    /// <summary>
    /// aDst = this & aSrc2 (bitwise AND)
    /// </summary>
    ImageView<T> &And(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this & aSrc2 (bitwise AND)
    /// </summary>
    ImageView<T> &And(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this & aConst (bitwise AND)
    /// </summary>
    ImageView<T> &And(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Not
    /// <summary>
    /// aDst = ~this (bitwise NOT)
    /// </summary>
    ImageView<T> &Not(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = ~this (bitwise NOT)
    /// </summary>
    ImageView<T> &Not(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Exp
    /// <summary>
    /// aDst = exp(this) (exponential function)
    /// </summary>
    ImageView<T> &Exp(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = exp(this) (exponential function)
    /// </summary>
    ImageView<T> &Exp(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Ln
    /// <summary>
    /// aDst = log(this) (natural logarithm)
    /// </summary>
    ImageView<T> &Ln(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = log(this) (natural logarithm)
    /// </summary>
    ImageView<T> &Ln(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region LShift
    /// <summary>
    /// aDst = this << aConst (left bitshift)
    /// </summary>
    ImageView<T> &LShift(uint aConst, ImageView<T> &aDst,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
    /// <summary>
    /// this = this << aConst (left bitshift)
    /// </summary>
    ImageView<T> &LShift(uint aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Or
    /// <summary>
    /// aDst = this | aSrc2 (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                     const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const T &aConst, ImageView<T> &aDst,
                     const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                     const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this | aSrc2 (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const ImageView<T> &aSrc2,
                     const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this | aConst (bitwise Or)
    /// </summary>
    ImageView<T> &Or(const opp::cuda::DevVarView<T> &aConst,
                     const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region RShift
    /// <summary>
    /// aDst = this >> aConst (right bitshift)
    /// </summary>
    ImageView<T> &RShift(uint aConst, ImageView<T> &aDst,
                         const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
    /// <summary>
    /// this = this >> aConst (right bitshift)
    /// </summary>
    ImageView<T> &RShift(uint aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;
#pragma endregion
#pragma region Sqr
    /// <summary>
    /// aDst = this * this (this^2)
    /// </summary>
    ImageView<T> &Sqr(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = this * this (this^2)
    /// </summary>
    ImageView<T> &Sqr(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Sqrt
    /// <summary>
    /// aDst = Sqrt(this) (square root function)
    /// </summary>
    ImageView<T> &Sqrt(ImageView<T> &aDst,
                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;

    /// <summary>
    /// this = Sqrt(this) (square root function)
    /// </summary>
    ImageView<T> &Sqrt(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexVector<T>;
#pragma endregion
#pragma region Xor
    /// <summary>
    /// aDst = this ^ aSrc2 (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// aDst = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this ^ aSrc2 (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T>;

    /// <summary>
    /// this = this ^ aConst (bitwise Xor)
    /// </summary>
    ImageView<T> &Xor(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
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
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires FourChannelNoAlpha<T> && RealVector<T>;

    /// <summary>
    /// Premultiplies pixels of an image with alpha from fourth color channel.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires FourChannelNoAlpha<T> && RealVector<T>;

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value. aAlpha is expected in value range 0..1
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealFloatingVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value. aAlpha is expected in value range 0..1
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealFloatingVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealIntVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value.
    /// Note: AlphaPremul does not exactly match the results from NPP for integer image types. NPP seems to scale the
    /// integer value by T::max() and then does the multiplications/divisions as integers. Here we cast to float and
    /// then round using RoundingMode::NearestTiesToEven which is nearly identical, but not exactly the
    /// same for all values. Values may differ by 1.
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires FourChannelAlpha<T>;

    /// <summary>
    /// Premultiplies pixels of an image with constant aAlpha value. aAlpha is expected in value range 0..1
    /// </summary>
    ImageView<T> &AlphaPremul(remove_vector_t<T> aAlpha,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires FourChannelAlpha<T>;
#pragma endregion
#pragma region AlphaComp
    /// <summary>
    /// Composite two images using alpha opacity values contained in each image. Last color channel is alpha channel, 1
    /// channel images are treated as alpha channel only.
    /// </summary>
    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires(!FourChannelAlpha<T>) && RealVector<T>;

    /// <summary>
    /// Composite two images using constant alpha values.
    /// </summary>
    ImageView<T> &AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                            remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
#pragma endregion

#pragma region Complex
    /// <summary>
    /// aDst = this * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// this = this * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    ImageView<T> &ConjMul(const ImageView<T> &aSrc2,
                          const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = conj(this) (complex conjugate)
    /// </summary>
    ImageView<T> &Conj(ImageView<T> &aDst,
                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// this = conj(this) (complex conjugate)
    /// </summary>
    ImageView<T> &Conj(const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = abs(this) (complex magnitude)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Magnitude(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = abs(this)^2 (complex magnitude squared)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &MagnitudeSqr(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = angle(this) (complex angle, atan2(imag, real))
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Angle(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>;

    /// <summary>
    /// aDst = this.real (real component of complex value)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Real(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// aDst = this.imag (imaginary component of complex value)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &Imag(
        ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires ComplexVector<T>;

    /// <summary>
    /// aDst.real = this, aDst.imag = 0 (converts real valued image to complex with imaginary part = 0)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealSignedVector<T> && (!FourChannelAlpha<T>);

    /// <summary>
    /// aDst.real = this, aDst.imag = aSrcImag (converts two real valued images to one complex image)
    /// </summary>
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &MakeComplex(
        const ImageView<T> &aSrcImag,
        ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
        const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealSignedVector<T> && (!FourChannelAlpha<T>);
#pragma endregion
#pragma endregion

#pragma region Statistics
#pragma region MinEvery
    /// <summary>
    /// aDst = min(this, aSrc2) (minimum per pixel, per channel)
    /// </summary>
    ImageView<T> &MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;

    /// <summary>
    /// this = min(this, aSrc2) (minimum per pixel, per channel)
    /// </summary>
    ImageView<T> &MinEvery(const ImageView<T> &aSrc2,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
#pragma endregion
#pragma region MaxEvery
    /// <summary>
    /// aDst = max(this, aSrc2) (maximum per pixel, per channel)
    /// </summary>
    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;

    /// <summary>
    /// this = max(this, aSrc2) (maximum per pixel, per channel)
    /// </summary>
    ImageView<T> &MaxEvery(const ImageView<T> &aSrc2,
                           const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
#pragma endregion
#pragma endregion

#pragma region Threshold and Compare
#pragma region Compare
    /// <summary>
    /// aDst pixel is set to 255 if all color channels in this and aSrc2 fulfill aCompare, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                  const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// aDst pixel is set to 255 if all color channels in this and aConst fulfill aCompare, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                  const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// aDst pixel is set to 255 if all color channels in this and aConst fulfill aCompare, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &Compare(const opp::cuda::DevVarView<T> &aConst, CompareOp aCompare,
                                  ImageView<Pixel8uC1> &aDst,
                                  const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get());

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(this - aSrc2) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &CompareEqEps(const ImageView<T> &aSrc2, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                       ImageView<Pixel8uC1> &aDst,
                                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(this - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                       ImageView<Pixel8uC1> &aDst,
                                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(this - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    ImageView<Pixel8uC1> &CompareEqEps(const opp::cuda::DevVarView<T> &aConst,
                                       complex_basetype_t<remove_vector_t<T>> aEpsilon, ImageView<Pixel8uC1> &aDst,
                                       const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>;
#pragma endregion
#pragma region Threshold
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &Threshold(const opp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare, ImageView<T> &aDst,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const opp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const opp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &Threshold(const T &aThreshold, CompareOp aCompare,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &Threshold(const opp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const opp::cuda::DevVarView<T> &aThreshold,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const opp::cuda::DevVarView<T> &aThreshold,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;

    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare,
                            const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLT(const T &aThreshold, const T &aValue,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdGT(const T &aThreshold, const T &aValue,
                              const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT,
                                ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
    ImageView<T> &ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT, const T &aValueGT,
                                const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealVector<T>;
#pragma endregion
#pragma endregion
};
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND