#pragma once
#include <common/moduleEnabler.h>
#if OPP_ENABLE_NPP_BACKEND
#include <common/image/pixelTypeEnabler.h>

#include <backends/cuda/cudaException.h>
#include <backends/cuda/stream.h>
#include <backends/npp/nppException.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/bound.h>
#include <common/image/gotoPtr.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <common/vector_typetraits.h>
#include <cstddef>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>
#include <nppi_statistics_functions.h>
#include <utility>
#include <vector>

namespace opp::image::npp
{

template <PixelType T> class ImageView
{
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
    /// Copy from Host to device memory
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

    [[nodiscard]] NppiSize NppiSizeRoi() const
    {
        return {mRoi.width, mRoi.height};
    }

    [[nodiscard]] NppiRect NppiRectRoi() const
    {
        return {mRoi.x, mRoi.y, mRoi.width, mRoi.height};
    }

    [[nodiscard]] NppiPoint NppiPointRoi() const
    {
        return {mRoi.x, mRoi.y};
    }

    [[nodiscard]] NppiSize NppiSizeFull() const
    {
        return {mSizeAlloc.x, mSizeAlloc.y};
    }

    [[nodiscard]] NppiRect NppiRectFull() const
    {
        return {0, 0, mSizeAlloc.x, mSizeAlloc.y};
    }

    /// <summary>
    /// Copy from device to device memory
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
    /// Copy from device to device memory
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
    /// Copy data from device to host memory
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
    /// Copy from device to host memory
    /// </summary>
    void operator>>(void *aDest) const
    {
        CopyToHost(aDest);
    }

    /// <summary>
    /// Copy from host to device memory
    /// </summary>
    void operator<<(const void *aSource)
    {
        CopyToDevice(aSource);
    }

    /// <summary>
    /// Copy from device to host memory
    /// </summary>
    void CopyToHost(std::vector<T> &aHostDest) const
    {
        CopyToHost(aHostDest.data());
    }

    /// <summary>
    /// Copy from host to device memory
    /// </summary>
    void CopyToDevice(const std::vector<T> &aHostSrc)
    {
        CopyToDevice(aHostSrc.data());
    }

    /// <summary>
    /// Copy from device to host memory
    /// </summary>
    void operator>>(std::vector<T> &aHostDest) const
    {
        CopyToHost(aHostDest.data());
    }

    /// <summary>
    /// Copy from host to device memory
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

    [[nodiscard]] static NppStreamContext GetStreamContext()
    {
        NppStreamContext nppCtx{};
        nppSafeCall(nppGetStreamContext(&nppCtx));
        return nppCtx;
    }

    static void SetStream(const opp::cuda::Stream &aStream)
    {
        nppSafeCall(nppSetStream(aStream.Original()));
    }

    [[nodiscard]] static opp::cuda::Stream GetStream()
    {
        return opp::cuda::Stream(nppGetStream());
    }

    [[nodiscard]] size_t FilterBoxBorderAdvancedGetDeviceBufferSize() const
        requires(sizeof(pixel_basetype_t<T>) < 8)
    {
        int retValue = 0;
        nppSafeCall(nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSizeRoi(), to_int(ChannelCount), &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] size_t FilterBoxBorderAdvancedGetDeviceBufferSize() const
        requires(sizeof(pixel_basetype_t<T>) >= 8)
    {
        int retValue = 0;
        nppSafeCall(nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSizeRoi(), to_int(ChannelCount), &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] NppiSize GetFilterGaussPyramidLayerDownBorderDstROI(float aRate) const
    {
        NppiSize retValue{};
        nppSafeCall(nppiGetFilterGaussPyramidLayerDownBorderDstROI(mRoi.width, mRoi.height, &retValue, aRate));
        return retValue;
    }

    [[nodiscard]] std::pair<NppiSize, NppiSize> GetFilterGaussPyramidLayerUpBorderDstROI(float aRate) const
    {
        NppiSize roiMin{};
        NppiSize roiMax{};
        nppSafeCall(nppiGetFilterGaussPyramidLayerUpBorderDstROI(mRoi.width, mRoi.height, &roiMin, &roiMax, aRate));
        return {roiMin, roiMax};
    }

    [[nodiscard]] size_t DistanceTransformPBAGetBufferSize() const
    {
        size_t retValue = 0;
        nppSafeCall(nppiDistanceTransformPBAGetBufferSize(NppiSizeRoi(), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t DistanceTransformPBAGetAntialiasingBufferSize() const
    {
        size_t retValue = 0;
        nppSafeCall(nppiDistanceTransformPBAGetAntialiasingBufferSize(NppiSizeRoi(), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t SignedDistanceTransformPBAGetBufferSize() const
    {
        size_t retValue = 0;
        nppSafeCall(nppiSignedDistanceTransformPBAGetBufferSize(NppiSizeRoi(), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t SignedDistanceTransformPBAGet64fBufferSize() const
    {
        size_t retValue = 0;
        nppSafeCall(nppiSignedDistanceTransformPBAGet64fBufferSize(NppiSizeRoi(), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t SignedDistanceTransformPBAGetAntialiasingBufferSize() const
    {
        size_t retValue = 0;
        nppSafeCall(nppiSignedDistanceTransformPBAGetAntialiasingBufferSize(NppiSizeRoi(), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t FilterCannyBorderGetBufferSize() const
    {
        int retValue = 0;
        nppSafeCall(nppiFilterCannyBorderGetBufferSize(NppiSizeRoi(), &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] size_t FilterHarrisCornersBorderGetBufferSize() const
    {
        int retValue = 0;
        nppSafeCall(nppiFilterHarrisCornersBorderGetBufferSize(NppiSizeRoi(), &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] size_t FilterHoughLineGetBufferSize(NppPointPolar aDelta, int aMaxLineCount) const
    {
        int retValue = 0;
        nppSafeCall(nppiFilterHoughLineGetBufferSize(NppiSizeRoi(), aDelta, aMaxLineCount, &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] size_t HistogramOfGradientsBorderGetBufferSize(const NppiHOGConfig aHOGConfig,
                                                                 const NppiPoint *ahpLocations, int aLocations) const
    {
        int retValue = 0;
        nppSafeCall(nppiHistogramOfGradientsBorderGetBufferSize(aHOGConfig, ahpLocations, aLocations, NppiSizeRoi(),
                                                                &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] static size_t HistogramOfGradientsBorderGetDescriptorsSize(const NppiHOGConfig aHOGConfig,
                                                                             int aLocations)
    {
        int retValue = 0;
        nppSafeCall(nppiHistogramOfGradientsBorderGetDescriptorsSize(aHOGConfig, aLocations, &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] size_t FloodFillGetBufferSize() const
    {
        int retValue = 0;
        nppSafeCall(nppiFloodFillGetBufferSize(NppiSizeRoi(), &retValue));
        return to_size_t(retValue);
    }

    [[nodiscard]] static std::vector<int> EvenLevels(int aLevels, int aLowerLevel, int aUpperLevel)
    {
        std::vector<int> retValue(to_size_t(aLevels));
        nppSafeCall(nppiEvenLevelsHost_32s(retValue.data(), aLevels, aLowerLevel, aUpperLevel));
        return retValue;
    }

    [[nodiscard]] size_t CrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize aTplRoiSize) const
    {
        constexpr int dstSize = TypeSize == 8 ? sizeof(double) : sizeof(float);
        size_t retValue       = 0;
        nppSafeCall(nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSizeRoi(), aTplRoiSize, dstSize,
                                                                             to_int(ChannelCount), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t CrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize aTplRoiSize) const
    {
        constexpr int dstSize = TypeSize == 8 ? sizeof(double) : sizeof(float);
        size_t retValue       = 0;
        nppSafeCall(nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSizeRoi(), aTplRoiSize, dstSize,
                                                                             to_int(ChannelCount), &retValue));
        return retValue;
    }

    [[nodiscard]] size_t CrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize aTplRoiSize) const
    {
        constexpr int dstSize = TypeSize == 8 ? sizeof(double) : sizeof(float);
        size_t retValue       = 0;
        nppSafeCall(nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSizeRoi(), aTplRoiSize, dstSize,
                                                                              to_int(ChannelCount), &retValue));
        return retValue;
    }

    [[nodiscard]] NppiRect GetResizeRect(double aXFactor, double aYFactor, double aXShift, double aYShift,
                                         int aInterpolation) const
    {
        NppiRect retValue{};
        nppSafeCall(nppiGetResizeRect(NppiRectRoi(), &retValue, aXFactor, aYFactor, aXShift, aYShift, aInterpolation));
        return retValue;
    }

    [[nodiscard]] NppiPoint GetResizeTiledSourceOffset(NppiRect aDstRectROI) const
    {
        NppiPoint retValue{};
        nppSafeCall(nppiGetResizeTiledSourceOffset(NppiRectRoi(), aDstRectROI, &retValue));
        return retValue;
    }

    void GetRotateQuad(Quad<double> &aQuad, double aAngle, double aShiftX, double aShiftY) const
    {
        nppSafeCall(nppiGetRotateQuad(NppiRectRoi(), reinterpret_cast<double(*)[2]>(&aQuad), aAngle, aShiftX, aShiftY));
    }

    void GetRotateBound(Bound<double> &aBoundingBox, double aAngle, double aShiftX, double aShiftY) const
    {
        nppSafeCall(
            nppiGetRotateBound(NppiRectRoi(), reinterpret_cast<double(*)[2]>(&aBoundingBox), aAngle, aShiftX, aShiftY));
    }

    void GetAffineTransform(const Quad<double> &aQuad, AffineTransformation<double> &aCoeffs) const
    {
        nppSafeCall(nppiGetAffineTransform(NppiRectRoi(), reinterpret_cast<const double(*)[2]>(&aQuad),
                                           reinterpret_cast<double(*)[3]>(&aCoeffs)));
    }

    void GetAffineQuad(Quad<double> &aQuad, const AffineTransformation<double> &aCoeffs) const
    {
        nppSafeCall(nppiGetAffineQuad(NppiRectRoi(), reinterpret_cast<double(*)[2]>(&aQuad),
                                      reinterpret_cast<const double(*)[3]>(&aCoeffs)));
    }

    void GetAffineBound(Bound<double> &aBound, const AffineTransformation<double> &aCoeffs) const
    {
        nppSafeCall(nppiGetAffineBound(NppiRectRoi(), reinterpret_cast<double(*)[2]>(&aBound),
                                       reinterpret_cast<const double(*)[3]>(&aCoeffs)));
    }

    void GetPerspectiveTransform(const Quad<double> &aQuad, PerspectiveTransformation<double> &aCoeffs) const
    {
        nppSafeCall(nppiGetPerspectiveTransform(NppiRectRoi(), reinterpret_cast<const double(*)[2]>(&aQuad),
                                                reinterpret_cast<double(*)[3]>(&aCoeffs)));
    }

    void GetPerspectiveQuad(Quad<double> &aQuad, const PerspectiveTransformation<double> &aCoeffs) const
    {
        nppSafeCall(nppiGetPerspectiveQuad(NppiRectRoi(), reinterpret_cast<double(*)[2]>(&aQuad),
                                           reinterpret_cast<const double(*)[3]>(&aCoeffs)));
    }

    void GetPerspectiveBound(Bound<double> &aBound, const PerspectiveTransformation<double> &aCoeffs) const
    {
        nppSafeCall(nppiGetPerspectiveBound(NppiRectRoi(), reinterpret_cast<double(*)[2]>(&aBound),
                                            reinterpret_cast<const double(*)[3]>(&aCoeffs)));
    }
};
} // namespace opp::image::npp
#endif // OPP_ENABLE_NPP_BACKEND