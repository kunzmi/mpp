#pragma once
#include "../dllexport_npp.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/stream.h>
#include <backends/npp/nppException.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/bound.h>
#include <common/image/gotoPtr.h>
#include <common/image/imageViewBase.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <common/vector4.h>
#include <common/vector4A.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_filtering_functions.h>
#include <nppi_geometry_transforms.h>
#include <nppi_statistics_functions.h>
#include <utility>
#include <vector>

namespace mpp::image::npp
{

template <PixelType T> class MPPEXPORT_NPP ImageView : public ImageViewBase<T>
{

  protected:
    ImageView() = default;

    explicit ImageView(const Size2D &aSize) : ImageViewBase<T>(aSize)
    {
    }

    [[nodiscard]] ImageViewBase<T>::MemoryType_enum MemoryType() const override
    {
        return ImageViewBase<T>::MemoryType_enum::CudaDefault;
    }

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

    /// <summary>
    /// Returns a new ImageView with the new ROI
    /// </summary>
    ImageView GetView(const Roi &aRoi)
    {
        return ImageView(Pointer(), SizePitched(SizeAlloc(), Pitch()), aRoi);
    }

    /// <summary>
    /// Returns a new ImageView with the current ROI adapted by aBorder
    /// </summary>
    ImageView GetView(const Border &aBorder = Border())
    {
        const Roi newRoi = ROI() + aBorder;
        checkRoiIsInRoi(newRoi, Roi(0, 0, SizeAlloc()));
        return ImageView(Pointer(), SizePitched(SizeAlloc(), Pitch()), newRoi);
    }

    [[nodiscard]] NppiSize NppiSizeRoi() const
    {
        return {ROI().width, ROI().height};
    }

    [[nodiscard]] NppiRect NppiRectRoi() const
    {
        return {ROI().x, ROI().y, ROI().width, ROI().height};
    }

    [[nodiscard]] NppiPoint NppiPointRoi() const
    {
        return {ROI().x, ROI().y};
    }

    [[nodiscard]] NppiSize NppiSizeFull() const
    {
        return {SizeAlloc().x, SizeAlloc().y};
    }

    [[nodiscard]] NppiRect NppiRectFull() const
    {
        return {0, 0, SizeAlloc().x, SizeAlloc().y};
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
            cudaSafeCall(cudaMemcpy2D(aDstView.Pointer(), aDstView.Pitch(), Pointer(), Pitch(), WidthInBytes(),
                                      to_size_t(Height()), cudaMemcpyDeviceToDevice));
        }
        else if (aDstView.MemoryType() == ImageViewBase<T>::MemoryType_enum::HostDefault)
        {
            CopyToHost(aDstView.Pointer(), aDstView.Pitch());
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
            cudaSafeCall(cudaMemcpy2D(Pointer(), Pitch(), aSrcView.Pointer(), aSrcView.Pitch(), WidthInBytes(),
                                      to_size_t(Height()), cudaMemcpyDeviceToDevice));
        }
        else if (aSrcView.MemoryType() == ImageViewBase<T>::MemoryType_enum::HostDefault)
        {
            CopyToDevice(aSrcView.Pointer(), aSrcView.Pitch());
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
            throw INVALIDARGUMENT(aSrcView, "Device to device copy with ROI is not implemented here.");
        }

        if (aDstView.MemoryType() == ImageViewBase<T>::MemoryType_enum::HostDefault)
        {
            CopyToHostRoi(aDstView.Pointer(), aDstView.Pitch(), aDstView.ROI());
        }
        else
        {
            throw INVALIDARGUMENT(
                aDstView,
                "Unknown memory location for destination ImageView. Cannot copy data from CUDA device to there.");
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
            throw INVALIDARGUMENT(aSrcView, "Device to device copy with ROI is not implemented here.");
        }

        if (aSrcView.MemoryType() == ImageViewBase<T>::MemoryType_enum::HostDefault)
        {
            CopyToDeviceRoi(aSrcView.Pointer(), aSrcView.Pitch(), aSrcView.ROI());
        }
        else
        {
            throw INVALIDARGUMENT(
                aSrcView, "Unknown memory location for source ImageView. Cannot copy data from there to CUDA device.");
        }
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
        cudaSafeCall(cudaMemcpy2D(Pointer(), Pitch(), aHostSrc, aHostStride, WidthInBytes(), to_size_t(Height()),
                                  cudaMemcpyHostToDevice));
    }

    /// <summary>
    /// Copy from device to device memory
    /// </summary>
    /// <param name="aDeviceSrc">Source</param>
    void CopyDeviceToDevice(const ImageView &aDeviceSrc)
    {
        if (SizeAlloc() != aDeviceSrc.SizeAlloc())
        {
            throw ROIEXCEPTION("The source image does not have the same size as the destination image. Source size "
                               << aDeviceSrc.SizeAlloc() << ", Destination size: " << SizeAlloc());
        }
        cudaSafeCall(cudaMemcpy2D(Pointer(), Pitch(), aDeviceSrc.Pointer(), aDeviceSrc.Pitch(), WidthInBytes(),
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
        cudaSafeCall(cudaMemcpy2D(Pointer(), Pitch(), aDeviceSrc, aSourcePitch, WidthInBytes(), to_size_t(Height()),
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
        cudaSafeCall(cudaMemcpy2D(aHostDest, aHostStride, Pointer(), Pitch(), WidthInBytes(), to_size_t(Height()),
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

        cudaSafeCall(cudaMemcpy2D(PointerRoi(), Pitch(), src, aSourcePitch, WidthRoiInBytes(), to_size_t(HeightRoi()),
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

        cudaSafeCall(cudaMemcpy2D(dest, aDestPitch, PointerRoi(), Pitch(), WidthRoiInBytes(), to_size_t(HeightRoi()),
                                  cudaMemcpyDeviceToHost));
    }

    [[nodiscard]] static NppStreamContext GetStreamContext()
    {
        NppStreamContext nppCtx{};

        cudaSafeCall(cudaGetDevice(&nppCtx.nCudaDeviceId));

        cudaDeviceProp props{};
        cudaSafeCall(cudaGetDeviceProperties(&props, nppCtx.nCudaDeviceId));
        nppCtx.nMultiProcessorCount               = props.multiProcessorCount;
        nppCtx.nMaxThreadsPerMultiProcessor       = props.maxThreadsPerMultiProcessor;
        nppCtx.nMaxThreadsPerBlock                = props.maxThreadsPerBlock;
        nppCtx.nSharedMemPerBlock                 = props.sharedMemPerBlock;
        nppCtx.nCudaDevAttrComputeCapabilityMajor = props.major;
        nppCtx.nCudaDevAttrComputeCapabilityMinor = props.minor;

        return nppCtx;
    }

    [[nodiscard]] static NppStreamContext GetStreamContext(const mpp::cuda::Stream &aStream)
    {
        NppStreamContext nppCtx = GetStreamContext();
        nppCtx.hStream          = aStream.Original();

        cudaSafeCall(cudaStreamGetFlags(nppCtx.hStream, &nppCtx.nStreamFlags));
        return nppCtx;
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
        nppSafeCall(nppiGetFilterGaussPyramidLayerDownBorderDstROI(ROI().width, ROI().height, &retValue, aRate));
        return retValue;
    }

    [[nodiscard]] std::pair<NppiSize, NppiSize> GetFilterGaussPyramidLayerUpBorderDstROI(float aRate) const
    {
        NppiSize roiMin{};
        NppiSize roiMax{};
        nppSafeCall(nppiGetFilterGaussPyramidLayerUpBorderDstROI(ROI().width, ROI().height, &roiMin, &roiMax, aRate));
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
} // namespace mpp::image::npp