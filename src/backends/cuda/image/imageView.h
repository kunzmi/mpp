#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/arithmetic/arithmeticKernel.h>
#include <backends/cuda/image/arithmetic/dataInitKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/safeCast.h>
#include <cstddef>
#include <cuda_runtime_api.h>
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
#pragma endregion

#pragma region Basics and Copy to device/host
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
#pragma endregion

#pragma region Data initialisation
    ImageView<T> &Set(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
    {
        InvokeSetC(aConst, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Set(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
    {
        InvokeSetDevC(aConst.Pointer(), PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Set(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
    {
        checkSameSize(ROI(), aMask.ROI());
        InvokeSetCMask(aMask.PointerRoi(), aMask.Pitch(), aConst, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Set(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
    {
        checkSameSize(ROI(), aMask.ROI());
        InvokeSetDevCMask(aMask.PointerRoi(), aMask.Pitch(), aConst.Pointer(), PointerRoi(), Pitch(), SizeRoi(),
                          aStreamCtx);

        return *this;
    }
#pragma endregion

#pragma region Arithmetic functions

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aDst.ROI());

        InvokeAddSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                        SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aDst.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                             scaleFactorFloat, SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());

        InvokeAddSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                           aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());

        InvokeAddSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                         aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(),
                              scaleFactorFloat, SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());

        InvokeAddInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                                 aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        InvokeAddInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const T &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        InvokeAddInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aDst.ROI());
        checkSameSize(ROI(), aMask.ROI());

        InvokeAddSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aDst.ROI());
        checkSameSize(ROI(), aMask.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                                 aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());
        checkSameSize(ROI(), aMask.ROI());

        InvokeAddSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                          aDst.Pitch(), SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());
        checkSameSize(ROI(), aMask.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                               aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());
        checkSameSize(ROI(), aMask.ROI());

        InvokeAddSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                             aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                      int aScaleFactor                       = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aDst.ROI());
        checkSameSize(ROI(), aMask.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

        return aDst;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aMask.ROI());

        InvokeAddInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aSrc2.ROI());
        checkSameSize(ROI(), aMask.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                     aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aMask.ROI());

        InvokeAddInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aMask.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                                   SizeRoi(), aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexFloatingVector<T>
    {
        checkSameSize(ROI(), aMask.ROI());

        InvokeAddInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                                 aStreamCtx);

        return *this;
    }

    ImageView<T> &Add(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor = 0,
                      const opp::cuda::StreamCtx &aStreamCtx = opp::cuda::StreamCtxSingleton::Get())
        requires RealOrComplexIntVector<T>
    {
        checkSameSize(ROI(), aMask.ROI());

        const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

        InvokeAddInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                      scaleFactorFloat, SizeRoi(), aStreamCtx);

        return *this;
    }
#pragma endregion
};
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND