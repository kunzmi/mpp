#pragma once

#include "imageView.h"
#include <backends/cuda/devVarView.h>
#include <backends/npp/nppException.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>

namespace opp::image::npp
{
// forward declaration:
class Image8uC1View;
class Image8uC3View;
class Image8uC4View;

class Image8uC2View : public ImageView<Pixel8uC2>
{
  protected:
    Image8uC2View() = default;
    explicit Image8uC2View(const Size2D &aSize);

  public:
    Image8uC2View(Pixel8uC2 *aBasePointer, const SizePitched &aSizeAlloc);
    Image8uC2View(Pixel8uC2 *aBasePointer, const SizePitched &aSizeAlloc, const Roi &aRoi);
    ~Image8uC2View() = default;

    Image8uC2View(const Image8uC2View &)     = default;
    Image8uC2View(Image8uC2View &&) noexcept = default;

    Image8uC2View &operator=(const Image8uC2View &)     = default;
    Image8uC2View &operator=(Image8uC2View &&) noexcept = default;

    /// <summary>
    /// Returns a new Image8uC2View with the new ROI
    /// </summary>
    [[nodiscard]] Image8uC2View GetView(const Roi &aRoi) const;

    /// <summary>
    /// Returns a new ImageView with the current ROI adapted by aBorder
    /// </summary>
    [[nodiscard]] Image8uC2View GetView(const Border &aBorder = Border()) const;

    //NOLINTBEGIN(readability-identifier-naming,readability-avoid-const-params-in-decls)

    void Set(const Pixel8uC2 &aValue, const NppStreamContext &nppStreamCtx);
    void YUV422ToRGB(Image8uC3View &pDst, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToRGB(Image8uC3View &pDst, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToRGB(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCrCb422ToRGB(Image8uC3View &pDst, const NppStreamContext &nppStreamCtx) const;
    void YCrCb422ToRGB(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToBGR(Image8uC3View &pDst, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToBGR(Image8uC4View &pDst, const Pixel8uC1 &nAval, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToRGB(Image8uC3View &pDst, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToBGR(Image8uC4View &pDst, const Pixel8uC1 &nAval, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToBGR_709HDTV(Image8uC3View &pDst, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToBGR_709HDTV(Image8uC4View &pDst, const Pixel8uC1 &nAval, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToYCrCb422(Image8uC2View &pDst, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToCbYCr422(Image8uC2View &pDst, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToYCbCr411(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToYCbCr420(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToYCbCr420(Image8uC1View &pDstY, Image8uC1View &pDstCbCr, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToYCrCb420(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToYCbCr411(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCbCr422ToYCbCr411(Image8uC1View &pDstY, Image8uC1View &pDstCbCr, const NppStreamContext &nppStreamCtx) const;
    void YCrCb422ToYCbCr422(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCrCb422ToYCbCr420(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void YCrCb422ToYCbCr411(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToYCbCr422(Image8uC2View &pDst, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToYCbCr422(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToYCbCr420(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToYCbCr420(Image8uC1View &pDstY, Image8uC1View &pDstCbCr, const NppStreamContext &nppStreamCtx) const;
    void CbYCr422ToYCrCb420(Image8uC1View &aDstChannel0, Image8uC1View &aDstChannel1, Image8uC1View &aDstChannel2, const NppStreamContext &nppStreamCtx) const;
    void ColorTwist32f(Image8uC2View &pDst, const Npp32f aTwist[3][4], const NppStreamContext &nppStreamCtx) const;
    void ColorTwist32f(const Npp32f aTwist[3][4], const NppStreamContext &nppStreamCtx);
    void YUV422ToRGB(Image8uC3View &pDst, const Npp32f aTwist[3][4], const NppStreamContext &nppStreamCtx) const;
    void Filter32f(Image8uC2View &pDst, const cuda::DevVarView<float> &pKernel, NppiSize oKernelSize, NppiPoint oAnchor, const NppStreamContext &nppStreamCtx) const;
    void FilterBorder32f(Image8uC2View &pDst, const cuda::DevVarView<float> &pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, const NppStreamContext &nppStreamCtx, const Roi &aFilterArea = Roi()) const;
    void MaximumError(const Image8uC2View &pSrc2, cuda::DevVarView<double> &pError, cuda::DevVarView<byte> &pDeviceBuffer, const NppStreamContext &nppStreamCtx) const;
    [[nodiscard]] size_t MaximumErrorGetBufferHostSize(const NppStreamContext &nppStreamCtx) const;
    void AverageError(const Image8uC2View &pSrc2, cuda::DevVarView<double> &pError, cuda::DevVarView<byte> &pDeviceBuffer, const NppStreamContext &nppStreamCtx) const;
    [[nodiscard]] size_t AverageErrorGetBufferHostSize(const NppStreamContext &nppStreamCtx) const;
    void MaximumRelativeError(const Image8uC2View &pSrc2, cuda::DevVarView<double> &pError, cuda::DevVarView<byte> &pDeviceBuffer, const NppStreamContext &nppStreamCtx) const;
    [[nodiscard]] size_t MaximumRelativeErrorGetBufferHostSize(const NppStreamContext &nppStreamCtx) const;
    void AverageRelativeError(const Image8uC2View &pSrc2, cuda::DevVarView<double> &pError, cuda::DevVarView<byte> &pDeviceBuffer, const NppStreamContext &nppStreamCtx) const;
    [[nodiscard]] size_t AverageRelativeErrorGetBufferHostSize(const NppStreamContext &nppStreamCtx) const;

    //NOLINTEND(readability-identifier-naming,readability-avoid-const-params-in-decls)
};
} // namespace opp::image::npp
