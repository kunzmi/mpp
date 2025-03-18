#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "arithmetic/addSquareProductWeightedOutputType.h"
#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/arithmetic/arithmeticKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image::cuda
{
#pragma region Add
template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeAddSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAddSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAddSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeAddInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeAddInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeAddInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Sub
template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeSubSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeSubSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeSubSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeSubInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeSubInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeSubInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeSubInvInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                                aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeSubInvInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeSubInvInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubInvInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                    aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubInvInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                                  SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeSubInvInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                                aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                   int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                     scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region Mul
template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeMulSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeMulSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeMulSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeMulInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeMulInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeMulInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeMulSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeMulSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeMulSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeMulInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeMulInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeMulInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region MulScale

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                     const ImageView<Pixel8uC1> &aMask, const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Div
template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeDivSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeDivSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor, RoundingMode aRoundingMode,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, aRoundingMode,
                       SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeDivSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeDivInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, aRoundingMode,
                             SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeDivInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeDivInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, aRoundingMode, SizeRoi(),
                              aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeDivInvInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat,
                                aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeDivInvInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    InvokeDivInvInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const opp::cuda::DevVarView<T> &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, aRoundingMode, SizeRoi(),
                                 aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(),
                             aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const ImageView<Pixel8uC1> &aMask, int aScaleFactor, RoundingMode aRoundingMode,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivInvInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                   RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                    aSrc2.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivInvInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                   RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                                  aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                   const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    InvokeDivInvInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                                aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                   int aScaleFactor, RoundingMode aRoundingMode, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aMask.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                     scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region AddSquare
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddSquare(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                            const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrcDst.ROI());

    InvokeAddSquareInplaceSrc(aSrcDst.PointerRoi(), aSrcDst.Pitch(), PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddSquare(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                            const ImageView<Pixel8uC1> &aMask,
                                                            const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrcDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddSquareInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), aSrcDst.PointerRoi(), aSrcDst.Pitch(),
                                  PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}
#pragma endregion

#pragma region AddProduct
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddProduct(const ImageView<T> &aSrc2,
                                                             ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                             const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aSrcDst.ROI());

    InvokeAddProductInplaceSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aSrcDst.PointerRoi(),
                                  aSrcDst.Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddProduct(const ImageView<T> &aSrc2,
                                                             ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                             const ImageView<Pixel8uC1> &aMask,
                                                             const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aSrcDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddProductInplaceSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                      aSrc2.Pitch(), aSrcDst.PointerRoi(), aSrcDst.Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}
#pragma endregion

#pragma region AddWeighted
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(const ImageView<T> &aSrc2,
                                                              ImageView<add_spw_output_for_t<T>> &aDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                              const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeAddWeightedSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                            aAlpha, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(const ImageView<T> &aSrc2,
                                                              ImageView<add_spw_output_for_t<T>> &aDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                              const ImageView<Pixel8uC1> &aMask,
                                                              const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddWeightedSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                              const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrcDst.ROI());

    InvokeAddWeightedInplaceSrc(aSrcDst.PointerRoi(), aSrcDst.Pitch(), PointerRoi(), Pitch(), aAlpha, SizeRoi(),
                                aStreamCtx);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                              const ImageView<Pixel8uC1> &aMask,
                                                              const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrcDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeAddWeightedInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), aSrcDst.PointerRoi(), aSrcDst.Pitch(),
                                    PointerRoi(), Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return aSrcDst;
}
#pragma endregion

#pragma region Abs
template <PixelType T>
ImageView<T> &ImageView<T>::Abs(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealSignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAbsSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Abs(const opp::cuda::StreamCtx &aStreamCtx)
    requires RealSignedVector<T>
{
    InvokeAbsInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region AbsDiff
template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                    const opp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeAbsDiffSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                        SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAbsDiffSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                    const opp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAbsDiffSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeAbsDiffInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    InvokeAbsDiffInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    InvokeAbsDiffInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region And
template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeAndSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAndSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAndSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeAndInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeAndInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeAndInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Not
template <PixelType T>
ImageView<T> &ImageView<T>::Not(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeNotSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Not(const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeNotInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Exp
template <PixelType T>
ImageView<T> &ImageView<T>::Exp(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeExpSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Exp(const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    InvokeExpInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Ln
template <PixelType T>
ImageView<T> &ImageView<T>::Ln(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLnSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Ln(const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    InvokeLnInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region LShift

template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLShiftSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeLShiftInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Or
template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeOrSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                   aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeOrSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                               const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeOrSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeOrInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeOrInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeOrInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region RShift

template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeRShiftSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeRShiftInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Sqr
template <PixelType T>
ImageView<T> &ImageView<T>::Sqr(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeSqrSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqr(const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    InvokeSqrInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Sqrt
template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeSqrtSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt(const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    InvokeSqrtInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Xor
template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeXorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeXorSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const opp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeXorSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeXorInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeXorInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    InvokeXorInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region AlphaPremul
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T> && RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAlphaPremulSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(const opp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T> && RealVector<T>
{
    InvokeAlphaPremulInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealFloatingVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());

    const T alphaVec(aAlpha);

    InvokeMulSrcC(PointerRoi(), Pitch(), alphaVec, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    const T alphaVec(aAlpha);

    InvokeMulSrcCScale(PointerRoi(), Pitch(), alphaVec, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealFloatingVector<T> && (!FourChannelAlpha<T>)
{
    const T alphaVec(aAlpha);

    InvokeMulInplaceC(PointerRoi(), Pitch(), alphaVec, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T> && (!FourChannelAlpha<T>)
{
    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    const T alphaVec(aAlpha);

    InvokeMulInplaceCScale(PointerRoi(), Pitch(), alphaVec, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelAlpha<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcTA = Vector4<remove_vector_t<T>>;

    InvokeAlphaPremulACSrc(reinterpret_cast<SrcTA *>(PointerRoi()), Pitch(),
                           reinterpret_cast<SrcTA *>(aDst.PointerRoi()), aDst.Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, const opp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelAlpha<T>
{
    using SrcTA = Vector4<remove_vector_t<T>>;
    InvokeAlphaPremulACInplace(reinterpret_cast<SrcTA *>(PointerRoi()), Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region AlphaComp
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires(!FourChannelAlpha<T>) && RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeAlphaCompSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                          aAlphaOp, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                                      remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeAlphaCompCSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                           aAlpha1, aAlpha2, aAlphaOp, SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Complex
template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                    const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeConjMulSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                        SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeConjMulInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj(ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeConjSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj(const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    InvokeConjInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Magnitude(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeMagnitudeSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::MagnitudeSqr(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeMagnitudeSqrSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Angle(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeAngleSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Real(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeRealSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Imag(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeImagSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires RealSignedVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeMakeComplexSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    const ImageView<T> &aSrcImag,
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
    const opp::cuda::StreamCtx &aStreamCtx)
    requires RealSignedVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aSrcImag.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeMakeComplexSrcSrc(PointerRoi(), Pitch(), aSrcImag.PointerRoi(), aSrcImag.Pitch(), aDst.PointerRoi(),
                            aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND