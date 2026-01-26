#pragma once
#include "arithmetic/addSquareProductWeightedOutputType.h"
#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/arithmetic/arithmeticKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/validateImage.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp::image::cuda
{
#pragma region Add
template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeAddSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAddSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAddSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeAddInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeAddInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeAddInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeAddSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeAddSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeAddSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    InvokeAddInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeAddInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeAddInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeAddInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Sub
template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeSubSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeSubSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeSubSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeSubInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeSubInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeSubInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeSubInvInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                                aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeSubInvInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeSubInvInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeSubSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeSubSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeSubSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    InvokeSubInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeSubInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeSubInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    InvokeSubInvInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                    aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeSubInvInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                                  SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeSubInvInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                                aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                         int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeSubInvInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                     scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region Mul
template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeMulSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeMulSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeMulSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeMulInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeMulInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeMulInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeMulSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeMulSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeMulSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    InvokeMulInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeMulInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeMulInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeMulInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region MulScale

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkNullptr(aConst);
    validateImage(*this);
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                           const ImageView<Pixel8uC1> &aMask,
                                           const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                           const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                           const ImageView<Pixel8uC1> &aMask,
                                           const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                           const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                           const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                           const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    InvokeMulInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Div
template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeDivSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeDivSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor, RoundingMode aRoundingMode,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcCScale(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, aRoundingMode,
                       SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeDivSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst, int aScaleFactor,
                                RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat,
                          aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeDivInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat, aRoundingMode,
                             SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeDivInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeDivInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, aRoundingMode, SizeRoi(),
                              aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeDivInvInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceSrcScale(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), scaleFactorFloat,
                                aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeDivInvInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceCScale(PointerRoi(), Pitch(), aConst, scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeDivInvInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const mpp::cuda::DevVarView<T> &aConst, int aScaleFactor, RoundingMode aRoundingMode,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceDevCScale(PointerRoi(), Pitch(), aConst.Pointer(), scaleFactorFloat, aRoundingMode, SizeRoi(),
                                 aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeDivSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                        aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, RoundingMode aRoundingMode,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                             aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(),
                             aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeDivSrcCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(),
                      SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, RoundingMode aRoundingMode,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, aDst.PointerRoi(),
                           aDst.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeDivSrcDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(),
                         aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, int aScaleFactor, RoundingMode aRoundingMode,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivSrcDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                              aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    InvokeDivInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                            SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeDivInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                               aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeDivInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                             aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, RoundingMode aRoundingMode,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                  scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    InvokeDivInvInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                         RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceSrcScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                    aSrc2.Pitch(), scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeDivInvInplaceCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                         RoundingMode aRoundingMode, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst, scaleFactorFloat,
                                  aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    InvokeDivInvInplaceDevCMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(),
                                aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                         int aScaleFactor, RoundingMode aRoundingMode,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aMask);
    checkSameSize(*this, aMask);

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    InvokeDivInvInplaceDevCScaleMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aConst.Pointer(),
                                     scaleFactorFloat, aRoundingMode, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region AddSquare
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddSquare(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                            const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrcDst);
    checkSameSize(*this, aSrcDst);

    InvokeAddSquareInplaceSrc(aSrcDst.PointerRoi(), aSrcDst.Pitch(), PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddSquareMasked(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                                  const ImageView<Pixel8uC1> &aMask,
                                                                  const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrcDst);
    validateImage(aMask);
    checkSameSize(*this, aSrcDst);
    checkSameSize(*this, aMask);

    InvokeAddSquareInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), aSrcDst.PointerRoi(), aSrcDst.Pitch(),
                                  PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}
#pragma endregion

#pragma region AddProduct
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddProduct(const ImageView<T> &aSrc2,
                                                             ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                             const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aSrcDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aSrcDst);

    InvokeAddProductInplaceSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aSrcDst.PointerRoi(),
                                  aSrcDst.Pitch(), SizeRoi(), aStreamCtx);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddProductMasked(const ImageView<T> &aSrc2,
                                                                   ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                                   const ImageView<Pixel8uC1> &aMask,
                                                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aSrcDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aSrcDst);
    checkSameSize(*this, aMask);

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
                                                              const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeAddWeightedSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                            aAlpha, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeightedMasked(const ImageView<T> &aSrc2,
                                                                    ImageView<add_spw_output_for_t<T>> &aDst,
                                                                    remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                                    const ImageView<Pixel8uC1> &aMask,
                                                                    const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    validateImage(aMask);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);
    checkSameSize(*this, aMask);

    InvokeAddWeightedSrcSrcMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                              const mpp::cuda::StreamCtx &aStreamCtx)
{
    validateImage(*this);
    validateImage(aSrcDst);
    checkSameSize(*this, aSrcDst);

    InvokeAddWeightedInplaceSrc(aSrcDst.PointerRoi(), aSrcDst.Pitch(), PointerRoi(), Pitch(), aAlpha, SizeRoi(),
                                aStreamCtx);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeightedMasked(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                                    remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                                    const ImageView<Pixel8uC1> &aMask,
                                                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    validateImage(*this);
    validateImage(aSrcDst);
    validateImage(aMask);
    checkSameSize(*this, aSrcDst);
    checkSameSize(*this, aMask);

    InvokeAddWeightedInplaceSrcMask(aMask.PointerRoi(), aMask.Pitch(), aSrcDst.PointerRoi(), aSrcDst.Pitch(),
                                    PointerRoi(), Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return aSrcDst;
}
#pragma endregion

#pragma region Abs
template <PixelType T>
ImageView<T> &ImageView<T>::Abs(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealSignedVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAbsSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Abs(const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealSignedVector<T>
{
    validateImage(*this);
    InvokeAbsInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region AbsDiff
template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealUnsignedVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeAbsDiffSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                        SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealUnsignedVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAbsDiffSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealUnsignedVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAbsDiffSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeAbsDiffInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    validateImage(*this);
    InvokeAbsDiffInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealUnsignedVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeAbsDiffInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region And
template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeAndSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAndSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAndSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeAndInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    InvokeAndInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeAndInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Not
template <PixelType T>
ImageView<T> &ImageView<T>::Not(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeNotSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Not(const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    InvokeNotInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Exp
template <PixelType T>
ImageView<T> &ImageView<T>::Exp(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeExpSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Exp(const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    InvokeExpInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Ln
template <PixelType T>
ImageView<T> &ImageView<T>::Ln(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeLnSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Ln(const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    InvokeLnInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region LShift

template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeLShiftSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    InvokeLShiftInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Or
template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeOrSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                   aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeOrSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeOrSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeOrInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    InvokeOrInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeOrInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region RShift

template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeRShiftSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    InvokeRShiftInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Sqr
template <PixelType T>
ImageView<T> &ImageView<T>::Sqr(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeSqrSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqr(const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    InvokeSqrInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Sqrt
template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeSqrtSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt(const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexVector<T>
{
    validateImage(*this);
    InvokeSqrtInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Xor
template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeXorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                    SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst, ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeXorSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const mpp::cuda::DevVarView<T> &aConst, ImageView<T> &aDst,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeXorSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeXorInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    validateImage(*this);
    InvokeXorInplaceC(PointerRoi(), Pitch(), aConst, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T>
{
    checkNullptr(aConst);
    validateImage(*this);
    InvokeXorInplaceDevC(PointerRoi(), Pitch(), aConst.Pointer(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region AlphaPremul
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires FourChannelNoAlpha<T> && RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAlphaPremulSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T> && RealVector<T>
{
    validateImage(*this);
    InvokeAlphaPremulInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealFloatingVector<T> && (!FourChannelAlpha<T>)
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    const T alphaVec(aAlpha);

    InvokeMulSrcC(PointerRoi(), Pitch(), alphaVec, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T> && (!FourChannelAlpha<T>)
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    const T alphaVec(aAlpha);

    InvokeMulSrcCScale(PointerRoi(), Pitch(), alphaVec, aDst.PointerRoi(), aDst.Pitch(), scaleFactorFloat, SizeRoi(),
                       aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealFloatingVector<T> && (!FourChannelAlpha<T>)
{
    validateImage(*this);
    const T alphaVec(aAlpha);

    InvokeMulInplaceC(PointerRoi(), Pitch(), alphaVec, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealIntVector<T> && (!FourChannelAlpha<T>)
{
    validateImage(*this);
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    const T alphaVec(aAlpha);

    InvokeMulInplaceCScale(PointerRoi(), Pitch(), alphaVec, scaleFactorFloat, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires FourChannelAlpha<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);
    using SrcTA = Vector4<remove_vector_t<T>>;

    InvokeAlphaPremulACSrc(reinterpret_cast<const SrcTA *>(PointerRoi()), Pitch(),
                           reinterpret_cast<SrcTA *>(aDst.PointerRoi()), aDst.Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelAlpha<T>
{
    validateImage(*this);
    using SrcTA = Vector4<remove_vector_t<T>>;
    InvokeAlphaPremulACInplace(reinterpret_cast<SrcTA *>(PointerRoi()), Pitch(), aAlpha, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region AlphaComp
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!FourChannelAlpha<T>) && RealVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeAlphaCompSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                          aAlphaOp, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                                      remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeAlphaCompCSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                           aAlpha1, aAlpha2, aAlphaOp, SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Complex
template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeConjMulSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                        SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeConjMulInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeConjSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj(const mpp::cuda::StreamCtx &aStreamCtx)
    requires ComplexVector<T>
{
    validateImage(*this);
    InvokeConjInplace(PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Magnitude(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeMagnitudeSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::MagnitudeSqr(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeMagnitudeSqrSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Angle(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeAngleSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Real(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeRealSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Imag(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires ComplexVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeImagSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealSignedVector<T> && (!FourChannelAlpha<T>) &&
             (std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
              std::same_as<float, remove_vector_t<T>>)
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeMakeComplexSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    const ImageView<T> &aSrcImag,
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealSignedVector<T> && (!FourChannelAlpha<T>) &&
             (std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
              std::same_as<float, remove_vector_t<T>>)
{
    validateImage(*this);
    validateImage(aSrcImag);
    validateImage(aDst);
    checkSameSize(*this, aSrcImag);
    checkSameSize(*this, aDst);

    InvokeMakeComplexSrcSrc(PointerRoi(), Pitch(), aSrcImag.PointerRoi(), aSrcImag.Pitch(), aDst.PointerRoi(),
                            aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion
} // namespace mpp::image::cuda