#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include "morphology/morphologyComputeT.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/filtering/fixedSizeMaxFilter.h> // for no mask erosion/dilation
#include <backends/cuda/image/filtering/fixedSizeMinFilter.h>
#include <backends/cuda/image/filtering/maxFilter.h>
#include <backends/cuda/image/filtering/minFilter.h>
#include <backends/cuda/image/morphology/morphologyKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/filterArea.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image::cuda
{

#pragma region No mask Erosion/Dilation
template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                     const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Dilation(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                     const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
    {
        // this implies aFilterArea.Size.x == aFilterArea.Size.y
        InvokeFixedSizeMaxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea.Size.x,
                                 aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                 aStreamCtx);
    }
    else
    {
        InvokeMaxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                        aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                    const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Erosion(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                    const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
    {
        // this implies aFilterArea.Size.x == aFilterArea.Size.y
        InvokeFixedSizeMinFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea.Size.x,
                                 aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                 aStreamCtx);
    }
    else
    {
        InvokeMinFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                        aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                    const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                    const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region Erosion

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                    const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                                    const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Erosion(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                    const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                    const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeErosion(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeErosion(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeErosion(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeErosion(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeErosion(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                      aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst,
                                        const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                        const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                                        const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->ErosionGray(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst,
                                        const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                        const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                        const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeErosionGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                   MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                   roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeErosionGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                   MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                   roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeErosionGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                   MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                   roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeErosionGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                   MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                   roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeErosionGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                          aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                    const FilterArea &aFilterArea, BorderType aBorder,
                                    const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                    const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                    const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst,
                                        const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                        const FilterArea &aFilterArea, BorderType aBorder,
                                        const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->ErosionGray(aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst,
                                        const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                        const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                        const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->ErosionGray(aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region Dilation

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                     const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Dilation(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                     const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                     const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeDilation(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeDilation(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeDilation(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeDilation(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeDilation(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                       aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst,
                                         const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                         const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                                         const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->DilationGray(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst,
                                         const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                         const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                         const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeDilationGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                    MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                    roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeDilationGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                    MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                    roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeDilationGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                    MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                    roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeDilationGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                    MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                    roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeDilationGray(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                           aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                     const FilterArea &aFilterArea, BorderType aBorder,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                     const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst,
                                         const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                         const FilterArea &aFilterArea, BorderType aBorder,
                                         const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->DilationGray(aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst,
                                         const opp::cuda::DevVarView<morph_gray_compute_type_t<T>> &aMask,
                                         const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                         const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->DilationGray(aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region Open

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                 const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi,
                                 const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Open(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                 const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                 const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset1 = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr1          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector2<int> roiOffset2 = aTemp.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr2 = gotoPtr(aTemp.Pointer(), aTemp.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeDilation(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeDilation(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeDilation(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeDilation(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset2, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                      aConstant, aAllowedReadRoi.Size(), roiOffset1, SizeRoi(), aStreamCtx);
        InvokeDilation(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea,
                       aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                 const FilterArea &aFilterArea, BorderType aBorder,
                                 const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Open(aTemp, aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                 const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                 const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Open(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region Close
template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst,
                                  const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                  BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Close(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst,
                                  const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                  T aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset1 = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr1          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector2<int> roiOffset2 = aTemp.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr2 = gotoPtr(aTemp.Pointer(), aTemp.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeErosion(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeErosion(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeErosion(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeErosion(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset2, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                       aConstant, aAllowedReadRoi.Size(), roiOffset1, SizeRoi(), aStreamCtx);
        InvokeErosion(allowedPtr2, aTemp.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea,
                      aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst,
                                  const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                  BorderType aBorder, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Close(aTemp, aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst,
                                  const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                  T aConstant, BorderType aBorder, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->Close(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region TopHat
template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                   const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                   BorderType aBorder, const Roi &aAllowedReadRoi,
                                   const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->TopHat(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                   const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                   T aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                   const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset1 = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr1          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector2<int> roiOffset2 = aTemp.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr2 = gotoPtr(aTemp.Pointer(), aTemp.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeTopHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                              aMask.Pointer(), MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeTopHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                              aMask.Pointer(), MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeTopHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                              aMask.Pointer(), MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                               MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                               roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeTopHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                              aMask.Pointer(), MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeErosion(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                      aConstant, aAllowedReadRoi.Size(), roiOffset1, SizeRoi(), aStreamCtx);
        InvokeTopHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                     aMask.Pointer(), aFilterArea, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset2, SizeRoi(),
                     aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                   const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                   BorderType aBorder, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->TopHat(aTemp, aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                   const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                   T aConstant, BorderType aBorder, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->TopHat(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region BlackHat
template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                     const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                     BorderType aBorder, const Roi &aAllowedReadRoi,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BlackHat(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                     const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                     T aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                     const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset1 = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr1          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector2<int> roiOffset2 = aTemp.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr2 = gotoPtr(aTemp.Pointer(), aTemp.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeBlackHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                                aMask.Pointer(), MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant,
                                aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeBlackHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                                aMask.Pointer(), MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant,
                                aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeBlackHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                                aMask.Pointer(), MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant,
                                aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(),
                                MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                                roiOffset1, SizeRoi(), aStreamCtx);
        InvokeFixedSizeBlackHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                                aMask.Pointer(), MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant,
                                aAllowedReadRoi.Size(), roiOffset2, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeDilation(allowedPtr1, Pitch(), aTemp.PointerRoi(), aTemp.Pitch(), aMask.Pointer(), aFilterArea, aBorder,
                       aConstant, aAllowedReadRoi.Size(), roiOffset1, SizeRoi(), aStreamCtx);
        InvokeBlackHat(allowedPtr2, aTemp.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                       aMask.Pointer(), aFilterArea, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset2, SizeRoi(),
                       aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                     const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                     BorderType aBorder, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->BlackHat(aTemp, aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst,
                                     const opp::cuda::DevVarView<Pixel8uC1> &aMask, const FilterArea &aFilterArea,
                                     T aConstant, BorderType aBorder, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->BlackHat(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion

#pragma region Morphology Gradient
template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                               const FilterArea &aFilterArea, BorderType aBorder,
                                               const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->MorphologyGradient(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                               const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                               const Roi &aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeMorphologyGradient(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                          MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant,
                                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeMorphologyGradient(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                          MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant,
                                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeMorphologyGradient(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                          MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant,
                                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeMorphologyGradient(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(),
                                          MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant,
                                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeMorphologyGradient(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMask.Pointer(), aFilterArea,
                                 aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                               const FilterArea &aFilterArea, BorderType aBorder,
                                               const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->MorphologyGradient(aDst, aMask, aFilterArea, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const opp::cuda::DevVarView<Pixel8uC1> &aMask,
                                               const FilterArea &aFilterArea, T aConstant, BorderType aBorder,
                                               const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return this->MorphologyGradient(aDst, aMask, aFilterArea, aConstant, aBorder, ROI(), aStreamCtx);
}
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND