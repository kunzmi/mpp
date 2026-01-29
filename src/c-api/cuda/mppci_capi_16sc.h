#ifndef MPPI_CUDA_CAPI_16SC_H
#define MPPI_CUDA_CAPI_16SC_H

#include "mppc_capi_defs.h"

// for datatype conversions:
#include "convertScale_16sc.h"
// for operations on different channel counts:
#include "copySwapChannelDup_16sc.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32s aLowerBorderSize[2], MPPBorderType aBorder,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    /// <param name="aBorder">Border control paramter</param>
    MPPErrorCode mppciCopyBorder_16sc_C1Cb(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2], Mpp16sc aConstant,
                                           MPPBorderType aBorder, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C1IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, ConstDevPtrMpp8u aMask,
                                     size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C1IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                      size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                       size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                      size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                       size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                      size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                       size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                      size_t aDstStep, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C1Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                         MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aScaleFactor,
                                       MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C1ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst, DevPtrMpp16sc aDst,
                                       size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C1MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C1IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C1I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C1I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C1I(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                               size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C1IM(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                                size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                                size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_16sc16s_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_16sc16s_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, Mpp16sc aConstant,
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, Mpp16sc aConstant, MPPBorderType aBorder,
                                                 MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              Mpp16sc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                     Mpp32s aFilterCenter, Mpp16sc aConstant, MPPBorderType aBorder,
                                                     MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                   Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           Mpp16sc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                  MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                  MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                  Mpp32s aFilterCenter, Mpp16sc aConstant, MPPBorderType aBorder,
                                                  MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, Mpp16sc aConstant, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea, Mpp16sc aConstant,
                                        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
                                      MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            Mpp16sc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
                                            CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                          const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                          MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, Mpp16sc aConstant,
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, Mpp16sc aConstant,
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, Mpp16sc aConstant,
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                   MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                                   MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, Mpp16sc aConstant,
                                        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                      const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                      MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPInterpolationMode aInterpolation, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, Mpp16sc aConstant,
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, Mpp16sc aConstant, MPPBorderType aBorder,
                                         MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C1RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, Mpp16sc aConstant, MPPBorderType aBorder,
                                       MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C1R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, ConstDevPtrMpp8u aMask,
                                               size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                                       size_t aBufferSize, MppiSize aSizeROI,
                                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the dot product of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the dot product is defined as: DotProduct = Sum(Src1(i,j) * Src2(i,j))
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciDotProduct_16sc32fc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp8u aBuffer,
                                             size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, ConstDevPtrMpp8u aMask,
                                              size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the Mean Square Error of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the MSE is defined as: MSE = Sum((Src1(i,j) - Src2(i,j))^2) / (W*H)
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMSE_16sc32fc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp32fc aDst, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, ConstDevPtrMpp8u aMask,
                                               size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                                       size_t aBufferSize, MppiSize aSizeROI,
                                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                        size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.
    /// </summary>
    /// <param name="aMean">Mean value</param>
    /// <param name="aStd">Standard deviation</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                      DevPtrMpp64f aStd, DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aMean">Mean value</param>
    /// <param name="aStd">Standard deviation</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C1M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst,
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, Mpp16sc aValue, DevPtrMpp16sc aDst,
                                        size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp16sc aConst,
                                         MPPCompareOp aCompare, Mpp16sc aValue, DevPtrMpp16sc aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, Mpp16sc aValue, DevPtrMpp16sc aDst, size_t aDstStep,
                                            MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, Mpp16sc aValue, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst,
                                          MPPCompareOp aCompare, Mpp16sc aValue, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C1I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             MPPCompareOp aCompare, Mpp16sc aValue, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2P2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDstChannel1,
                                     size_t aDstChannel1Step, DevPtrMpp16sc aDstChannel2, size_t aDstChannel2Step,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_P2C2(DevPtrMpp16sc aSrcChannel1, size_t aSrcChannel1Step, DevPtrMpp16sc aSrcChannel2,
                                     size_t aSrcChannel2Step, DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32s aLowerBorderSize[2], MPPBorderType aBorder,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    /// <param name="aBorder">Border control paramter</param>
    MPPErrorCode mppciCopyBorder_16sc_C2Cb(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2],
                                           const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C2IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C2IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C2CI(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aChannel,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C2CI(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        Mpp32s aChannel, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels for two channel images.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (inplace) for two channel images.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                      MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C2Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                         MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                       Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C2ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C2MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C2IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C2I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C2I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C2I(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                               size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C2IM(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                                size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                                size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_16sc16s_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_16sc16s_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, const Mpp16sc aConstant[2],
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp16sc aConstant[2],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                     Mpp32s aFilterCenter, const Mpp16sc aConstant[2],
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                   Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                  MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                  MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                  Mpp32s aFilterCenter, const Mpp16sc aConstant[2],
                                                  MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, const Mpp16sc aConstant[2],
                                           MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea,
                                        const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
                                      MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                            CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                          const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                          MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                            ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                            size_t aDst2Step, MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                            MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                          ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                          MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                          MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                          CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                DevPtrMpp16sc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                                const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                                CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                              size_t aDst2Step, MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                 DevPtrMpp16sc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                 const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                                 const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                                 CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                               ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                               size_t aDst2Step, MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                               MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                               MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                   MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                                   MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_P2RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        MppiSize aDstSize, const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
        const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                   DevPtrMpp16sc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                   const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                                   MPPBorderType aBorder, MppiRect aSrcROI,
                                                   CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp16sc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                      const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                      MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                        ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                        MppiSize aDstSize, Mpp64f aAngleInDeg, const Mpp64f aShift[2],
                                        MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                      ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                      MppiSize aDstSize, Mpp64f aAngleInDeg, const Mpp64f aShift[2],
                                      MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPInterpolationMode aInterpolation, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_P2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                     size_t aDst2Step, MPPInterpolationMode aInterpolation, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                DevPtrMpp16sc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                              size_t aDst2Step, MppiSize aDstSize, const Mpp64f aScale[2],
                                              const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                       MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                         ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                         MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                       MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_P2RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                       MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[2],
                                       MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_P2R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr,
                                     size_t aSrc2Step, MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                     DevPtrMpp16sc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                              DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/> For multi-channel images, the result is computed for each channel seperatly in
    /// aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute value is used
    /// for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp64f aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                             DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                              DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/> For multi-channel images, the result is computed for each channel
    /// seperatly in aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute
    /// value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp64f aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                      DevPtrMpp64sc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                      DevPtrMpp64fc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                       DevPtrMpp64sc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                        DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar is also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar is also
    /// nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                      DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values where only pixels with mask != 0 are used.<para/>For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar is also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar is also
    /// nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C2M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C2C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C2C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                           MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C2C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                              MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, const Mpp16sc aValue[2],
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[2],
                                         MPPCompareOp aCompare, const Mpp16sc aValue[2], DevPtrMpp16sc aDst,
                                         size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, const Mpp16sc aValue[2], DevPtrMpp16sc aDst,
                                            size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, const Mpp16sc aValue[2],
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[2],
                                          MPPCompareOp aCompare, const Mpp16sc aValue[2], MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C2I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             MPPCompareOp aCompare, const Mpp16sc aValue[2], MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3P3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDstChannel1,
                                     size_t aDstChannel1Step, DevPtrMpp16sc aDstChannel2, size_t aDstChannel2Step,
                                     DevPtrMpp16sc aDstChannel3, size_t aDstChannel3Step, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_P3C3(DevPtrMpp16sc aSrcChannel1, size_t aSrcChannel1Step, DevPtrMpp16sc aSrcChannel2,
                                     size_t aSrcChannel2Step, DevPtrMpp16sc aSrcChannel3, size_t aSrcChannel3Step,
                                     DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32s aLowerBorderSize[2], MPPBorderType aBorder,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    /// <param name="aBorder">Border control paramter</param>
    MPPErrorCode mppciCopyBorder_16sc_C3Cb(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2],
                                           const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C3IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C3IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C3CI(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aChannel,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C3CI(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        Mpp32s aChannel, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (inplace)<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts aSrcDst to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp32s aDstChannels[3],
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                      MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C3Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                         MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                       Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C3ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C3MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C3IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C3I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C3I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C3I(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                               size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C3IM(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                                size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                                size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_16sc16s_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_16sc16s_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, const Mpp16sc aConstant[3],
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp16sc aConstant[3],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                     Mpp32s aFilterCenter, const Mpp16sc aConstant[3],
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                   Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                  MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                  MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                  Mpp32s aFilterCenter, const Mpp16sc aConstant[3],
                                                  MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, const Mpp16sc aConstant[3],
                                           MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea,
                                        const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
                                      MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                            CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                          const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                          MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_P3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                            ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                            ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                            size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                            CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                          ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                          ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                          DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                          const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                          MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_P3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3,
                                                size_t aDst3Step, MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                              ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                              size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step,
                                              MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_P3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                 ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                 DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3,
                                                 size_t aDst3Step, MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                               ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                               ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                               size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step,
                                               MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                               MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                               MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                   MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                                   MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_P3RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
        DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                                   ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                                   DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3,
                                                   size_t aDst3Step, MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                   MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                                   MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                      const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                      MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_P3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                        ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                        DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp16sc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                      ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                      ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                      DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                      const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                      MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPInterpolationMode aInterpolation, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_P3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, ConstDevPtrMpp16sc aSrc3, size_t aSrc3Step, DevPtrMpp16sc aDst1,
                                     size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3,
                                     size_t aDst3Step, MPPInterpolationMode aInterpolation, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_P3RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step,
        DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
        const Mpp64f aScale[2], const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                              ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                              size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step,
                                              MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                       MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_P3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                         ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                         ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                         DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                       DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_P3RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                       DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[3],
                                       MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_P3R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr,
                                     size_t aSrc2Step, ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                     MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                     size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                              DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/> For multi-channel images, the result is computed for each channel seperatly in
    /// aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute value is used
    /// for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp64f aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                             DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                              DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/> For multi-channel images, the result is computed for each channel
    /// seperatly in aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute
    /// value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp64f aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                      DevPtrMpp64sc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                      DevPtrMpp64fc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                       DevPtrMpp64sc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                        DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar is also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar is also
    /// nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                      DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values where only pixels with mask != 0 are used.<para/>For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar is also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar is also
    /// nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C3M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C3C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C3C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                           MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C3C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                              MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, const Mpp16sc aValue[3],
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[3],
                                         MPPCompareOp aCompare, const Mpp16sc aValue[3], DevPtrMpp16sc aDst,
                                         size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, const Mpp16sc aValue[3], DevPtrMpp16sc aDst,
                                            size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, const Mpp16sc aValue[3],
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[3],
                                          MPPCompareOp aCompare, const Mpp16sc aValue[3], MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C3I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             MPPCompareOp aCompare, const Mpp16sc aValue[3], MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4P4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDstChannel1,
                                     size_t aDstChannel1Step, DevPtrMpp16sc aDstChannel2, size_t aDstChannel2Step,
                                     DevPtrMpp16sc aDstChannel3, size_t aDstChannel3Step, DevPtrMpp16sc aDstChannel4,
                                     size_t aDstChannel4Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_P4C4(DevPtrMpp16sc aSrcChannel1, size_t aSrcChannel1Step, DevPtrMpp16sc aSrcChannel2,
                                     size_t aSrcChannel2Step, DevPtrMpp16sc aSrcChannel3, size_t aSrcChannel3Step,
                                     DevPtrMpp16sc aSrcChannel4, size_t aSrcChannel4Step, DevPtrMpp16sc aDst,
                                     size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32s aLowerBorderSize[2], MPPBorderType aBorder,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    /// <param name="aBorder">Border control paramter</param>
    MPPErrorCode mppciCopyBorder_16sc_C4Cb(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2],
                                           const Mpp16sc aConstant[4], MPPBorderType aBorder, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C4IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C4IM(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_16sc_C4CI(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, Mpp16sc aConst, Mpp32s aChannel,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_16sc_C4CI(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                        Mpp32s aChannel, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (inplace)<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts aSrcDst to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp32s aDstChannels[4],
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSub_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMul_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                     MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                      MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C4Sfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                         DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aScaleFactor,
                                         MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                       Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C4ISfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                      size_t aMaskStep, Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                       DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels
    /// where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C4MSfs(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                          DevPtrMpp16sc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                       MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                        MPPRoundingMode aRoundingMode, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all pixels where
    /// aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                          Mpp32s aScaleFactor, MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                           MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, with floating point scaling factor with scale factor = 2^-aScaleFactor, for all
    /// pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_16sc_C4IMSfs(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, Mpp32s aScaleFactor,
                                              MPPRoundingMode aRoundingMode, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C4I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_16sc32fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                             size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C4I(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_16sc32fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C4I(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                               size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_16sc32fc_C4IM(DevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                                size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                                size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_16sc16s_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_16sc16s_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, const Mpp16sc aConstant[4],
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp16sc aConstant[4],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp16sc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                     Mpp32s aFilterCenter, const Mpp16sc aConstant[4],
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_16sc32fc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                   Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           const Mpp16sc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                  MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                  MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                  Mpp32s aFilterCenter, const Mpp16sc aConstant[4],
                                                  MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_16sc32fc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                Mpp32s aFilterCenter, MPPBorderType aBorder, MppiRect aSrcROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, const Mpp16sc aConstant[4],
                                           MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea,
                                        const Mpp16sc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
                                      MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp16sc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                            CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                          const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                          MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_P4RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffine_16sc_P4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                          ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                          ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                          ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                          DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                          DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                          MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                          MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                          CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_P4RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpAffineBack_16sc_P4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                              ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                              ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                              size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step,
                                              DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
                                              const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                               const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_P4RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspective_16sc_P4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                               ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                               ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                               ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                               DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2,
                                               size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step,
                                               DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
                                               const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                                     MPPBorderType aBorder, MppiRect aSrcROI,
                                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                   MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                   MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                                   MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_P4RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_16sc_P4R(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
        CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp16sc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                      const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                      MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_P4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                        ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                        ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                        DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                        MppiSize aDstSize, Mpp64f aAngleInDeg, const Mpp64f aShift[2],
                                        MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRotate_16sc_P4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                      ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                      ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                      ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                      DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                      DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                      MppiSize aDstSize, Mpp64f aAngleInDeg, const Mpp64f aShift[2],
                                      MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPInterpolationMode aInterpolation, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Resize<para/>
    /// Simplified API to rescale from source image ROI to destination image ROI.<para/>
    /// NOTE: the result is NOT the same as in NPP using the same function. The shift applied in NPP for the same
    /// function don't make much sense to me, in MPP Resize matches the input extent [-0.5 .. srcWidth-0.5[ to the
    /// output [-0.5 .. dstWidth-0.5[. Whereas NPP applies different strategies for up-and downscaling. In order to get
    /// the same results as in NPP, use an user defined scaling factor of <para/> Vec2d scaleFactor =
    /// Vec2d(dstImg.SizeRoi()) / Vec2d(srcImg.SizeRoi());<para/> and a shift given by ResizeGetNPPShift().
    /// </summary>
    MPPErrorCode mppciResize_16sc_P4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                     size_t aSrc2Step, ConstDevPtrMpp16sc aSrc3, size_t aSrc3Step,
                                     ConstDevPtrMpp16sc aSrc4, size_t aSrc4Step, DevPtrMpp16sc aDst1, size_t aDst1Step,
                                     DevPtrMpp16sc aDst2, size_t aDst2Step, DevPtrMpp16sc aDst3, size_t aDst3Step,
                                     DevPtrMpp16sc aDst4, size_t aDst4Step, MPPInterpolationMode aInterpolation,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp16sc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                                MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_P4RCb(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aScale[2], const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciResizeSqrPixel_16sc_P4R(
        ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
        DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aScale[2], const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
        MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                       MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_C4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_P4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                         ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                         ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                         ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                         DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                         DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                         MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemapC2_16sc_P4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                       ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                       DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                       MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                       MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                       CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_P4RCb(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp16sc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                       ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                       DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                       DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                       MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp16sc aConstant[4],
                                       MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciRemap_16sc_P4R(ConstDevPtrMpp16sc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2BasePtr,
                                     size_t aSrc2Step, ConstDevPtrMpp16sc aSrc3BasePtr, size_t aSrc3Step,
                                     ConstDevPtrMpp16sc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                     DevPtrMpp16sc aDst1, size_t aDst1Step, DevPtrMpp16sc aDst2, size_t aDst2Step,
                                     DevPtrMpp16sc aDst3, size_t aDst3Step, DevPtrMpp16sc aDst4, size_t aDst4Step,
                                     MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                              DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average error is defined as: AverageError = Sum(|Src1(i,j) - Src2(i,j)|)/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageError_16sc64f_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/> For multi-channel images, the result is computed for each channel seperatly in
    /// aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute value is used
    /// for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp64f aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the average relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the average relative error is defined as: AverageRelativeError = Sum(|Src1(i,j) - Src2(i,j)| / max(|Src1(i,j)|,
    /// |Src2(i,j)|))/(W * H) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciAverageRelativeError_16sc64f_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
                                                    CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                             DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciDotProduct_16sc32fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_16sc32fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp32fc aDst, DevPtrMpp32fc aDstScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
                                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                              DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum error is defined as: MaximumError = max(|Src1(i,j) - Src2(i,j)|) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.<para/>
    /// Note: Same as NormDiffInf
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumError_16sc64f_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
                                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/> For multi-channel images, the result is computed for each channel
    /// seperatly in aDst, or for all channels in aDstScalar. <para/> If the image is in complex format, the absolute
    /// value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                      DevPtrMpp64f aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the maximum relative error between two images where only pixels with mask != 0 are used.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the maximum relative error is defined as: MaximumRelativeError = max((|Src1(i,j) - Src2(i,j)|) /
    /// max(|Src1(i,j)|, |Src2(i,j)|)) <para/>
    /// For multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar. <para/> If the image is in complex format, the absolute value is used for computation.
    /// </summary>
    /// <param name="aSrc2">Second image to compare aSrc1 image to</param>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMaximumRelativeError_16sc64f_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp16sc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64s_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_16sc64f_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                      DevPtrMpp64sc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                      DevPtrMpp64fc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64sc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64sc aDst,
                                       DevPtrMpp64sc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_16sc64fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_16sc64fc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                        DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_16sc_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar is also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar is also
    /// nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                      DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values where only pixels with mask != 0 are used.<para/>For
    /// multi-channel images, the result is computed for each channel seperatly in aDst, or for all channels in
    /// aDstScalar.
    /// </summary>
    /// <param name="aMean">Per-channel mean value, can be nullptr if aStd is also nullptr</param>
    /// <param name="aStd">Per-channel standard deviation value, can be nullptr if aMean is also nullptr</param>
    /// <param name="aMeanScalar">Mean value for all channels, can be nullptr if aStdScalar is also nullptr</param>
    /// <param name="aStdScalar">Standard deviation for all channels, can be nullptr if aMeanScalar is also
    /// nullptr</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_16sc_C4M(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C4C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                          size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C4C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                           MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C4C1(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                              MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompare_16sc8u_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareC_16sc8u_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareDevC_16sc8u_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, const Mpp16sc aValue[4],
                                        DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, const Mpp16sc aConst[4],
                                         MPPCompareOp aCompare, const Mpp16sc aValue[4], DevPtrMpp16sc aDst,
                                         size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, ConstDevPtrMpp16sc aConst,
                                            MPPCompareOp aCompare, const Mpp16sc aValue[4], DevPtrMpp16sc aDst,
                                            size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, const Mpp16sc aValue[4],
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, const Mpp16sc aConst[4],
                                          MPPCompareOp aCompare, const Mpp16sc aValue[4], MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_16sc_C4I(DevPtrMpp16sc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp16sc aConst,
                                             MPPCompareOp aCompare, const Mpp16sc aValue[4], MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_16SC_H
