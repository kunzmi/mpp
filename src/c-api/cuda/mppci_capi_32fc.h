#ifndef MPPI_CUDA_CAPI_32FC_H
#define MPPI_CUDA_CAPI_32FC_H

#include "mppc_capi_defs.h"

// for datatype conversions:
#include "convertScale_32fc.h"
// for operations on different channel counts:
#include "copySwapChannelDup_32fc.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciCopyBorder_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
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
    MPPErrorCode mppciCopyBorder_32fc_C1Cb(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2], Mpp32fc aConstant,
                                           MPPBorderType aBorder, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, ConstDevPtrMpp8u aMask,
                                     size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                    size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, ConstDevPtrMpp8u aMask,
                                     size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                    size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, ConstDevPtrMpp8u aMask,
                                     size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                    size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, ConstDevPtrMpp8u aMask,
                                     size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst, DevPtrMpp32fc aDst,
                                    size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, ConstDevPtrMpp8u aMask,
                                     size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C1IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C1I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C1I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                           size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C1I(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                           size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C1IM(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                            size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                            size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1) (complex magnitude)
    /// </summary>
    MPPErrorCode mppciMagnitude_32fc32f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                           size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1)^2 (complex magnitude squared)
    /// </summary>
    MPPErrorCode mppciMagnitudeSqr_32fc32f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = angle(aSrc1) (complex angle, atan2(imag, real))
    /// </summary>
    MPPErrorCode mppciAngle_32fc32f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_32fc32f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_32fc32f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, Mpp32fc aConstant,
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, Mpp32fc aConstant, MPPBorderType aBorder,
                                                 MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              Mpp32fc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, Mpp32fc aConstant, MPPBorderType aBorder,
                                                 MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           Mpp32fc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              Mpp32fc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, Mpp32fc aConstant, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea, Mpp32fc aConstant,
                                        MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
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
    MPPErrorCode mppciWarpAffine_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            Mpp32fc aConstant, MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpAffine_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffineBack_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, Mpp32fc aConstant,
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
    MPPErrorCode mppciWarpAffineBack_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspective_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, Mpp32fc aConstant,
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
    MPPErrorCode mppciWarpPerspective_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, Mpp32fc aConstant,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciRotate_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, Mpp32fc aConstant,
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
    MPPErrorCode mppciRotate_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
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
    MPPErrorCode mppciResize_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, Mpp32fc aConstant,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
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
    MPPErrorCode mppciRemapC2_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, Mpp32fc aConstant, MPPBorderType aBorder,
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
    MPPErrorCode mppciRemapC2_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemap_32fc_C1RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, Mpp32fc aConstant, MPPBorderType aBorder,
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
    MPPErrorCode mppciRemap_32fc_C1R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageError_32fc64f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciAverageError_32fc64f_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, ConstDevPtrMpp8u aMask,
                                               size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                                       size_t aBufferSize, MppiSize aSizeROI,
                                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp8u aBuffer,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64fc aDst, ConstDevPtrMpp8u aMask,
                                              size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the Mean Square Error of two images.<para/>
    /// Given two images Src1 and Src2 both with width W and height H,
    /// the MSE is defined as: MSE = Sum((Src1(i,j) - Src2(i,j))^2) / (W*H)
    /// </summary>
    /// <param name="aSrc2">Second source image</param>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMSE_32fc64fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp8u aBuffer, size_t aBufferSize,
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
    MPPErrorCode mppciMSE_32fc64fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp64fc aDst, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumError_32fc64f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciMaximumError_32fc64f_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, ConstDevPtrMpp8u aMask,
                                               size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                                       size_t aBufferSize, MppiSize aSizeROI,
                                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_32fc64fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                      DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_32fc64fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_32fc64fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values where only pixels with mask != 0 are used.
    /// </summary>
    /// <param name="aDst">Result</param>
    /// <param name="aMask"></param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_32fc64fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                        size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C1(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C1M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean and standard deviation of pixel values.
    /// </summary>
    /// <param name="aMean">Mean value</param>
    /// <param name="aStd">Standard deviation</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMeanStd_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
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
    MPPErrorCode mppciMeanStd_32fc_C1M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst,
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel
    /// images:<para/> CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/> CompareOp::Eq |
    /// CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aSrc2) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEps_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                             size_t aSrc2Step, Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsC_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst,
                                              Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsDevC_32fc8u_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                                 Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                                 CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, Mpp32fc aValue, DevPtrMpp32fc aDst,
                                        size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32fc aConst,
                                         MPPCompareOp aCompare, Mpp32fc aValue, DevPtrMpp32fc aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, Mpp32fc aValue, DevPtrMpp32fc aDst, size_t aDstStep,
                                            MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), aSrc1
    /// otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             Mpp32fc aValue, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, Mpp32fc aValue, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst,
                                          MPPCompareOp aCompare, Mpp32fc aValue, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                             MPPCompareOp aCompare, Mpp32fc aValue, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst fulfills aCompare (for floating point checks, e.g. isinf()), aSrcDst
    /// otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C1I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPCompareOp aCompare,
                                              Mpp32fc aValue, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C2P2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDstChannel1,
                                     size_t aDstChannel1Step, DevPtrMpp32fc aDstChannel2, size_t aDstChannel2Step,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_P2C2(DevPtrMpp32fc aSrcChannel1, size_t aSrcChannel1Step, DevPtrMpp32fc aSrcChannel2,
                                     size_t aSrcChannel2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
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
    MPPErrorCode mppciCopyBorder_32fc_C2Cb(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2],
                                           const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C2CI(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, Mpp32s aChannel,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C2CI(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        Mpp32s aChannel, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels for two channel images.
    /// </summary>
    MPPErrorCode mppciSwapChannel_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (inplace) for two channel images.
    /// </summary>
    MPPErrorCode mppciSwapChannel_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C2IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C2I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C2I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                           size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C2I(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                           size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C2IM(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                            size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                            size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1) (complex magnitude)
    /// </summary>
    MPPErrorCode mppciMagnitude_32fc32f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                           size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1)^2 (complex magnitude squared)
    /// </summary>
    MPPErrorCode mppciMagnitudeSqr_32fc32f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = angle(aSrc1) (complex angle, atan2(imag, real))
    /// </summary>
    MPPErrorCode mppciAngle_32fc32f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_32fc32f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_32fc32f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, const Mpp32fc aConstant[2],
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp32fc aConstant[2],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp32fc aConstant[2],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, const Mpp32fc aConstant[2],
                                           MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea,
                                        const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
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
    MPPErrorCode mppciWarpAffine_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpAffine_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffine_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                            ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                            size_t aDst2Step, MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                            MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciWarpAffine_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                          ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
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
    MPPErrorCode mppciWarpAffineBack_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciWarpAffineBack_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffineBack_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                DevPtrMpp32fc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                                const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpAffineBack_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
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
    MPPErrorCode mppciWarpPerspective_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciWarpPerspective_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspective_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                 DevPtrMpp32fc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                 const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
                                                 const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpPerspective_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                               ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_P2RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        MppiSize aDstSize, const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation,
        const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                   ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                   DevPtrMpp32fc aDst2, size_t aDst2Step, MppiSize aDstSize,
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
    MPPErrorCode mppciRotate_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp32fc aConstant[2], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciRotate_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
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
    MPPErrorCode mppciRotate_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                        ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                        MppiSize aDstSize, Mpp64f aAngleInDeg, const Mpp64f aShift[2],
                                        MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciRotate_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                      ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
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
    MPPErrorCode mppciResize_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciResize_32fc_P2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                     size_t aSrc2Step, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                DevPtrMpp32fc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                                const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciResizeSqrPixel_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                              size_t aDst2Step, MppiSize aDstSize, const Mpp64f aScale[2],
                                              const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                              MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
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
    MPPErrorCode mppciRemapC2_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciRemapC2_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemap_32fc_C2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciRemap_32fc_C2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemapC2_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                         ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                         MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciRemapC2_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
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
    MPPErrorCode mppciRemap_32fc_P2RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                       MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[2],
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
    MPPErrorCode mppciRemap_32fc_P2R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr,
                                     size_t aSrc2Step, MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                     DevPtrMpp32fc aDst2, size_t aDst2Step, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageError_32fc64f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciAverageError_32fc64f_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_32fc64fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
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
    MPPErrorCode mppciMSE_32fc64fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumError_32fc64f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciMaximumError_32fc64f_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_32fc64fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
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
    MPPErrorCode mppciSum_32fc64fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_32fc64fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
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
    MPPErrorCode mppciMean_32fc64fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                        DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C2(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C2M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMeanStd_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
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
    MPPErrorCode mppciMeanStd_32fc_C2M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                           MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                              MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel
    /// images:<para/> CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/> CompareOp::Eq |
    /// CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                               DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The comparison is performed for each channel individually and the flag CompareOp::PerChannel
    /// must be set for aCompare.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aSrc2) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEps_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsC_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                                Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsDevC_32fc8u_C2C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                   ConstDevPtrMpp32fc aConst, Mpp32f aEpsilon, DevPtrMpp8u aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, const Mpp32fc aValue[2],
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[2],
                                         MPPCompareOp aCompare, const Mpp32fc aValue[2], DevPtrMpp32fc aDst,
                                         size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, const Mpp32fc aValue[2], DevPtrMpp32fc aDst,
                                            size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), aSrc1
    /// otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             const Mpp32fc aValue[2], DevPtrMpp32fc aDst, size_t aDstStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, const Mpp32fc aValue[2],
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[2],
                                          MPPCompareOp aCompare, const Mpp32fc aValue[2], MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                             MPPCompareOp aCompare, const Mpp32fc aValue[2], MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst fulfills aCompare (for floating point checks, e.g. isinf()), aSrcDst
    /// otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C2I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPCompareOp aCompare,
                                              const Mpp32fc aValue[2], MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C3P3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDstChannel1,
                                     size_t aDstChannel1Step, DevPtrMpp32fc aDstChannel2, size_t aDstChannel2Step,
                                     DevPtrMpp32fc aDstChannel3, size_t aDstChannel3Step, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_P3C3(DevPtrMpp32fc aSrcChannel1, size_t aSrcChannel1Step, DevPtrMpp32fc aSrcChannel2,
                                     size_t aSrcChannel2Step, DevPtrMpp32fc aSrcChannel3, size_t aSrcChannel3Step,
                                     DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
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
    MPPErrorCode mppciCopyBorder_32fc_C3Cb(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2],
                                           const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C3CI(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, Mpp32s aChannel,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C3CI(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        Mpp32s aChannel, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (inplace)<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts aSrcDst to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32s aDstChannels[3],
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C3IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C3I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C3I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                           size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C3I(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                           size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C3IM(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                            size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                            size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1) (complex magnitude)
    /// </summary>
    MPPErrorCode mppciMagnitude_32fc32f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                           size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1)^2 (complex magnitude squared)
    /// </summary>
    MPPErrorCode mppciMagnitudeSqr_32fc32f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = angle(aSrc1) (complex angle, atan2(imag, real))
    /// </summary>
    MPPErrorCode mppciAngle_32fc32f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_32fc32f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_32fc32f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, const Mpp32fc aConstant[3],
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp32fc aConstant[3],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp32fc aConstant[3],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, const Mpp32fc aConstant[3],
                                           MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea,
                                        const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
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
    MPPErrorCode mppciWarpAffine_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpAffine_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffine_32fc_P3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                            ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                            ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                            size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpAffine_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                          ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                          ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                          DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffineBack_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciWarpAffineBack_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffineBack_32fc_P3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3,
                                                size_t aDst3Step, MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciWarpAffineBack_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                              ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                              size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step,
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
    MPPErrorCode mppciWarpPerspective_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciWarpPerspective_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspective_32fc_P3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                 ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                 DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3,
                                                 size_t aDst3Step, MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciWarpPerspective_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                               ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                               ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                               size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_P3RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
        DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                   ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                                   ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                                   DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3,
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
    MPPErrorCode mppciRotate_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciRotate_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
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
    MPPErrorCode mppciRotate_32fc_P3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                        ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                        DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp32fc aConstant[3], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciRotate_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                      ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                      ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                      DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize, Mpp64f aAngleInDeg,
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
    MPPErrorCode mppciResize_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciResize_32fc_P3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                     size_t aSrc2Step, ConstDevPtrMpp32fc aSrc3, size_t aSrc3Step, DevPtrMpp32fc aDst1,
                                     size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_P3RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step,
        DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
        const Mpp64f aScale[2], const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciResizeSqrPixel_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                              ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                              size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step,
                                              MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                              MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
                                              MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
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
    MPPErrorCode mppciRemapC2_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciRemapC2_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemap_32fc_C3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciRemap_32fc_C3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemapC2_32fc_P3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                         ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                         ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                         DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciRemapC2_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                       DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
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
    MPPErrorCode mppciRemap_32fc_P3RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                       DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[3],
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
    MPPErrorCode mppciRemap_32fc_P3R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr,
                                     size_t aSrc2Step, ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                     size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step, MppiSize aDstSize,
                                     ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageError_32fc64f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciAverageError_32fc64f_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_32fc64fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
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
    MPPErrorCode mppciMSE_32fc64fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumError_32fc64f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciMaximumError_32fc64f_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_32fc64fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
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
    MPPErrorCode mppciSum_32fc64fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_32fc64fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
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
    MPPErrorCode mppciMean_32fc64fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                        DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C3(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C3M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMeanStd_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
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
    MPPErrorCode mppciMeanStd_32fc_C3M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                           MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                              MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel
    /// images:<para/> CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/> CompareOp::Eq |
    /// CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                               DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The comparison is performed for each channel individually and the flag CompareOp::PerChannel
    /// must be set for aCompare.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aSrc2) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEps_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsC_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                                Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsDevC_32fc8u_C3C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                   ConstDevPtrMpp32fc aConst, Mpp32f aEpsilon, DevPtrMpp8u aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, const Mpp32fc aValue[3],
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[3],
                                         MPPCompareOp aCompare, const Mpp32fc aValue[3], DevPtrMpp32fc aDst,
                                         size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, const Mpp32fc aValue[3], DevPtrMpp32fc aDst,
                                            size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), aSrc1
    /// otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             const Mpp32fc aValue[3], DevPtrMpp32fc aDst, size_t aDstStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, const Mpp32fc aValue[3],
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[3],
                                          MPPCompareOp aCompare, const Mpp32fc aValue[3], MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                             MPPCompareOp aCompare, const Mpp32fc aValue[3], MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst fulfills aCompare (for floating point checks, e.g. isinf()), aSrcDst
    /// otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C3I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPCompareOp aCompare,
                                              const Mpp32fc aValue[3], MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                    ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy packed image pixels to planar images.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_C4P4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDstChannel1,
                                     size_t aDstChannel1Step, DevPtrMpp32fc aDstChannel2, size_t aDstChannel2Step,
                                     DevPtrMpp32fc aDstChannel3, size_t aDstChannel3Step, DevPtrMpp32fc aDstChannel4,
                                     size_t aDstChannel4Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy planar image pixels to packed pixel image.
    /// </summary>
    MPPErrorCode mppciCopy_32fc_P4C4(DevPtrMpp32fc aSrcChannel1, size_t aSrcChannel1Step, DevPtrMpp32fc aSrcChannel2,
                                     size_t aSrcChannel2Step, DevPtrMpp32fc aSrcChannel3, size_t aSrcChannel3Step,
                                     DevPtrMpp32fc aSrcChannel4, size_t aSrcChannel4Step, DevPtrMpp32fc aDst,
                                     size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy image with border.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aLowerBorderSize">Size of the border to add on the lower coordinate side
    /// (usually left and top side of the image)</param>
    /// <param name="aBorder">Border control paramter</param>
    /// <param name="aConstant">Constant value needed in case BorderType::Constant</param>
    MPPErrorCode mppciCopyBorder_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
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
    MPPErrorCode mppciCopyBorder_32fc_C4Cb(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, const Mpp32s aLowerBorderSize[2],
                                           const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Copy subpix.
    /// </summary>
    /// <param name="aDst">Destination image</param>
    /// <param name="aDelta">Fractional part of source image coordinate</param>
    /// <param name="aInterpolation">Interpolation mode to use</param>
    MPPErrorCode mppciCopySubpix_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                         size_t aDstStep, const Mpp32f aDelta[2], MPPInterpolationMode aInterpolation,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set all pixels with aMask != 0 to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetC_32fc_C4CI(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, Mpp32fc aConst, Mpp32s aChannel,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Set channel aChannel of all pixels in current ROI to aConst
    /// </summary>
    MPPErrorCode mppciSetDevC_32fc_C4CI(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        Mpp32s aChannel, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (inplace)<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts aSrcDst to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32s aDstChannels[4],
                                           MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Transpose image.
    /// </summary>
    MPPErrorCode mppciTranspose_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                        MppiSize aSizeROISrc, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 + aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAdd_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciAddDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2
    /// </summary>
    MPPErrorCode mppciSub_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 - aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSub_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst -= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInv_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst - aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciSubInvDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2
    /// </summary>
    MPPErrorCode mppciMul_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc2 for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMul_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst *= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciMulDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                  size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                   DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                   size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask,
                                   size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                    DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                    MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 / aConst for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                       DevPtrMpp32fc aDst, size_t aDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aSrc2, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDiv_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                    size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                     ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst /= aConst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc2 / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInv_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                        ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aConst / aSrcDst, for all pixels where aMask != 0
    /// </summary>
    MPPErrorCode mppciDivInvDevC_32fc_C4IM(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C4I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1^2
    /// </summary>
    MPPErrorCode mppciAddSquare_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                         size_t aSrcDstStep, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C4I(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst += aSrc1 * aSrc2
    /// </summary>
    MPPErrorCode mppciAddProduct_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aSrcDst, size_t aSrcDstStep,
                                          ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * alpha + aSrc2 * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                           size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, Mpp32fc aAlpha,
                                           ConstDevPtrMpp8u aMask, size_t aMaskStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C4I(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                           size_t aSrcDstStep, Mpp32fc aAlpha, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrc1 * alpha + aSrcDst * (1 - alpha)
    /// </summary>
    MPPErrorCode mppciAddWeighted_32fc_C4IM(DevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aSrcDst,
                                            size_t aSrcDstStep, Mpp32fc aAlpha, ConstDevPtrMpp8u aMask,
                                            size_t aMaskStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = exp(aSrc1) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = exp(aSrcDst) (exponential function)
    /// </summary>
    MPPErrorCode mppciExp_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = log(aSrc1) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                 MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = log(aSrcDst) (natural logarithm)
    /// </summary>
    MPPErrorCode mppciLn_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                  CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * aSrc1 (aSrc1^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                  MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * aSrcDst (aSrcDst^2)
    /// </summary>
    MPPErrorCode mppciSqr_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                   CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = Sqrt(aSrc1) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = Sqrt(aSrcDst) (square root function)
    /// </summary>
    MPPErrorCode mppciSqrt_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1 * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = aSrcDst * conj(aSrc2) (complex conjugate multiplication)
    /// </summary>
    MPPErrorCode mppciConjMul_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = conj(aSrc1) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                   MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aSrcDst = conj(aSrcDst) (complex conjugate)
    /// </summary>
    MPPErrorCode mppciConj_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MppiSize aSizeROI,
                                    CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1) (complex magnitude)
    /// </summary>
    MPPErrorCode mppciMagnitude_32fc32f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                           size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = abs(aSrc1)^2 (complex magnitude squared)
    /// </summary>
    MPPErrorCode mppciMagnitudeSqr_32fc32f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                              size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = angle(aSrc1) (complex angle, atan2(imag, real))
    /// </summary>
    MPPErrorCode mppciAngle_32fc32f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.real (real component of complex value)
    /// </summary>
    MPPErrorCode mppciReal_32fc32f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst = aSrc1.imag (imaginary component of complex value)
    /// </summary>
    MPPErrorCode mppciImag_32fc32f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                             DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                             MPPFixedFilter aFilter, MPPMaskSize aMaskSize, const Mpp32fc aConstant[4],
                                             MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an mpp::FixedFilter to the source image.
    /// </summary>
    MPPErrorCode mppciFixedFilter_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MPPFixedFilter aFilter, MPPMaskSize aMaskSize, MPPBorderType aBorder,
                                           MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp32fc aConstant[4],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined seperable filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciSeparableFilter_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined column wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciColumnFilter_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, Mpp32f aScalingValue, Mpp32s aFilterSize,
                                                 Mpp32s aFilterCenter, const Mpp32fc aConstant[4],
                                                 MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a column wise box-filter to the image, i.e. the pixels are summed up along columns with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciColumnWindowSum_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                               Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                               MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                           const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined row wise filter to the image. Note that the filter parameters must sum up to 1.
    /// </summary>
    MPPErrorCode mppciRowFilter_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aFilter, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                         MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                              Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                              const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies a row wise box-filter to the image, i.e. the pixels are summed up along rows with the specified
    /// length. The result is then scaled by aScalingValue.
    /// </summary>
    MPPErrorCode mppciRowWindowSum_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            Mpp32f aScalingValue, Mpp32s aFilterSize, Mpp32s aFilterCenter,
                                            MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                           DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                           MppiFilterArea aFilterArea, const Mpp32fc aConstant[4],
                                           MPPBorderType aBorder, MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an averaging box-filter to the image.
    /// </summary>
    MPPErrorCode mppciBoxFilter_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         MppiFilterArea aFilterArea, MPPBorderType aBorder, MppiRect aSrcROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                        ConstDevPtrMpp32f aFilter, MppiFilterArea aFilterArea,
                                        const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Applies an user defined filter, the filter parameters should sum up to 1.<para/>
    /// Note that the filter is applied in "cross-correlation orientation" and not in "convolution orientation", i.e.
    /// the filter has the same orientation as the image (same behavior as in Matlab, mirrored filter as compared to
    /// NPP).
    /// </summary>
    MPPErrorCode mppciFilter_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, ConstDevPtrMpp32f aFilter,
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
    MPPErrorCode mppciWarpAffine_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                            DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                            const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation,
                                            const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciWarpAffine_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffine_32fc_P4RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpAffine_32fc_P4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                          ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                          ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                          ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                          DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                          DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
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
    MPPErrorCode mppciWarpAffineBack_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aAffine[2][3],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpAffineBack_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpAffineBack_32fc_P4RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aAffine[2][3], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpAffineBack_32fc_P4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                              ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                              ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                              ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                              size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step,
                                              DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspective_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                 MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                 MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                 MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpPerspective_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspective_32fc_P4RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpPerspective_32fc_P4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                               ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                               ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                               ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                               DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2,
                                               size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step,
                                               DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                     MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                     MppiSize aDstSize, const Mpp64f aPerspective[3][3],
                                                     MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                   MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_P4RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aPerspective[3][3], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciWarpPerspectiveBack_32fc_P4R(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
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
    MPPErrorCode mppciRotate_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
                                        const Mpp64f aShift[2], MPPInterpolationMode aInterpolation,
                                        const Mpp32fc aConstant[4], MPPBorderType aBorder, MppiRect aSrcROI,
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
    MPPErrorCode mppciRotate_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize, Mpp64f aAngleInDeg,
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
    MPPErrorCode mppciRotate_32fc_P4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                        ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                        ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                        DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
                                        MppiSize aDstSize, Mpp64f aAngleInDeg, const Mpp64f aShift[2],
                                        MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciRotate_32fc_P4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                      ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                      ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                      ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                      DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                      DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
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
    MPPErrorCode mppciResize_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
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
    MPPErrorCode mppciResize_32fc_P4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                     size_t aSrc2Step, ConstDevPtrMpp32fc aSrc3, size_t aSrc3Step,
                                     ConstDevPtrMpp32fc aSrc4, size_t aSrc4Step, DevPtrMpp32fc aDst1, size_t aDst1Step,
                                     DevPtrMpp32fc aDst2, size_t aDst2Step, DevPtrMpp32fc aDst3, size_t aDst3Step,
                                     DevPtrMpp32fc aDst4, size_t aDst4Step, MPPInterpolationMode aInterpolation,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                                MppiSize aSrcFullSize, DevPtrMpp32fc aDst, size_t aDstStep,
                                                MppiSize aDstSize, const Mpp64f aScale[2], const Mpp64f aShift[2],
                                                MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciResizeSqrPixel_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                              DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciResizeSqrPixel_32fc_P4RCb(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aScale[2], const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciResizeSqrPixel_32fc_P4R(
        ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
        ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step, ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step,
        MppiSize aSrcFullSize, DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
        DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step, MppiSize aDstSize,
        const Mpp64f aScale[2], const Mpp64f aShift[2], MPPInterpolationMode aInterpolation, MPPBorderType aBorder,
        MppiRect aSrcROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst, size_t aDstStep,
                                     MPPMirrorAxis aAxis, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Mirror<para/>
    /// Mirror an image along the provided axis (inplace operation)
    /// </summary>
    MPPErrorCode mppciMirror_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPMirrorAxis aAxis, MppiSize aSizeROI,
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
    MPPErrorCode mppciRemapC2_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                         ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciRemapC2_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemap_32fc_C4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
                                       ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciRemap_32fc_C4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, MppiSize aSrcFullSize,
                                     DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aDstSize,
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
    MPPErrorCode mppciRemapC2_32fc_P4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                         ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                         ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                         ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                         DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                         DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
                                         MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMap, size_t aCoordinateMapStep,
                                         MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciRemapC2_32fc_P4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                       ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                       DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
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
    MPPErrorCode mppciRemap_32fc_P4RCb(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step,
                                       ConstDevPtrMpp32fc aSrc2BasePtr, size_t aSrc2Step,
                                       ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                       ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                       DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                       DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
                                       MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                       ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                       MPPInterpolationMode aInterpolation, const Mpp32fc aConstant[4],
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
    MPPErrorCode mppciRemap_32fc_P4R(ConstDevPtrMpp32fc aSrc1BasePtr, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2BasePtr,
                                     size_t aSrc2Step, ConstDevPtrMpp32fc aSrc3BasePtr, size_t aSrc3Step,
                                     ConstDevPtrMpp32fc aSrc4BasePtr, size_t aSrc4Step, MppiSize aSrcFullSize,
                                     DevPtrMpp32fc aDst1, size_t aDst1Step, DevPtrMpp32fc aDst2, size_t aDst2Step,
                                     DevPtrMpp32fc aDst3, size_t aDst3Step, DevPtrMpp32fc aDst4, size_t aDst4Step,
                                     MppiSize aDstSize, ConstDevPtrMpp32f aCoordinateMapX, size_t aCoordinateMapXStep,
                                     ConstDevPtrMpp32f aCoordinateMapY, size_t aCoordinateMapYStep,
                                     MPPInterpolationMode aInterpolation, MPPBorderType aBorder, MppiRect aSrcROI,
                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageErrorBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageError_32fc64f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciAverageError_32fc64f_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for AverageRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciAverageRelativeErrorBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciAverageRelativeError_32fc64f_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProduct.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for DotProductMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciDotProductBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                             size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
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
    MPPErrorCode mppciDotProduct_32fc64fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                              size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
                                              ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                              size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSE.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MSEMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMSEBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMSE_32fc64fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                      size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
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
    MPPErrorCode mppciMSE_32fc64fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                       size_t aSrc2Step, DevPtrMpp64fc aDst, DevPtrMpp64fc aDstScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                     CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumErrorBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumError_32fc64f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
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
    MPPErrorCode mppciMaximumError_32fc64f_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, DevPtrMpp64f aDst, DevPtrMpp64f aDstScalar,
                                               ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                               size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeError.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI,
                                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MaximumRelativeErrorMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMaximumRelativeErrorBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                      ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
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
    MPPErrorCode mppciMaximumRelativeError_32fc64f_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                       ConstDevPtrMpp32fc aSrc2, size_t aSrc2Step, DevPtrMpp64f aDst,
                                                       DevPtrMpp64f aDstScalar, ConstDevPtrMpp8u aMask,
                                                       size_t aMaskStep, DevPtrMpp8u aBuffer, size_t aBufferSize,
                                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Sum.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for SumMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    /// <param name="aDst">Used as output type indicator</param>
    MPPErrorCode mppciSumBufferSize_32fc64f_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the sum of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciSum_32fc64fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
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
    MPPErrorCode mppciSum_32fc64fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                       DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                       DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for Mean.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Computes the mean of pixel values.<para/>For multi-channel images, the
    /// result is computed for each channel seperatly in aDst, or for all channels in aDstScalar.
    /// </summary>
    /// <param name="aDst">Per-channel result, can be nullptr</param>
    /// <param name="aDstScalar">Result for all channels, can be nullptr</param>
    /// <param name="aBuffer">Temporary device memory buffer for computation.</param>
    /// <param name="aStreamCtx"></param>
    MPPErrorCode mppciMean_32fc64fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
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
    MPPErrorCode mppciMean_32fc64fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aDst,
                                        DevPtrMpp64fc aDstScalar, ConstDevPtrMpp8u aMask, size_t aMaskStep,
                                        DevPtrMpp8u aBuffer, size_t aBufferSize, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStd.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C4(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Returns the required temporary buffer size for MeanStdMasked.<para/>
    /// Note: the buffer size differs for varying ROI sizes.
    /// </summary>
    MPPErrorCode mppciMeanStdBufferSize_32fc_C4M(size_t *aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

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
    MPPErrorCode mppciMeanStd_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
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
    MPPErrorCode mppciMeanStd_32fc_C4M(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp64fc aMean,
                                       DevPtrMpp64f aStd, DevPtrMpp64fc aMeanScalar, DevPtrMpp64f aStdScalar,
                                       ConstDevPtrMpp8u aMask, size_t aMaskStep, DevPtrMpp8u aBuffer,
                                       size_t aBufferSize, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                          size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                           MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                              MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                              MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel
    /// images:<para/> CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/> CompareOp::Eq |
    /// CompareOp::AnyChannel is true if any of the channels in a pixel is equal.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                               DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aSrc2 fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompare_32fc8u_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareC_32fc8u_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                         MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 and aConst fulfill aCompare, 0 otherwise.<para/>
    /// The comparison is performed for each channel individually and the flag CompareOp::PerChannel must be set for
    /// aCompare.
    /// </summary>
    MPPErrorCode mppciCompareDevC_32fc8u_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), 0
    /// otherwise.<para/> The comparison is performed for each channel individually and the flag CompareOp::PerChannel
    /// must be set for aCompare.
    /// </summary>
    MPPErrorCode mppciCompareFloat_32fc8u_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aSrc2) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEps_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                               size_t aSrc2Step, Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep,
                                               MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsC_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                                Mpp32f aEpsilon, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                                CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to 255 if all color channels for abs(aSrc1 - aConst) are &lt;= aEpsilon, 0 otherwise.
    /// </summary>
    MPPErrorCode mppciCompareEqEpsDevC_32fc8u_C4C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step,
                                                   ConstDevPtrMpp32fc aConst, Mpp32f aEpsilon, DevPtrMpp8u aDst,
                                                   size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aSrc2 fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aSrc2,
                                        size_t aSrc2Step, MPPCompareOp aCompare, const Mpp32fc aValue[4],
                                        DevPtrMpp32fc aDst, size_t aDstStep, MppiSize aSizeROI,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, const Mpp32fc aConst[4],
                                         MPPCompareOp aCompare, const Mpp32fc aValue[4], DevPtrMpp32fc aDst,
                                         size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 and aConst fulfill aCompare, aSrc1 otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, ConstDevPtrMpp32fc aConst,
                                            MPPCompareOp aCompare, const Mpp32fc aValue[4], DevPtrMpp32fc aDst,
                                            size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// aDst pixel is set to aValue if aSrc1 fulfills aCompare (for floating point checks, e.g. isinf()), aSrc1
    /// otherwise.<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, MPPCompareOp aCompare,
                                             const Mpp32fc aValue[4], DevPtrMpp32fc aDst, size_t aDstStep,
                                             MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aSrc2 fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIf_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aSrc2,
                                         size_t aSrc2Step, MPPCompareOp aCompare, const Mpp32fc aValue[4],
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, const Mpp32fc aConst[4],
                                          MPPCompareOp aCompare, const Mpp32fc aValue[4], MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst and aConst fulfill aCompare, aSrcDst otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfDevC_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, ConstDevPtrMpp32fc aConst,
                                             MPPCompareOp aCompare, const Mpp32fc aValue[4], MppiSize aSizeROI,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// A pixel is set to aValue if aSrcDst fulfills aCompare (for floating point checks, e.g. isinf()), aSrcDst
    /// otherwise (inplace operation).<para/>
    /// The flag CompareOp::AnyChannel controls how the comparison is performed for multi channel images:<para/>
    /// CompareOp::Eq is true only if all channels in a pixel are equal whereas <para/>
    /// CompareOp::Eq | CompareOp::AnyChannel is true if any of the channels in a pixel is equal.<para/>
    /// Without the CompareOp::PerChannel flag, a pixel is compared for all channels and replaced by all channels. With
    /// the CompareOp::PerChannel flag, each channel is compared and replaced seperately.
    /// </summary>
    MPPErrorCode mppciReplaceIfFloat_32fc_C4I(DevPtrMpp32fc aSrcDst, size_t aSrcDstStep, MPPCompareOp aCompare,
                                              const Mpp32fc aValue[4], MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_32FC_H
