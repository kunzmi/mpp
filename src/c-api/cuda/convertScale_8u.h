#ifndef MPPI_CUDA_CAPI_CS_8U_H
#define MPPI_CUDA_CAPI_CS_8U_H

#include "mppc_capi_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // 1 channel (C1)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16bf_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u64f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C1Sfs(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                         MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u8s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                    DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u8s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                        Mpp32s aDstMin, Mpp32s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u8s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                            MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u8s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                         DevPtrMpp8s aDst, size_t aDstStep, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32s_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32u_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16bf_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u64f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16bf_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u64f_C1(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    // 2 channels  (C2)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16bf_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u64f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C2Sfs(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                         MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u8s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                    DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u8s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                        Mpp32s aDstMin, Mpp32s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u8s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                            MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u8s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                         DevPtrMpp8s aDst, size_t aDstStep, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32s_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32u_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16bf_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u64f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16bf_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u64f_C2(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    // 3 channels  (C3)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16bf_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u64f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C3Sfs(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                         MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u8s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                    DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u8s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                        Mpp32s aDstMin, Mpp32s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u8s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                            MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u8s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                         DevPtrMpp8s aDst, size_t aDstStep, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32s_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32u_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16bf_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u64f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16bf_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u64f_C3(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    // 4 channels (C4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16bf_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u64f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_C4Sfs(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                         MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u8s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                    DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                    MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                     MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u8s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                       CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                        Mpp32s aDstMin, Mpp32s aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                        MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u8s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                            MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                            CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u8s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                         DevPtrMpp8s aDst, size_t aDstStep, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32s_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32u_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16bf_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u64f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                     MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16bf_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u64f_C4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    // 4 channels with alpha  (AC4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u16bf_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u32f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_8u64f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_8u8s_AC4Sfs(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u8s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                     DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u8s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                        Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                         Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                         Mpp16u aDstMin, Mpp16u aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst, size_t aDstStep,
                                         Mpp32s aDstMin, Mpp32s aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                         Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u8s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u16u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_8u32u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u8s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                          DevPtrMpp8s aDst, size_t aDstStep, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                           DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u16u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                           DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32s_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                           DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_8u32u_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                           DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u16bf_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                       DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u32f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_8u64f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, Mpp8u aSrcMin, Mpp8u aSrcMax,
                                      DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                         Mpp16f aDstMin, Mpp16f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u16bf_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                          Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u32f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                         Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_8u64f_AC4(ConstDevPtrMpp8u aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_CSCD_8U_H
