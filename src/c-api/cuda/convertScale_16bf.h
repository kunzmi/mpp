#ifndef MPPI_CUDA_CAPI_CS_16BF_H
#define MPPI_CUDA_CAPI_CS_16BF_H

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
    MPPErrorCode mppciConvert_16bf32f_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf64f_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16f_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32f_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf64f_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf8s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_16bf8u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                           DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32s_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32u_C1(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    // 2 channels  (C2)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf32f_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf64f_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16f_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32f_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf64f_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf8s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_16bf8u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                           DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32s_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32u_C2(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    // 3 channels  (C3)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf32f_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf64f_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16f_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32f_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf64f_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf8s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_16bf8u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                           DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32s_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32u_C3(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    // 4 channels (C4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf32f_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf64f_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16f_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32f_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf64f_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                      DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf8s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_16bf8u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                           DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32s_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32u_C4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    // 4 channels with alpha  (AC4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf32f_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_16bf64f_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf8u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf16u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32s aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_16bf32u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16f_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32f_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf64f_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp64f aDst, size_t aDstStep, Mpp64f aDstMin, Mpp64f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf8u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                       DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf16u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp32s aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_16bf32u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin, Mpp16bf aSrcMax,
                                        DevPtrMpp32u aDst, size_t aDstStep, Mpp32u aDstMin, Mpp32u aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf8s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp8s aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf8u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                            Mpp16bf aSrcMax, DevPtrMpp8u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                             Mpp16bf aSrcMax, DevPtrMpp16s aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf16u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                             Mpp16bf aSrcMax, DevPtrMpp16u aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32s_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                             Mpp16bf aSrcMax, DevPtrMpp32s aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_16bf32u_AC4(ConstDevPtrMpp16bf aSrc1, size_t aSrc1Step, Mpp16bf aSrcMin,
                                             Mpp16bf aSrcMax, DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_CSCD_16BF_H
