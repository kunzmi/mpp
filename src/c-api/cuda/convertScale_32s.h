#ifndef MPPI_CUDA_CAPI_CS_32S_H
#define MPPI_CUDA_CAPI_CS_32S_H

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
    MPPErrorCode mppciConvert_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16bf_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s64f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C1Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                        Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                         Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
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
    MPPErrorCode mppciScaleToFVR_32s8s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s8u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16s_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s32u_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                           DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16bf_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s64f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s16f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16bf_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                          size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s32f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s64f_C1(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    // 2 channels  (C2)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16bf_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s64f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C2Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                        Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                         Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
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
    MPPErrorCode mppciScaleToFVR_32s8s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s8u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16s_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s32u_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                           DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16bf_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s64f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s16f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16bf_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                          size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s32f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s64f_C2(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    // 3 channels  (C3)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16bf_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s64f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C3Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                        Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                         Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
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
    MPPErrorCode mppciScaleToFVR_32s8s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s8u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16s_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s32u_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                           DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16bf_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s64f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s16f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16bf_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                          size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s32f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s64f_C3(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    // 4 channels (C4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16bf_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s64f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                          MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_C4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                           size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                           int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                     DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI,
                                     MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                        Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                         Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                         MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
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
    MPPErrorCode mppciScaleToFVR_32s8s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s8u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16s_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s32u_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                           DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16bf_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                      MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s64f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s16f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16bf_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                          size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s32f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s64f_C4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                         CPtrMppStreamCtx aStreamCtx);

    // 4 channels with alpha  (AC4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s16bf_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s32f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32s64f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                         MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8s_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
                                           MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s8u_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                           MppiSize aSizeROI, MPPRoundingMode aRoundingMode, int aScaleFactor,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16s_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                            size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                            int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32s16u_AC4Sfs(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                            size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                            int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp8s aDst, size_t aDstStep, Mpp8s aDstMin, Mpp8s aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                      DevPtrMpp8u aDst, size_t aDstStep, Mpp8u aDstMin, Mpp8u aDstMax,
                                      MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16s aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16u aDst, size_t aDstStep, Mpp16u aDstMin, Mpp16u aDstMax,
                                       MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst, size_t aDstStep,
                                         Mpp8u aDstMin, Mpp8u aDstMax, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                         CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst, size_t aDstStep,
                                          Mpp32u aDstMin, Mpp32u aDstMax, MppiSize aSizeROI,
                                          MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp8u aDst,
                                              size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                              CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16s aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16u aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32u aDst,
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
    MPPErrorCode mppciScaleToFVR_32s8s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s8u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16s_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s16u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleToFVR_32s32u_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                            DevPtrMpp32u aDst, size_t aDstStep, MppiSize aSizeROI,
                                            MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp16f aDst, size_t aDstStep, Mpp16f aDstMin, Mpp16f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s16bf_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp16bf aDst, size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s32f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                       DevPtrMpp32f aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                       MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32s64f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
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
    MPPErrorCode mppciScaleFVR_32s16f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s16bf_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp16bf aDst,
                                           size_t aDstStep, Mpp16bf aDstMin, Mpp16bf aDstMax, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32s32f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp32f aDst, size_t aDstStep,
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
    MPPErrorCode mppciScaleFVR_32s64f_AC4(ConstDevPtrMpp32s aSrc1, size_t aSrc1Step, DevPtrMpp64f aDst, size_t aDstStep,
                                          Mpp64f aDstMin, Mpp64f aDstMax, MppiSize aSizeROI,
                                          CPtrMppStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_CSCD_32S_H
