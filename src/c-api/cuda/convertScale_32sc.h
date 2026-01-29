#ifndef MPPI_CUDA_CAPI_CS_32SC_H
#define MPPI_CUDA_CAPI_CS_32SC_H

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
    MPPErrorCode mppciConvert_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc32fc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C1Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
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
    MPPErrorCode mppciScaleToFVR_32sc16sc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc32fc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc32fc_C1(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    // 2 channels  (C2)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc32fc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C2Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
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
    MPPErrorCode mppciScaleToFVR_32sc16sc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc32fc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc32fc_C2(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    // 3 channels  (C3)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc32fc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C3Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
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
    MPPErrorCode mppciScaleToFVR_32sc16sc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc32fc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc32fc_C3(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

    // 4 channels (C4)

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Integer to Integer, Integer to float32/double, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, float to half or bfloat are rounded using
    /// RoundingMode::NearestTiesToEven (real and complex).
    /// </summary>
    MPPErrorCode mppciConvert_32sc32fc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                          size_t aDstStep, MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32sc16sc_C4Sfs(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                           size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax, MppiSize aSizeROI,
                                           MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to ouput type value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).<para/>
    /// For source and destination, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVRToVFR_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
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
    MPPErrorCode mppciScaleToFVR_32sc16sc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32sc32fc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, Mpp32s aSrcMin, Mpp32s aSrcMax,
                                        DevPtrMpp32fc aDst, size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax,
                                        MppiSize aSizeROI, CPtrMppStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.<para/>
    /// For the source type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleFVR_32sc32fc_C4(ConstDevPtrMpp32sc aSrc1, size_t aSrc1Step, DevPtrMpp32fc aDst,
                                           size_t aDstStep, Mpp32f aDstMin, Mpp32f aDstMax, MppiSize aSizeROI,
                                           CPtrMppStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_CSCD_32SC_H
