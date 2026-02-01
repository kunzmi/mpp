#ifndef MPPI_CUDA_CAPI_CS_32FC_H
#define MPPI_CUDA_CAPI_CS_32FC_H

#include "mppc_capi_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // 1 channel (C1)

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc16sc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc32sc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc16sc_C1Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc32sc_C1Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc16sc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc32sc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp32sc aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc16sc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc32sc_C1(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp32sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    // 2 channels  (C2)

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc16sc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc32sc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp8s aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc16sc_C2Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc32sc_C2Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc16sc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc32sc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp32sc aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc16sc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc32sc_C2(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp32sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    // 3 channels  (C3)

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc16sc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc32sc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc16sc_C3Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc32sc_C3Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc16sc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc32sc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp32sc aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc16sc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc32sc_C3(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp32sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    // 4 channels (C4)

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc16sc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert Floating point to Integer, float32 to half-float16/bfloat16. Values are clamped to
    /// maximum value range of destination type if needed, using rounding mode as provided.
    /// Note: For float32 to half16/bfloat: RoundingMode::NearestTiesAwayFromZero is NOT supported.
    /// </summary>
    MPPErrorCode mppciConvertRound_32fc32sc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                               size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                               CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc16sc_C4Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with prior floating point scaling. Operation is SrcT -&gt; float -&gt; scale -&gt; DstT
    /// </summary>
    MPPErrorCode mppciConvert_32fc32sc_C4Sfs(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, DevPtrMpp32sc aDst,
                                             size_t aDstStep, MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                             int aScaleFactor, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc16sc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp16sc aDst, size_t aDstStep, Mpp16s aDstMin, Mpp16s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary for integer output types.
    /// </summary>
    MPPErrorCode mppciScale_32fc32sc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                        DevPtrMpp32sc aDst, size_t aDstStep, Mpp32s aDstMin, Mpp32s aDstMax,
                                        MppiSize aSizeROI, MPPRoundingMode aRoundingMode,
                                        CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc16sc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Convert with scaling the pixel value from input type value range to provided ouput value range using the
    /// equation:<para/>
    /// dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)<para/>
    /// whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue). Values
    /// smaller or larger the output range are clamped to min or max value if necessary.<para/>
    /// For the destination type, the min/max values are implicitly given by the type's min/max values.
    /// </summary>
    MPPErrorCode mppciScaleToFVR_32fc32sc_C4(ConstDevPtrMpp32fc aSrc1, size_t aSrc1Step, Mpp32f aSrcMin, Mpp32f aSrcMax,
                                             DevPtrMpp32sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                             MPPRoundingMode aRoundingMode, CPtrMppCudaStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_CSCD_32FC_H
