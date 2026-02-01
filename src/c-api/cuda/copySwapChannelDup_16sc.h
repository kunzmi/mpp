#ifndef MPPI_CUDA_CAPI_CSCD_16SC_H
#define MPPI_CUDA_CAPI_CSCD_16SC_H

#include "mppc_capi_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif
    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    MPPErrorCode mppciDup_16sc_C1C2(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    MPPErrorCode mppciDup_16sc_C1C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Duplicates a one channel image to all channels in a multi-channel image
    /// </summary>
    MPPErrorCode mppciDup_16sc_C1C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                    MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy single channel image to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C1C2C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                      Mpp32s aDstChannel, MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy single channel image to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C1C3C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                      Mpp32s aDstChannel, MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy single channel image to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C1C4C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst, size_t aDstStep,
                                      Mpp32s aDstChannel, MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2C1C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel, DevPtrMpp16sc aDst,
                                    size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                    CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2C3C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C2C4C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3C1C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3C2C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel, DevPtrMpp16sc aDst,
                                    size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                    CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C3C4C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts this to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, const Mpp32s aDstChannels[3], MppiSize aSizeROI,
                                          CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels (3-channel to 4-channel with additional value).<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts this to BGR channel order.<para/>
    /// If aDstChannels[i] == 3, channel i of aDst is set to aValue, if aDstChannels[i] > 3, channel i of aDst is kept
    /// unchanged.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C3C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                            size_t aDstStep, const Mpp32s aDstChannels[3], Mpp16sc aValue,
                                            MppiSize aSizeROI, CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to single channel image aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4C1C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4C2C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4C3C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel,
                                      DevPtrMpp16sc aDst, size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                      CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Copy channel aSrcChannel to channel aDstChannel of aDst.
    /// </summary>
    MPPErrorCode mppciCopy_16sc_C4C(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, Mpp32s aSrcChannel, DevPtrMpp16sc aDst,
                                    size_t aDstStep, Mpp32s aDstChannel, MppiSize aSizeROI,
                                    CPtrMppCudaStreamCtx aStreamCtx);
    /// <summary>
    /// Swap channels<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts this to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C4(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                          size_t aDstStep, const Mpp32s aDstChannels[4], MppiSize aSizeROI,
                                          CPtrMppCudaStreamCtx aStreamCtx);

    /// <summary>
    /// Swap channels<para/>
    /// aDstChannels describes how channel values are permutated. The n-th entry
    /// of the array contains the number of the channel that is stored in the n-th channel of
    /// the output image. <para/>
    /// E.g. Given an RGB image, aDstChannels = [2,1,0] converts this to BGR channel order.
    /// </summary>
    MPPErrorCode mppciSwapChannel_16sc_C4C3(ConstDevPtrMpp16sc aSrc1, size_t aSrc1Step, DevPtrMpp16sc aDst,
                                            size_t aDstStep, const Mpp32s aDstChannels[3], MppiSize aSizeROI,
                                            CPtrMppCudaStreamCtx aStreamCtx);

#ifdef __cplusplus
}
#endif
#endif // MPPI_CUDA_CAPI_CSCD_16SC_H
