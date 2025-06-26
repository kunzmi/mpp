#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "cudaException.h"
#include "streamCtx.h"
#include <backends/cuda/stream.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace mpp::cuda
{
void StreamCtxSingleton::UpdateContext()
{
    cudaSafeCall(cudaGetDevice(&tlSingletonCtx.DeviceId));

    cudaDeviceProp props{};
    cudaSafeCall(cudaGetDeviceProperties(&props, tlSingletonCtx.DeviceId));

    tlSingletonCtx.MultiProcessorCount         = props.multiProcessorCount;
    tlSingletonCtx.MaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    tlSingletonCtx.MaxThreadsPerBlock          = props.maxThreadsPerBlock;
    tlSingletonCtx.SharedMemPerBlock           = props.sharedMemPerBlock;
    tlSingletonCtx.WarpSize                    = props.warpSize;

    cudaSafeCall(cudaDeviceGetAttribute(&tlSingletonCtx.ComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
                                        tlSingletonCtx.DeviceId));

    cudaSafeCall(cudaDeviceGetAttribute(&tlSingletonCtx.ComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
                                        tlSingletonCtx.DeviceId));

    // set to NULL default stream:
    tlSingletonCtx.Stream = Stream::Null.Original();
    cudaSafeCall(cudaStreamGetFlags(tlSingletonCtx.Stream, &tlSingletonCtx.StreamFlags));
}
const StreamCtx &StreamCtxSingleton::Get()
{
    // invalid device ID indicates first call to Get() --> fill with data from api calls:
    if (tlSingletonCtx.DeviceId < 0)
    {
        UpdateContext();
    }
    return tlSingletonCtx;
}

void StreamCtxSingleton::SetStream(const Stream &aStream)
{
    tlSingletonCtx.Stream = aStream.Original();
    cudaSafeCall(cudaStreamGetFlags(tlSingletonCtx.Stream, &tlSingletonCtx.StreamFlags));
}

thread_local StreamCtx StreamCtxSingleton::tlSingletonCtx = StreamCtx();

} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_BACKEND