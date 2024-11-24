#include "cudaException.h"
#include "streamCtx.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace opp::cuda
{

StreamCtx &StreamCtxSingleton::Get()
{
    // invalid device ID indicates first call to Get() --> fill with data from api calls:
    if (tlSingletonCtx.DeviceId < 0)
    {
        // mSingletonCtx.Stream = nullptr;
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

        cudaSafeCall(cudaStreamGetFlags(tlSingletonCtx.Stream, &tlSingletonCtx.StreamFlags));
    }

    return tlSingletonCtx;
}

thread_local StreamCtx StreamCtxSingleton::tlSingletonCtx = StreamCtx();

} // namespace opp::cuda