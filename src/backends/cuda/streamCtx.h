#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/stream.h>
#include <common/defines.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace mpp::cuda
{

struct StreamCtx
{
    /// <summary>
    /// The cuda stream to use for kernel execution
    /// </summary>
    cudaStream_t Stream{nullptr};

    /// <summary>
    /// From cudaGetDevice()
    /// </summary>
    int DeviceId{-1};
    // -1 is an invalid device ID and is used as indicator that the StreamCtx has not been populated correctly

    /// <summary>
    /// From cudaGetDeviceProperties()
    /// </summary>
    int MultiProcessorCount{0};

    /// <summary>
    /// From cudaGetDeviceProperties()
    /// </summary>
    int MaxThreadsPerMultiProcessor{0};

    /// <summary>
    /// From cudaGetDeviceProperties()
    /// </summary>
    int MaxThreadsPerBlock{0};

    /// <summary>
    /// From cudaGetDeviceProperties
    /// </summary>
    std::size_t SharedMemPerBlock{0};

    /// <summary>
    /// From cudaGetDeviceAttribute()
    /// </summary>
    int ComputeCapabilityMajor{0};

    /// <summary>
    /// From cudaGetDeviceAttribute()
    /// </summary>
    int ComputeCapabilityMinor{0};

    /// <summary>
    /// From cudaStreamGetFlags()
    /// </summary>
    uint StreamFlags{0};

    /// <summary>
    ///
    /// </summary>
    int WarpSize{0};
};

class StreamCtxSingleton
{
  private:
    static thread_local StreamCtx tlSingletonCtx;

  public:
    /// <summary>
    /// Gets the StreamCtx that is associated with the current CPU thread, i.e. with the current cuda context.
    /// If a new cuda context gets bound to the CPU thread, this StreamCtx gets invalid and must be updated!
    /// </summary>
    /// <returns></returns>
    [[nodiscard]] static const StreamCtx &Get();

    /// <summary>
    /// Reinitializes the stream context to the new current Cuda context
    /// </summary>
    /// <returns></returns>
    static void UpdateContext();

    /// <summary>
    /// Sets the given cuda stream as the new default stream to the stream context
    /// </summary>
    /// <returns></returns>
    static void SetStream(const Stream &aStream);
};
} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_BACKEND