#pragma once
#include "cudaException.h"
#include <common/defines.h>
#include <cuda_runtime_api.h>

namespace opp::cuda
{
// forward declaration:
class Event;

/// <summary>
/// A wrapper class for a cudaStream_t.
/// </summary>
class Stream
{
  private:
    cudaStream_t mStream{nullptr};
    bool mIsOwner{false};

  public:
    /// <summary>
    /// Creates a CUstream with flags CU_STREAM_DEFAULT
    /// </summary>
    Stream();
    explicit Stream(cudaStream_t aStream);
    explicit Stream(uint aFlags);

    ~Stream();

    Stream(const Stream &aOther);
    Stream(Stream &&aOther) noexcept;

    Stream &operator=(const Stream &aOther);
    Stream &operator=(Stream &&aOther) noexcept;

    /// <summary>
    /// The wrapped cudaStream_t
    /// </summary>
    [[nodiscard]] cudaStream_t Original() const
    {
        return mStream;
    }

    /// <summary>
    /// Waits until the device has completed all operations in the stream. If the context was created
    /// with the <see cref="CUCtxFlags.BlockingSync"/> flag, the CPU thread will block until the stream is finished with
    /// all of its tasks.
    /// </summary>
    void Synchronize() const;

    /// <summary>
    /// Returns true if all operations in the stream have completed, or
    /// false if not.
    /// </summary>
    [[nodiscard]] bool Query() const;

    /// <summary>
    /// Make a compute stream wait on an event<para/>
    /// Makes all future work submitted to the Stream wait until <c>hEvent</c>
    /// reports completion before beginning execution. This synchronization
    /// will be performed efficiently on the device.
    /// <para/>
    /// The stream will wait only for the completion of the most recent
    /// host call to <see cref="CudaEvent.Record()"/> on <c>hEvent</c>. Once this call has returned,
    /// any functions (including <see cref="CudaEvent.Record()"/> and <see cref="Dispose()"/> may be
    /// called on <c>hEvent</c> again, and the subsequent calls will not have any
    /// effect on this stream.
    /// <para/>
    /// If <c>hStream</c> is 0 (the NULL stream) any future work submitted in any stream
    /// will wait for <c>hEvent</c> to complete before beginning execution. This
    /// effectively creates a barrier for all future work submitted to the context.
    /// <para/>
    /// If <see cref="CudaEvent.Record()"/> has not been called on <c>hEvent</c>, this call acts as if
    /// the record has already completed, and so is a functional no-op.
    /// </summary>
    void WaitEvent(cudaEvent_t aCUevent, uint aFlags = 0) const;

    /// <summary>
    /// Make a compute stream wait on an event<para/>
    /// Makes all future work submitted to the Stream wait until <c>hEvent</c>
    /// reports completion before beginning execution. This synchronization
    /// will be performed efficiently on the device.
    /// <para/>
    /// The stream will wait only for the completion of the most recent
    /// host call to <see cref="CudaEvent.Record()"/> on <c>hEvent</c>. Once this call has returned,
    /// any functions (including <see cref="CudaEvent.Record()"/> and <see cref="Dispose()"/> may be
    /// called on <c>hEvent</c> again, and the subsequent calls will not have any
    /// effect on this stream.
    /// <para/>
    /// If <c>hStream</c> is 0 (the NULL stream) any future work submitted in any stream
    /// will wait for <c>hEvent</c> to complete before beginning execution. This
    /// effectively creates a barrier for all future work submitted to the context.
    /// <para/>
    /// If <see cref="CudaEvent.Record()"/> has not been called on <c>hEvent</c>, this call acts as if
    /// the record has already completed, and so is a functional no-op.
    /// </summary>
    void WaitEvent(const Event &aEvent, uint aFlags = 0) const;

    /// <summary>
    /// Adds a callback to be called on the host after all currently enqueued
    /// items in the stream have completed.  For each
    /// cuStreamAddCallback call, the callback will be executed exactly once.
    /// The callback will block later work in the stream until it is finished.
    /// <para/>
    /// The callback may be passed <see cref="CUResult.Success"/> or an error code.  In the event
    /// of a device error, all subsequently executed callbacks will receive an
    /// appropriate <see cref="CUResult"/>.
    /// <para/>
    /// Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
    /// will result in <see cref="CUResult.ErrorNotPermitted"/>.  Callbacks must not perform any
    /// synchronization that may depend on outstanding device work or other callbacks
    /// that are not mandated to run earlier.  Callbacks without a mandated order
    /// (in independent streams) execute in undefined order and may be serialized.
    /// <para/>
    /// This API requires compute capability 1.1 or greater.  See
    /// cuDeviceGetAttribute or ::cuDeviceGetProperties to query compute
    /// capability.  Attempting to use this API with earlier compute versions will
    /// return <see cref="CUResult.ErrorNotSupported"/>.
    /// </summary>
    void AddCallback(cudaStreamCallback_t aCallback, void *aUserData, uint aFlags);

    /// <summary>
    /// Wrapper for the CUDA NULL-Stream
    /// </summary>
    static const Stream Null;

    /// <summary>
    /// Wrapper for the CUDA Legacy-Stream (CU_STREAM_LEGACY)
    /// </summary>
    static const Stream Legacy;

    /// <summary>
    /// Wrapper for the CUDA PerThread-Stream (CU_STREAM_PER_THREAD)
    /// </summary>
    static const Stream PerThread;

  private:
};
} // namespace opp::cuda