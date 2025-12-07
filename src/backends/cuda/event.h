#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_CUDA_CORE

#include "cudaException.h"
#include "dllexport_cudacore.h"
#include <common/defines.h>
#include <cuda_runtime_api.h>

namespace mpp::cuda
{
// forward declaration:
class MPPEXPORT_CUDACORE Stream;

/// <summary>
/// A wrapper class for a cudaEvent_t.
/// </summary>
class MPPEXPORT_CUDACORE Event
{
  private:
    cudaEvent_t mEvent{nullptr};
    bool mIsOwner{false};

  public:
    /// <summary>
    /// Creates a CUstream with flags CU_STREAM_DEFAULT
    /// </summary>
    Event();
    explicit Event(cudaEvent_t aEvent);
    explicit Event(uint aFlags);

    ~Event();

    Event(const Event &aOther);
    Event(Event &&aOther) noexcept;

    Event &operator=(const Event &aOther);
    Event &operator=(Event &&aOther) noexcept;

    /// <summary>
    /// The wrapped cudaEvent_t
    /// </summary>
    [[nodiscard]] cudaEvent_t Original() const
    {
        return mEvent;
    }

    /// <summary>
    /// Records an event<para/>
    /// Captures in mEvent the contents of null-stream at the time of this call.<para/>
    /// Calls such as Event::Query() or Stream::WaitEvent() will then
    /// examine or wait for completion of the work that was captured. Uses of
    /// null-stream after this call do not modify mEvent. See note on default
    /// stream behavior for what is captured in the default case.<para/>
    /// Record() can be called multiple times on the same event and
    /// will overwrite the previously captured state. Other APIs such as
    /// Stream::WaitEvent() use the most recently captured state at the time
    /// of the API call, and are not affected by later calls to
    /// Record(). Before the first call to Record(), an
    /// event represents an empty set of work, so for example Query() would return cudaSuccess.
    /// </summary>
    void Record() const;

    /// <summary>
    /// Records an event<para/>
    /// Captures in mEvent the contents of aStream at the time of this call.<para/>
    /// Event and aStream must be on the same CUDA context.<para/>
    /// Calls such as Event::Query() or Stream::WaitEvent() will then
    /// examine or wait for completion of the work that was captured. Uses of
    /// null-stream after this call do not modify mEvent. See note on default
    /// stream behavior for what is captured in the default case.<para/>
    /// Record() can be called multiple times on the same event and
    /// will overwrite the previously captured state. Other APIs such as
    /// Stream::WaitEvent() use the most recently captured state at the time
    /// of the API call, and are not affected by later calls to
    /// Record(). Before the first call to Record(), an
    /// event represents an empty set of work, so for example Query() would return cudaSuccess.
    /// </summary>
    void Record(cudaStream_t aStream) const;

    /// <summary>
    /// Records an event<para/>
    /// Captures in mEvent the contents of aStream at the time of this call.<para/>
    /// Event and aStream must be on the same CUDA context.<para/>
    /// Calls such as Event::Query() or Stream::WaitEvent() will then
    /// examine or wait for completion of the work that was captured. Uses of
    /// null-stream after this call do not modify mEvent. See note on default
    /// stream behavior for what is captured in the default case.<para/>
    /// Record() can be called multiple times on the same event and
    /// will overwrite the previously captured state. Other APIs such as
    /// Stream::WaitEvent() use the most recently captured state at the time
    /// of the API call, and are not affected by later calls to
    /// Record(). Before the first call to Record(), an
    /// event represents an empty set of work, so for example Query() would return cudaSuccess.
    /// </summary>
    void Record(const Stream &aStream) const;

    /// <summary>
    /// Records an event<para/>
    /// Captures in mEvent the contents of null-stream at the time of this call.<para/>
    /// Calls such as Event::Query() or Stream::WaitEvent() will then
    /// examine or wait for completion of the work that was captured. Uses of
    /// null-stream after this call do not modify mEvent. See note on default
    /// stream behavior for what is captured in the default case.<para/>
    /// Record() can be called multiple times on the same event and
    /// will overwrite the previously captured state. Other APIs such as
    /// Stream::WaitEvent() use the most recently captured state at the time
    /// of the API call, and are not affected by later calls to
    /// Record(). Before the first call to Record(), an
    /// event represents an empty set of work, so for example Query() would return cudaSuccess.
    /// </summary>
    void Record(uint aFlags) const;

    /// <summary>
    /// Records an event<para/>
    /// Captures in mEvent the contents of aStream at the time of this call.<para/>
    /// Event and aStream must be on the same CUDA context.<para/>
    /// Calls such as Event::Query() or Stream::WaitEvent() will then
    /// examine or wait for completion of the work that was captured. Uses of
    /// null-stream after this call do not modify mEvent. See note on default
    /// stream behavior for what is captured in the default case.<para/>
    /// Record() can be called multiple times on the same event and
    /// will overwrite the previously captured state. Other APIs such as
    /// Stream::WaitEvent() use the most recently captured state at the time
    /// of the API call, and are not affected by later calls to
    /// Record(). Before the first call to Record(), an
    /// event represents an empty set of work, so for example Query() would return cudaSuccess.
    /// </summary>
    void Record(cudaStream_t aStream, uint aFlags) const;

    /// <summary>
    /// Records an event<para/>
    /// Captures in mEvent the contents of aStream at the time of this call.<para/>
    /// Event and aStream must be on the same CUDA context.<para/>
    /// Calls such as Event::Query() or Stream::WaitEvent() will then
    /// examine or wait for completion of the work that was captured. Uses of
    /// null-stream after this call do not modify mEvent. See note on default
    /// stream behavior for what is captured in the default case.<para/>
    /// Record() can be called multiple times on the same event and
    /// will overwrite the previously captured state. Other APIs such as
    /// Stream::WaitEvent() use the most recently captured state at the time
    /// of the API call, and are not affected by later calls to
    /// Record(). Before the first call to Record(), an
    /// event represents an empty set of work, so for example Query() would return cudaSuccess.
    /// </summary>
    void Record(const Stream &aStream, uint aFlags) const;

    /// <summary>
    /// Waits until the event has actually been recorded. If <see cref="Record()"/> has been called on this event, the
    /// function returns ErrorInvalidValue. Waiting for an event that was created with the CUEventFlags.BlockingSync
    /// flag will cause the calling CPU thread to block until the event has actually been recorded. <para/> If <see
    /// cref="Record()"/> has previously been called and the event has not been recorded yet, this function throws
    /// ErrorInvalidValue.
    /// </summary>
    void Synchronize() const;

    /// <summary>
    /// Returns true if the event has actually been recorded, or false if not. If
    /// <see cref="Record()"/> has not been called on this event, the function throws ErrorInvalidValue.
    /// </summary>
    [[nodiscard]] bool Query() const;

    /// <summary>
    /// Computes the elapsed time between two events (in milliseconds with a
    /// resolution of around 0.5 microseconds).<para/>
    /// If either event was last recorded in a non-NULL stream, the resulting time
    /// may be greater than expected (even if both used the same stream handle). This
    /// happens because the EventRecord() operation takes place asynchronously
    /// and there is no guarantee that the measured latency is actually just between
    /// the two events. Any number of other different stream operations could execute
    /// in between the two measured events, thus altering the timing in a significant
    /// way.<para/>
    /// If EventRecord() has not been called on either event, then
    /// ErrorInvalidResourceHandle is returned. If EventRecord() has been
    /// called on both events but one or both of them has not yet been completed
    /// (that is, EventQuery() would return ErrorNotReady on at least one
    /// of the events), ErrorNotReady is returned. If either event was created
    /// with the EventDisableTiming flag, then this function will return
    /// cudaErrorInvalidResourceHandle.
    /// </summary>
    static float ElapsedTime(const Event &aEventStart, const Event &aEventEnd);

  private:
};

/// <summary>
/// Shortcut for elapsed time, mathematically correct we put aEventEnd on the left side of the -sign
/// </summary>
MPPEXPORT_CUDACORE float operator-(const Event &aEventEnd, const Event &aEventStart);
} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_CORE