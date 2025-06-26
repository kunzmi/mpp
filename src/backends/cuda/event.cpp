#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_CORE

#include "cudaException.h"
#include "event.h"
#include "stream.h"
#include <common/defines.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace mpp::cuda
{
Event::Event() : mIsOwner(true)
{
    cudaSafeCall(cudaEventCreate(&mEvent));
}
Event::Event(cudaEvent_t aEvent) : mEvent(aEvent)
{
}
Event::Event(uint aFlags) : mIsOwner(true)
{
    cudaSafeCall(cudaEventCreateWithFlags(&mEvent, aFlags));
}
Event::~Event()
{
    if (mIsOwner)
    {
        // ignore if failing:
        cudaEventDestroy(mEvent);
        mIsOwner = false;
    }
}
Event::Event(const Event &aOther) : mEvent(aOther.mEvent)
{
}

Event::Event(Event &&aOther) noexcept : mEvent(aOther.mEvent), mIsOwner(aOther.mIsOwner)
{
    aOther.mEvent   = nullptr;
    aOther.mIsOwner = false;
}

Event &Event::operator=(const Event &aOther)
{
    if (&aOther == this)
    {
        return *this;
    }

    // if *this is owner of a previously associated event, destroy it:
    if (mIsOwner)
    {
        cudaSafeCall(cudaEventDestroy(mEvent));
        mIsOwner = false;
    }

    mEvent   = aOther.mEvent;
    mIsOwner = false;

    return *this;
}
Event &Event::operator=(Event &&aOther) noexcept
{
    if (&aOther == this)
    {
        return *this;
    }

    if (mIsOwner)
    {
        cudaEventDestroy(mEvent); // noexcept, so ignore if failing
        mIsOwner = false;
    }

    mEvent   = aOther.mEvent;
    mIsOwner = aOther.mIsOwner;

    aOther.mEvent   = nullptr;
    aOther.mIsOwner = false;

    return *this;
}

void Event::Record() const
{
    cudaSafeCall(cudaEventRecord(mEvent));
}

void Event::Record(cudaStream_t aStream) const
{
    cudaSafeCall(cudaEventRecord(mEvent, aStream));
}

void Event::Record(const Stream &aStream) const
{
    Record(aStream.Original());
}

void Event::Record(uint aFlags) const
{
    cudaSafeCall(cudaEventRecordWithFlags(mEvent, nullptr, aFlags));
}

void Event::Record(cudaStream_t aStream, uint aFlags) const
{
    cudaSafeCall(cudaEventRecordWithFlags(mEvent, aStream, aFlags));
}

void Event::Record(const Stream &aStream, uint aFlags) const
{
    Record(aStream.Original(), aFlags);
}

void Event::Synchronize() const
{
    cudaSafeCall(cudaEventSynchronize(mEvent));
}
bool Event::Query() const
{
    const cudaError_t res = cudaEventQuery(mEvent);
    if (res != cudaSuccess && res != cudaErrorNotReady)
    {
        cudaSafeCall(res);
    }

    return res == cudaSuccess;
}

float Event::ElapsedTime(const Event &aEventStart, const Event &aEventEnd)
{
    float time = 0;
    cudaSafeCall(cudaEventElapsedTime(&time, aEventStart.Original(), aEventEnd.Original()));
    return time;
}

float operator-(const Event &aEventEnd, const Event &aEventStart)
{
    return Event::ElapsedTime(aEventStart, aEventEnd);
}
} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_CORE
