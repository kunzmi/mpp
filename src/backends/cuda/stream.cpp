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
Stream::Stream() : mIsOwner(true)
{
    cudaSafeCall(cudaStreamCreate(&mStream));
}
Stream::Stream(cudaStream_t aStream) : mStream(aStream)
{
}
Stream::Stream(uint aFlags) : mIsOwner(true)
{
    cudaSafeCall(cudaStreamCreateWithFlags(&mStream, aFlags));
}
Stream::~Stream()
{
    if (mIsOwner)
    {
        // ignore if failing:
        cudaStreamDestroy(mStream);
        mIsOwner = false;
    }
}
Stream::Stream(const Stream &aOther) : mStream(aOther.mStream)
{
}

Stream::Stream(Stream &&aOther) noexcept : mStream(aOther.mStream), mIsOwner(aOther.mIsOwner)
{
    aOther.mStream  = nullptr;
    aOther.mIsOwner = false;
}

Stream &Stream::operator=(const Stream &aOther)
{
    if (&aOther == this)
    {
        return *this;
    }

    // if *this is owner of a previously associated stream, destroy it:
    if (mIsOwner)
    {
        cudaSafeCall(cudaStreamDestroy(mStream));
        mIsOwner = false;
    }

    mStream  = aOther.mStream;
    mIsOwner = false;

    return *this;
}
Stream &Stream::operator=(Stream &&aOther) noexcept
{
    if (&aOther == this)
    {
        return *this;
    }

    if (mIsOwner)
    {
        cudaStreamDestroy(mStream); // noexcept, so ignore if failing
        mIsOwner = false;
    }

    mStream  = aOther.mStream;
    mIsOwner = aOther.mIsOwner;

    aOther.mStream  = nullptr;
    aOther.mIsOwner = false;

    return *this;
}
void Stream::Synchronize() const
{
    cudaSafeCall(cudaStreamSynchronize(mStream));
}
bool Stream::Query() const
{
    const cudaError_t res = cudaStreamQuery(mStream);
    if (res != cudaSuccess && res != cudaErrorNotReady)
    {
        cudaSafeCall(res);
    }

    return res == cudaSuccess;
}
void Stream::WaitEvent(cudaEvent_t aCUevent, uint aFlags) const
{
    cudaSafeCall(cudaStreamWaitEvent(mStream, aCUevent, aFlags));
}

void Stream::WaitEvent(const Event &aEvent, uint aFlags) const
{
    WaitEvent(aEvent.Original(), aFlags);
}

void Stream::AddCallback(cudaStreamCallback_t aCallback, void *aUserData, uint aFlags)
{
    cudaSafeCall(cudaStreamAddCallback(mStream, aCallback, aUserData, aFlags));
}

#include <common/disableWarningsBegin.h>
// NOLINTNEXTLINE(hicpp-use-nullptr,modernize-use-nullptr, cert-err58-cpp)
const Stream Stream::Null(static_cast<cudaStream_t>(0));

const Stream Stream::Legacy(cudaStreamLegacy); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast, cert-err58-cpp)

const Stream Stream::PerThread(cudaStreamPerThread); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast, cert-err58-cpp)

#include <common/disableWarningsEnd.h>
} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_CORE
