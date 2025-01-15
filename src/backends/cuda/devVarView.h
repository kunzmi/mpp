#pragma once
#include <common/moduleEnabler.h>
#if OPP_ENABLE_CUDA_CORE

#include "cudaException.h"
#include "stream.h"
#include <common/defines.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <vector>

namespace opp::cuda
{

/// <summary>
/// A wrapper class for a CUDA Device variable.
/// </summary>
template <typename T> class DevVarView
{
  private:
    T *mDevPtr{nullptr};
    size_t mSize{0};
    static constexpr size_t mTypeSize{sizeof(T)};

  protected:
    /// <summary>
    /// Creates an empty DevVarView (nullptr)
    /// </summary>
    DevVarView() = default;

    T *&PointerRef()
    {
        return mDevPtr;
    }

    size_t &SizeRef()
    {
        return mSize;
    }

    DevVarView(size_t aSize) : mDevPtr(nullptr), mSize(aSize)
    {
    }

  public:
    /// <summary>
    /// Creates a new DevVarView from an existing T*.
    /// </summary>
    /// <param name="aDevPtr"></param>
    /// <param name="aSizeInBytes">Size in Bytes</param>
    DevVarView(T *aDevPtr, size_t aSizeInBytes) : mDevPtr(aDevPtr)
    {
        mSize = aSizeInBytes / mTypeSize;
        if (aSizeInBytes != mSize * mTypeSize)
        {
            throw CUDAEXCEPTION("Variable size is not a multiple of its type size. Size in bytes: "
                                << aSizeInBytes << " number of elements: " << mSize << " type size: " << mTypeSize);
        }
    }

    ~DevVarView() = default;

    DevVarView(const DevVarView &aOther)     = default;
    DevVarView(DevVarView &&aOther) noexcept = default;

    DevVarView &operator=(const DevVarView &aOther)     = default;
    DevVarView &operator=(DevVarView &&aOther) noexcept = default;

    /// <summary>
    /// Checks for equality (pointed address and size)
    /// </summary>
    [[nodiscard]] bool operator==(const DevVarView &aOther) const
    {
        bool ret = true;
        ret &= mDevPtr == aOther.mDevPtr;
        ret &= mSize == aOther.mSize;
        // typesize is equal as it is the same type...

        return ret;
    }

    /// <summary>
    /// Checks for inequality (pointed address and size)
    /// </summary>
    [[nodiscard]] bool operator!=(const DevVarView &aOther) const
    {
        bool ret = true;
        ret &= mDevPtr == aOther.mDevPtr;
        ret &= mSize == aOther.mSize;
        // typesize is equal as it is the same type...

        return !ret;
    }

    /// <summary>
    /// Gets a null-pointer equivalent
    /// </summary>
    static DevVarView Null()
    {
        return DevVarView(nullptr, 0);
    }

    /// <summary>
    /// The inner T*
    /// </summary>
    [[nodiscard]] const T *Pointer() const
    {
        return mDevPtr;
    }

    /// <summary>
    /// The inner T*
    /// </summary>
    [[nodiscard]] T *Pointer()
    {
        return mDevPtr;
    }

    /// <summary>
    /// Size in bytes
    /// </summary>
    [[nodiscard]] size_t SizeInBytes() const
    {
        return mSize * mTypeSize;
    }

    /// <summary>
    /// Type size in bytes
    /// </summary>
    [[nodiscard]] static constexpr size_t TypeSize()
    {
        return mTypeSize;
    }

    /// <summary>
    /// Size in elements
    /// </summary>
    [[nodiscard]] size_t Size() const
    {
        return mSize;
    }

    /// <summary>
    /// Memset (value for each byte)
    /// </summary>
    void Memset(int aValue)
    {
        cudaSafeCall(cudaMemset(mDevPtr, aValue, mSize * mTypeSize));
    }

    /// <summary>
    /// Memset (value for each byte)
    /// </summary>
    void Memset(int aValue, const Stream &aStream)
    {
        cudaSafeCall(cudaMemsetAsync(mDevPtr, aValue, mSize * mTypeSize, aStream.Original()));
    }

    /// <summary>
    /// Copy data from device to device memory
    /// </summary>
    /// <param name="aSource">aSource</param>
    void CopyToDevice(const DevVarView &aSource)
    {
        const size_t sizeInBytes = std::min(SizeInBytes(), aSource.SizeInBytes());
        cudaSafeCallExt(cudaMemcpy(mDevPtr, aSource.Pointer(), sizeInBytes, cudaMemcpyDeviceToDevice),
                        "Dest number of elements: " << mSize << " Source number of elements: " << aSource.mSize
                                                    << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from host to device memory
    /// </summary>
    /// <param name="aSource">Source pointer to host memory</param>
    void CopyToDevice(const std::vector<T> &aSource)
    {
        const size_t sizeInBytes = std::min(SizeInBytes(), aSource.size() * mTypeSize);
        cudaSafeCallExt(cudaMemcpy(mDevPtr, aSource.data(), sizeInBytes, cudaMemcpyHostToDevice),
                        "Dest number of elements: " << mSize << " Source number of elements: " << aSource.size()
                                                    << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from host to device memory
    /// </summary>
    void CopyToDevice(T *aSource, size_t aSizeInElements = 0)
    {
        size_t sizeInBytes = std::min(SizeInBytes(), aSizeInElements * mTypeSize);
        if (aSizeInElements == 0)
        {
            sizeInBytes = SizeInBytes();
        }
        cudaSafeCallExt(cudaMemcpy(mDevPtr, aSource, sizeInBytes, cudaMemcpyHostToDevice),
                        "Dest number of elements: " << mSize << " Source number of elements: " << aSizeInElements
                                                    << " (0 = default) type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from host to device memory
    /// </summary>
    void CopyToDevice(const T &aSource)
    {
        cudaSafeCallExt(cudaMemcpy(mDevPtr, &aSource, mTypeSize, cudaMemcpyHostToDevice),
                        "Dest number of elements: " << mSize << " Source number of elements: " << 1
                                                    << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from host to device memory
    /// </summary>
    void CopyToDevice(const void *aSource, size_t aSizeInBytes = 0)
    {
        size_t sizeInBytes = std::min(SizeInBytes(), aSizeInBytes);
        if (aSizeInBytes == 0)
        {
            sizeInBytes = SizeInBytes();
        }
        cudaSafeCallExt(cudaMemcpy(mDevPtr, aSource, sizeInBytes, cudaMemcpyHostToDevice),
                        "Dest number of elements: " << mSize << " Source size in bytes: " << aSizeInBytes
                                                    << " (0 = default) type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from device to host memory
    /// </summary>
    void CopyToHost(std::vector<T> &aDest) const
    {
        const size_t sizeInBytes = std::min(SizeInBytes(), aDest.size() * mTypeSize);
        cudaSafeCallExt(cudaMemcpy(aDest.data(), mDevPtr, sizeInBytes, cudaMemcpyDeviceToHost),
                        "Dest number of elements: " << aDest.size() << " Source number of elements: " << mSize
                                                    << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from device to host memory
    /// </summary>
    void CopyToHost(T *aDest, size_t aSizeInElements = 0)
    {
        size_t sizeInBytes = std::min(SizeInBytes(), aSizeInElements * mTypeSize);
        if (aSizeInElements == 0)
        {
            sizeInBytes = SizeInBytes();
        }
        cudaSafeCallExt(cudaMemcpy(aDest, mDevPtr, sizeInBytes, cudaMemcpyDeviceToHost),
                        "Dest number of elements: " << aSizeInElements << " (0 = default) Source number of elements: "
                                                    << mSize << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from device to host memory
    /// </summary>
    void CopyToHost(T &aDest) const
    {
        cudaSafeCallExt(cudaMemcpy(&aDest, mDevPtr, mTypeSize, cudaMemcpyDeviceToHost),
                        "Dest number of elements: " << 1 << " Source number of elements: " << mSize
                                                    << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy data from device to host memory
    /// </summary>
    void CopyToHost(void *aDest, size_t aSizeInBytes = 0) const
    {
        size_t sizeInBytes = std::min(SizeInBytes(), aSizeInBytes);
        if (aSizeInBytes == 0)
        {
            sizeInBytes = SizeInBytes();
        }
        cudaSafeCallExt(cudaMemcpy(aDest, mDevPtr, sizeInBytes, cudaMemcpyDeviceToHost),
                        "Dest size in bytes: " << aSizeInBytes << " (0 = default) Source number of elements: " << mSize
                                               << " type size: " << mTypeSize);
    }

    /// <summary>
    /// Copy from device to host memory
    /// </summary>
    void operator>>(void *aDest) const
    {
        CopyToHost(aDest);
    }

    /// <summary>
    /// Copy from host to device memory
    /// </summary>
    void operator<<(const void *aSource)
    {
        CopyToDevice(aSource);
    }

    void operator>>(std::vector<T> &aDest) const
    {
        CopyToHost(aDest);
    }

    void operator<<(const std::vector<T> &aSource)
    {
        CopyToDevice(aSource);
    }

    void operator>>(T &aDest) const
    {
        CopyToHost(aDest);
    }

    void operator<<(const T &aSource)
    {
        CopyToDevice(aSource);
    }
};

} // namespace opp::cuda
#endif // OPP_ENABLE_CUDA_CORE
