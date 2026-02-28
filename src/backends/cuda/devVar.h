#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_CUDA_CORE

#include "DevVarView.h"
#include "cudaException.h"
#include <common/defines.h>
#include <cuda_runtime_api.h>

namespace mpp::cuda
{
/// <summary>
/// A wrapper class for a CUDA Device variable.
/// </summary>
template <typename T> class DevVar : public DevVarView<T>
{
  public:
    DevVar() = delete;

    /// <summary>
    /// Creates a new DevVar and allocates the memory on the device
    /// </summary>
    /// <param name="aSize">In elements</param>
    explicit DevVar(size_t aSize) : DevVarView<T>(aSize)
    {
        cudaSafeCallExt(
            cudaMalloc(reinterpret_cast<void **>(&DevVarView<T>::PointerRef()), DevVarView<T>::TypeSize() * aSize),
            "Number of elements = " << aSize << " element size = " << DevVarView<T>::TypeSize() << " [bytes]");
    }

    ~DevVar()
    {
        if (DevVarView<T>::PointerRef() != nullptr)
        {
            // ignore if failing, as the destructor is also called after an cuda-exception and
            // the cuda-context might already be invalid
            cudaFree(DevVarView<T>::PointerRef());
        }
        DevVarView<T>::PointerRef() = nullptr;
        DevVarView<T>::SizeRef()    = 0;
    }

    DevVar(const DevVar &aOther) = delete;
    DevVar(DevVar &&aOther) noexcept
    {
        DevVarView<T>::PointerRef() = aOther.PointerRef();
        DevVarView<T>::SizeRef()    = aOther.SizeRef();

        aOther.PointerRef() = nullptr;
        aOther.SizeRef()    = 0;
    }

    DevVar &operator=(const DevVar &aOther) = delete;
    DevVar &operator=(DevVar &&aOther) noexcept
    {
        if (std::addressof(aOther) == std::addressof(*this))
        {
            return *this;
        }
        DevVarView<T>::PointerRef() = aOther.PointerRef();
        DevVarView<T>::SizeRef()    = aOther.SizeRef();

        aOther.PointerRef() = nullptr;
        aOther.SizeRef()    = 0;
        return *this;
    }
};

} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_CORE
