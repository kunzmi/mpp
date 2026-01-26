#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_CUDA_CORE

#include "dllexport_cudacore.h"
#include <common/defines.h>
#include <common/exception.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace mpp::cuda
{
/// <summary>
/// CudaException is thrown when a CUDA API call via cudaSafeCall does not return cudaSuccess.
/// </summary>
class MPPEXPORT_CUDACORE CudaException : public MPPException
{
  private:
    static const char *ConvertErrorCodeToMessage(cudaError_t aErrorCode);

  public:
    CudaException(cudaError_t aCuResult, const std::filesystem::path &aCodeFileName, int aLineNumber,
                  const std::string &aFunctionName);
    CudaException(cudaError_t aCuResult, const std::string &aMessage, const std::filesystem::path &aCodeFileName,
                  int aLineNumber, const std::string &aFunctionName);
    CudaException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                  const std::string &aFunctionName);
    ~CudaException() noexcept override;

    // we have linking issues in MSVC with derived exceptions in DLL if we use the default constructors:
    CudaException(CudaException && /*aOther*/) noexcept;
    CudaException(const CudaException & /*aOther*/);
    CudaException &operator=(const CudaException &) = delete;
    CudaException &operator=(CudaException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::Cuda;
    }
};

/// <summary>
/// CudaUnsupported is thrown when we try to execute a kernel on an unsupported hardware version.
/// </summary>
class MPPEXPORT_CUDACORE CudaUnsupportedException : public MPPException
{
  private:
    std::string mKernelName;

  public:
    CudaUnsupportedException(const std::string &aKernelName, const std::string &aMessage,
                             const std::filesystem::path &aCodeFileName, int aLineNumber,
                             const std::string &aFunctionName);
    ~CudaUnsupportedException() noexcept override = default;

    CudaUnsupportedException(CudaUnsupportedException &&)                 = default;
    CudaUnsupportedException(const CudaUnsupportedException &)            = default;
    CudaUnsupportedException &operator=(const CudaUnsupportedException &) = delete;
    CudaUnsupportedException &operator=(CudaUnsupportedException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::CudaUnsupported;
    }
};
} // namespace mpp::cuda

/// <summary>
/// Checks if a cudaError_t is cudaSuccess and throws a corresponding CudaExcption if not
/// </summary>
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
inline void __cudaSafeCall(cudaError_t aErr, const char *aCodeFile, int aLine, const char *aFunction)
{
    if (cudaSuccess != aErr)
    {
        throw mpp::cuda::CudaException(aErr, aCodeFile, aLine, aFunction);
    }
}

/// <summary>
/// Checks if a cudaError_t is cudaSuccess and throws a corresponding CudaExcption if not
/// </summary>
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
inline void __cudaSafeCall(cudaError_t aErr, const std::string &aMessage, const char *aCodeFile, int aLine,
                           const char *aFunction)
{
    if (cudaSuccess != aErr)
    {
        throw mpp::cuda::CudaException(aErr, aMessage, aCodeFile, aLine, aFunction);
    }
}

// NOLINTBEGIN --> function like macro, parantheses for "msg"...
#define cudaSafeCall(aErr) __cudaSafeCall(aErr, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#define cudaSafeCallExt(aErr, msg)                                                                                     \
    {                                                                                                                  \
        cudaError_t __res = (aErr);                                                                                    \
        if (__res != cudaSuccess)                                                                                      \
        {                                                                                                              \
            __cudaSafeCall(__res, (std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__);       \
        }                                                                                                              \
    }

#define CUDAEXCEPTION(msg)                                                                                             \
    (mpp::cuda::CudaException((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define peekAndCheckLastCudaError(msg)                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                                                \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            throw mpp::cuda::CudaException(cudaStatus, (std::ostringstream() << msg).str(), __FILE__, __LINE__,        \
                                           __PRETTY_FUNCTION__);                                                       \
        }                                                                                                              \
    }

#define CheckLastCudaError(msg)                                                                                        \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaGetLastError();                                                                   \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            throw mpp::cuda::CudaException(cudaStatus, (std::ostringstream() << msg).str(), __FILE__, __LINE__,        \
                                           __PRETTY_FUNCTION__);                                                       \
        }                                                                                                              \
    }

#define CUDAUNSUPPORTED(kernelName, msg)                                                                               \
    (mpp::cuda::CudaUnsupportedException((#kernelName), (std::ostringstream() << msg).str(), __FILE__, __LINE__,       \
                                         __PRETTY_FUNCTION__))
// NOLINTEND

/// <summary>
/// define a pretty print for dim3 type used for block and grid sizes
/// </summary>
inline std::ostream &operator<<(std::ostream &aOs, const dim3 &aDim3)
{
    aOs << '(' << aDim3.x << ", " << aDim3.y << ", " << aDim3.z << ')';
    return aOs;
}

/// <summary>
/// define a pretty print for dim3 type used for block and grid sizes
/// </summary>
inline std::wostream &operator<<(std::wostream &aOs, const dim3 &aDim3)
{
    aOs << '(' << aDim3.x << ", " << aDim3.y << ", " << aDim3.z << ')';
    return aOs;
}

#endif // MPP_ENABLE_CUDA_CORE