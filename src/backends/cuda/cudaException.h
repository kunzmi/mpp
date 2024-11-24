#pragma once
#include <common/defines.h>
#include <common/exception.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace opp::cuda
{
/// <summary>
/// CudaException is thrown when a CUDA API call via cudaSafeCall does not return cudaSuccess.
/// </summary>
class CudaException : public OPPException
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
    ~CudaException() noexcept override = default;

    CudaException(CudaException &&)                 = default;
    CudaException(const CudaException &)            = default;
    CudaException &operator=(const CudaException &) = delete;
    CudaException &operator=(CudaException &&)      = delete;
};
} // namespace opp::cuda

/// <summary>
/// Checks if a cudaError_t is cudaSuccess and throws a corresponding CudaExcption if not
/// </summary>
// NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
inline void __cudaSafeCall(cudaError_t aErr, const char *aCodeFile, int aLine, const char *aFunction)
{
    if (cudaSuccess != aErr)
    {
        throw opp::cuda::CudaException(aErr, aCodeFile, aLine, aFunction);
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
        throw opp::cuda::CudaException(aErr, aMessage, aCodeFile, aLine, aFunction);
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
    (cuda::CudaException((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define peekAndCheckLastCudaError(msg)                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                                                \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            throw opp::cuda::CudaException(cudaStatus, (std::ostringstream() << msg).str(), __FILE__, __LINE__,        \
                                           __PRETTY_FUNCTION__);                                                       \
        }                                                                                                              \
    }

#define CheckLastCudaError(msg)                                                                                        \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaGetLastError();                                                                   \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            throw opp::cuda::CudaException(cudaStatus, (std::ostringstream() << msg).str(), __FILE__, __LINE__,        \
                                           __PRETTY_FUNCTION__);                                                       \
        }                                                                                                              \
    }
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