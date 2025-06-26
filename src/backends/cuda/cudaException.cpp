#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_CORE

#include "cudaException.h"
#include <common/exception.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace mpp::cuda
{
const char *CudaException::ConvertErrorCodeToMessage(cudaError_t aErrorCode)
{
    return cudaGetErrorString(aErrorCode);
}

CudaException::CudaException(cudaError_t aCuResult, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                             [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
    const char *errorDescr = ConvertErrorCodeToMessage(aCuResult);
#ifdef NDEBUG
    std::stringstream ss;
    ss << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Cuda-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl;
#endif

    What() = ss.str();
}

CudaException::CudaException(cudaError_t aCuResult, const std::string &aMessage,
                             [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                             [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
    : MPPException(aMessage)
{
    const char *errorDescr = ConvertErrorCodeToMessage(aCuResult);

#ifdef NDEBUG
    std::stringstream ss;
    ss << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl << "Additional info: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Cuda-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl
       << "Additional info: " << aMessage;
#endif

    What() = ss.str();
}

CudaException::CudaException(const std::string &aMessage, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                             [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
    : MPPException(aMessage)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Error message: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Cuda-Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CudaUnsupportedException::CudaUnsupportedException(const std::string &aKernelName, const std::string &aMessage,
                                                   [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                                                   [[maybe_unused]] int aLineNumber,
                                                   [[maybe_unused]] const std::string &aFunctionName)
    : MPPException(aMessage), mKernelName(aKernelName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "Kernel name: " << aKernelName << std::endl << "Error message: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Unsupported Cuda platform error in " << codeFileName.generic_string() << " in function " << aFunctionName
       << " @ " << aLineNumber << std::endl
       << "Kernel name: " << aKernelName << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

} // namespace mpp::cuda
#endif // MPP_ENABLE_CUDA_CORE