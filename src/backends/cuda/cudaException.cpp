#include "cudaException.h"
#include <common/exception.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>

namespace opp::cuda
{
const char *CudaException::ConvertErrorCodeToMessage(cudaError_t aErrorCode)
{
    return cudaGetErrorString(aErrorCode);
}

CudaException::CudaException(cudaError_t aCuResult, const std::filesystem::path &aCodeFileName, int aLineNumber,
                             const std::string &aFunctionName)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

    const char *errorDescr = ConvertErrorCodeToMessage(aCuResult);
#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl;
#else
    std::stringstream ss;
    ss << "Cuda-Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl;
#endif

    What() = ss.str();
}

CudaException::CudaException(cudaError_t aCuResult, const std::string &aMessage,
                             const std::filesystem::path &aCodeFileName, int aLineNumber,
                             const std::string &aFunctionName)
    : OPPException(aMessage)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

    const char *errorDescr = ConvertErrorCodeToMessage(aCuResult);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl << "Additional info: " << aMessage;
#else
    std::stringstream ss;
    ss << "Cuda-Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "'" << errorDescr << "' (" << aCuResult << ") " << std::endl
       << "Additional info: " << aMessage;
#endif

    What() = ss.str();
}

CudaException::CudaException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                             const std::string &aFunctionName)
    : OPPException(aMessage)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "Error message: " << aMessage;
#else
    std::stringstream ss;
    ss << "Cuda-Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

} // namespace opp::cuda