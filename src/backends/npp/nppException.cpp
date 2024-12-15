#include "nppException.h"
#include <common/defines.h>
#include <common/exception.h>
#include <filesystem>
#include <npp.h>
#include <sstream>

namespace opp::npp
{
NppException::NppException(NppStatus aNppStatus, const std::filesystem::path &aCodeFileName, int aLineNumber,
                           const std::string &aFunctionName)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl;
#else
    std::stringstream ss;
    ss << "Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber << std::endl
       << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl;
#endif

    What() = ss.str();
}

NppException::NppException(NppStatus aNppStatus, const std::string &aMessage,
                           const std::filesystem::path &aCodeFileName, int aLineNumber,
                           const std::string &aFunctionName)
{
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

#ifdef NDEBUG
    if (codeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl
       << "Additional info: " << aMessage;
#else
    std::stringstream ss;
    ss << "Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber << std::endl
       << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl
       << "Additional info: " << aMessage;
#endif

    What() = ss.str();
}

NppException::NppException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                           const std::string &aFunctionName)
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
    ss << "Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}
} // namespace opp::npp