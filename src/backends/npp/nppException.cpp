#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_NPP_BACKEND

#include "nppException.h"
#include <filesystem>
#include <nppdefs.h>
#include <ostream>
#include <sstream>
#include <string>

namespace opp::npp
{
NppException::NppException(NppStatus aNppStatus, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                           [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl;
#endif

    What() = ss.str();
}

NppException::NppException(NppStatus aNppStatus, const std::string &aMessage,
                           [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                           [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl
       << "Additional info: " << aMessage;
#else
#ifdef PROJECT_SOURCE_DIR
    const std::filesystem::path src          = PROJECT_SOURCE_DIR;
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);
#else
    const std::filesystem::path codeFileName = aCodeFileName;
#endif

    std::stringstream ss;
    ss << "Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "'" << ConvertErrorCodeToName(aNppStatus) << "' (" << aNppStatus << ") "
       << ConvertErrorCodeToMessage(aNppStatus) << std::endl
       << "Additional info: " << aMessage;
#endif

    What() = ss.str();
}

NppException::NppException(const std::string &aMessage, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                           [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
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
    ss << "Error in " << codeFileName.generic_string() << " in function " << aFunctionName << " @ " << aLineNumber
       << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}
} // namespace opp::npp
#endif // OPP_ENABLE_NPP_BACKEND