#include "exception.h"
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace opp
{
std::string &OPPException::What()
{
    return mWhat;
}
std::string &OPPException::Message()
{
    return mMessage;
}
OPPException::OPPException(std::string aMessage) : mMessage(std::move(aMessage))
{
}
const char *OPPException::what() const noexcept
{
    return mWhat.c_str();
}
const std::string &OPPException::Message() const noexcept
{
    return mMessage;
}

Exception::Exception(const std::string &aMessage, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                     [[maybe_unused]] int aLineNumber, [[maybe_unused]] const std::string &aFunctionName)
    : OPPException(aMessage)
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

InvalidArgumentException::InvalidArgumentException(const std::string &aArgumentName, const std::string &aMessage,
                                                   [[maybe_unused]] const std::filesystem::path &aCodeFileName,
                                                   [[maybe_unused]] int aLineNumber,
                                                   [[maybe_unused]] const std::string &aFunctionName)
    : OPPException(aMessage)
{
#ifdef NDEBUG
    std::stringstream ss;
    ss << "InvalidArgumentException for argument: '" << aArgumentName << "'" << std::endl
       << "Error message: " << aMessage;
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
       << "InvalidArgumentException for argument: '" << aArgumentName << "'" << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}
} // namespace opp