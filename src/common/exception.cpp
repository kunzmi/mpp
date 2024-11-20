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

Exception::Exception(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                     const std::string &aFunctionName)
    : OPPException(aMessage)
{
#ifdef NDEBUG
    if (aCodeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "Error message: " << aMessage;
#else
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

    std::stringstream ss;
    ss << "Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}

InvalidArgumentException::InvalidArgumentException(const std::string &aArgumentName, const std::string &aMessage,
                                                   const std::filesystem::path &aCodeFileName, int aLineNumber,
                                                   const std::string &aFunctionName)
    : OPPException(aMessage)
{
#ifdef NDEBUG
    if (aCodeFileName.empty() && aLineNumber > 0 && aFunctionName.empty())
    {
        // dummy to avoid warning because of unused arguments...
    }

    std::stringstream ss;
    ss << "InvalidArgumentException for argument: '" << aArgumentName << "'" << std::endl
       << "Error message: " << aMessage;
#else
    const std::filesystem::path src          = "../../../";
    const std::filesystem::path codeFileName = aCodeFileName.lexically_relative(src);

    std::stringstream ss;
    ss << "Error in " << codeFileName.string() << " in function " << aFunctionName << " @ " << aLineNumber << std::endl
       << "InvalidArgumentException for argument: '" << aArgumentName << "'" << std::endl
       << "Error message: " << aMessage;
#endif

    What() = ss.str();
}
} // namespace opp