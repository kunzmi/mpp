#include "exception.h"
#include <exception>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace mpp
{
// NOLINTNEXTLINE(hicpp-use-equals-default,modernize-use-equals-default)
MPPException::~MPPException() noexcept
{
}

MPPException::MPPException(MPPException &&aOther) noexcept
    // NOLINTNEXTLINE(bugprone-use-after-move,hicpp-invalid-access-moved)
    : std::exception(std::move(aOther)), mWhat(std::move(aOther.mWhat)), mMessage(std::move(aOther.mMessage))
{
}
// NOLINTNEXTLINE(hicpp-use-equals-default,modernize-use-equals-default)
MPPException::MPPException(const MPPException &aOther)
    : std::exception(aOther), mWhat(aOther.mWhat), mMessage(aOther.mMessage)
{
}
std::string &MPPException::What()
{
    return mWhat;
}
std::string &MPPException::Message()
{
    return mMessage;
}
MPPException::MPPException(std::string aMessage) : mMessage(std::move(aMessage))
{
}
const char *MPPException::what() const noexcept
{
    return mWhat.c_str();
}
const std::string &MPPException::Message() const noexcept
{
    return mMessage;
}

Exception::Exception(const std::string &aMessage, [[maybe_unused]] const std::filesystem::path &aCodeFileName,
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
    : MPPException(aMessage)
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
} // namespace mpp