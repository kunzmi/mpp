#pragma once
#include <filesystem>
#include <sstream>
#include <stdexcept>

#ifdef _MSC_VER
// to have the same macro for GCC and MSVC:
#define __PRETTY_FUNCTION__ __FUNCSIG__ // NOLINT(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#endif

namespace opp
{
/// <summary>
/// Exception base class for all exceptions thrown in OPP.
/// </summary>
class OPPException : public std::exception
{
  private:
    std::string mWhat;
    std::string mMessage;

  protected:
    std::string &What();
    std::string &Message();

  public:
    OPPException() noexcept = default;
    explicit OPPException(std::string aMessage);
    ~OPPException() noexcept override = default;

    OPPException(OPPException &&)                 = default;
    OPPException(const OPPException &)            = default;
    OPPException &operator=(const OPPException &) = delete;
    OPPException &operator=(OPPException &&)      = delete;

    /// <summary>
    /// A detailed string containing the error message and the location where the error happend. More verbose in debug
    /// builds.
    /// </summary>
    [[nodiscard]] const char *what() const noexcept override;
    /// <summary>
    /// The error message that was provided when throwing the exception
    /// </summary>
    [[nodiscard]] const std::string &Message() const noexcept;
};

/// <summary>
/// Exception is a general exception.
/// </summary>
class Exception : public OPPException
{
  public:
    Exception(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
              const std::string &aFunctionName);
    ~Exception() noexcept override = default;

    Exception(Exception &&)                 = default;
    Exception(const Exception &)            = default;
    Exception &operator=(const Exception &) = delete;
    Exception &operator=(Exception &&)      = delete;
};

/// <summary>
/// InvalidArgumentException is thrown if an argument is not in an acceptable value range.
/// </summary>
class InvalidArgumentException : public OPPException
{
  public:
    InvalidArgumentException(const std::string &aArgumentName, const std::string &aMessage,
                             const std::filesystem::path &aCodeFileName, int aLineNumber,
                             const std::string &aFunctionName);
    ~InvalidArgumentException() noexcept override = default;

    InvalidArgumentException(InvalidArgumentException &&)                 = default;
    InvalidArgumentException(const InvalidArgumentException &)            = default;
    InvalidArgumentException &operator=(const InvalidArgumentException &) = delete;
    InvalidArgumentException &operator=(InvalidArgumentException &&)      = delete;
};
} // namespace opp

// NOLINTBEGIN --> function like macro, parantheses for "msg"...
#define EXCEPTION(msg) (opp::Exception((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define INVALIDARGUMENT(argument, msg)                                                                                 \
    (opp::InvalidArgumentException(#argument, (std::ostringstream() << msg).str(), __FILE__, __LINE__,                 \
                                   __PRETTY_FUNCTION__))
// NOLINTEND