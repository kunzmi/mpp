#pragma once
#include "dllexport_common.h"
#include <filesystem>
#include <sstream>
#include <stdexcept>

#ifdef _MSC_VER
// to have the same macro for GCC and MSVC:
#define __PRETTY_FUNCTION__ __FUNCSIG__ // NOLINT(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#endif

namespace mpp
{
// each exception type is identified with a unique code that we will centralize here in this enum:
enum class ExceptionCode : int // NOLINT
{
    Unknown              = -999999,
    InvalidArgument      = -1,
    NullPtr              = -2,
    ScratchBuffer        = -3,
    Roi                  = -4,
    Channel              = -5,
    FilterArea           = -6,
    AffineTransformation = -7,
    Matrix               = -8,
    Cuda                 = -1000,
    CudaUnsupported      = -1001,
    Npp                  = -10000
};

/// <summary>
/// Exception base class for all exceptions thrown in MPP.
/// </summary>
class MPPEXPORT_COMMON MPPException : public std::exception
{
  private:
    std::string mWhat;
    std::string mMessage;

  protected:
    std::string &What();
    std::string &Message();

  public:
    MPPException() noexcept = default;
    explicit MPPException(std::string aMessage);
    ~MPPException() noexcept override;

    // we have linking issues in MSVC with derived exceptions in DLL if we use the default constructors:
    MPPException(MPPException &&aOther) noexcept;
    MPPException(const MPPException &aOther);
    MPPException &operator=(const MPPException &) = delete;
    MPPException &operator=(MPPException &&)      = delete;

    /// <summary>
    /// A detailed string containing the error message and the location where the error happend. More verbose in debug
    /// builds.
    /// </summary>
    [[nodiscard]] const char *what() const noexcept override;
    /// <summary>
    /// The error message that was provided when throwing the exception
    /// </summary>
    [[nodiscard]] const std::string &Message() const noexcept;

    [[nodiscard]] virtual ExceptionCode GetCode() const
    {
        return ExceptionCode::Unknown;
    }
};

/// <summary>
/// Exception is a general exception.
/// </summary>
class MPPEXPORT_COMMON Exception : public MPPException
{
  public:
    Exception(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
              const std::string &aFunctionName);
    ~Exception() noexcept override = default;

    Exception(Exception &&) noexcept        = default;
    Exception(const Exception &)            = default;
    Exception &operator=(const Exception &) = delete;
    Exception &operator=(Exception &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::Unknown;
    }
};

/// <summary>
/// InvalidArgumentException is thrown if an argument is not in an acceptable value range.
/// </summary>
class MPPEXPORT_COMMON InvalidArgumentException : public MPPException
{
  public:
    InvalidArgumentException(const std::string &aArgumentName, const std::string &aMessage,
                             const std::filesystem::path &aCodeFileName, int aLineNumber,
                             const std::string &aFunctionName);
    ~InvalidArgumentException() noexcept override = default;

    InvalidArgumentException(InvalidArgumentException &&) noexcept        = default;
    InvalidArgumentException(const InvalidArgumentException &)            = default;
    InvalidArgumentException &operator=(const InvalidArgumentException &) = delete;
    InvalidArgumentException &operator=(InvalidArgumentException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::InvalidArgument;
    }
};

/// <summary>
/// NullPtrException is thrown if an argument is nullptr but shouldn't.
/// </summary>
class MPPEXPORT_COMMON NullPtrException : public MPPException
{
  public:
    NullPtrException(const std::string &aArgumentName, const std::filesystem::path &aCodeFileName, int aLineNumber,
                     const std::string &aFunctionName);
    NullPtrException(const std::string &aArgumentName, const std::string &aMessage,
                     const std::filesystem::path &aCodeFileName, int aLineNumber, const std::string &aFunctionName);
    ~NullPtrException() noexcept override = default;

    NullPtrException(NullPtrException &&) noexcept        = default;
    NullPtrException(const NullPtrException &)            = default;
    NullPtrException &operator=(const NullPtrException &) = delete;
    NullPtrException &operator=(NullPtrException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::NullPtr;
    }
};
} // namespace mpp

// NOLINTBEGIN --> function like macro, parantheses for "msg"...
/// <summary>
/// Checks if a pointer is nullptr and throws NullPtrException if it is
/// </summary>
inline void __checkNullptr(const void *aPtr, const std::string &aName, const char *aCodeFile, int aLine,
                           const char *aFunction)
{
    if (aPtr == nullptr)
    {
        throw mpp::NullPtrException(aName, aCodeFile, aLine, aFunction);
    }
}
#define EXCEPTION(msg) (mpp::Exception((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define INVALIDARGUMENT(argument, msg)                                                                                 \
    (mpp::InvalidArgumentException(#argument, (std::ostringstream() << msg).str(), __FILE__, __LINE__,                 \
                                   __PRETTY_FUNCTION__))

#define NULLPTR(argument, msg)                                                                                         \
    (mpp::NullPtrException(#argument, (std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define checkNullptr(aPtr) __checkNullptr(aPtr, #aPtr, __FILE__, __LINE__, __PRETTY_FUNCTION__)

// NOLINTEND