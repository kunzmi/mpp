#pragma once
#include <common/defines.h>
#include <common/exception.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace opp::fileIO
{
/// <summary>
/// FileIOException is thrown when a file cannot be accessed as intended.
/// </summary>
class FileIOException : public OPPException
{
  public:
    FileIOException(const std::filesystem::path &aFileName, const std::string &aMessage,
                    const std::filesystem::path &aCodeFileName, int aLineNumber, const std::string &aFunctionName);
    ~FileIOException() noexcept override = default;

    FileIOException(FileIOException &&)                 = default;
    FileIOException(const FileIOException &)            = default;
    FileIOException &operator=(const FileIOException &) = delete;
    FileIOException &operator=(FileIOException &&)      = delete;
};
} // namespace opp::fileIO

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

#define FILEIOEXCEPTION(filename, msg)                                                                                 \
    (opp::fileIO::FileIOException(filename, (std::ostringstream() << msg).str(), __FILE__, __LINE__,                   \
                                  __PRETTY_FUNCTION__))

// NOLINTEND
