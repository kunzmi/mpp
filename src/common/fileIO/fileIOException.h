#pragma once
#include "dllexport_fileio.h"
#include <common/defines.h>
#include <common/exception.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace mpp::fileIO
{
/// <summary>
/// FileIOException is thrown when a file cannot be accessed as intended.
/// </summary>
class MPPEXPORT_COMMON_FILEIO FileIOException : public MPPException
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
} // namespace mpp::fileIO

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

#define FILEIOEXCEPTION(filename, msg)                                                                                 \
    (mpp::fileIO::FileIOException(filename, (std::ostringstream() << msg).str(), __FILE__, __LINE__,                   \
                                  __PRETTY_FUNCTION__))

// NOLINTEND
