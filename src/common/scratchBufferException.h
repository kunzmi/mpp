#pragma once
#include <common/defines.h>
#include <common/exception.h>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace opp
{
/// <summary>
/// ScratchBufferException is thrown when a provided scratch buffer is smaller than the required buffer size.
/// </summary>
class ScratchBufferException : public OPPException
{
  public:
    ScratchBufferException(size_t aRequiredSize, size_t aProvidedSize, const std::string &aMessage,
                           const std::filesystem::path &aCodeFileName, int aLineNumber,
                           const std::string &aFunctionName);
    ~ScratchBufferException() noexcept override = default;

    ScratchBufferException(ScratchBufferException &&)                 = default;
    ScratchBufferException(const ScratchBufferException &)            = default;
    ScratchBufferException &operator=(const ScratchBufferException &) = delete;
    ScratchBufferException &operator=(ScratchBufferException &&)      = delete;
};
} // namespace opp

// NOLINTBEGIN --> function like macro, parantheses for "msg"...
#define SCRATCHBUFFEREXCEPTION_EXT(requiredSize, providedSize, msg)                                                    \
    (opp::ScratchBufferException(requiredSize, providedSize, (std::ostringstream() << msg).str(), __FILE__, __LINE__,  \
                                 __PRETTY_FUNCTION__))

#define SCRATCHBUFFEREXCEPTION(requiredSize, providedSize)                                                             \
    (opp::ScratchBufferException(requiredSize, providedSize, "", __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define CHECK_BUFFER_SIZE(buffer, providedSize)                                                                        \
    {                                                                                                                  \
        if (buffer.GetTotalBufferSize() > (providedSize))                                                              \
        {                                                                                                              \
            throw SCRATCHBUFFEREXCEPTION(buffer.GetTotalBufferSize(), (providedSize));                                 \
        }                                                                                                              \
    }
// NOLINTEND
