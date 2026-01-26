#pragma once
#include "../dllexport_common.h"
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/channel.h>
#include <common/vector_typetraits.h>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

namespace mpp::image
{
/// <summary>
/// ChannelException is thrown when a channel exceeds the valid range.
/// </summary>
class MPPEXPORT_COMMON ChannelException : public MPPException
{
  public:
    ChannelException(const std::string &aMessage, const std::filesystem::path &aCodeFileName, int aLineNumber,
                     const std::string &aFunctionName);
    ChannelException(const Channel &aChannel, uint aMaxChannels, const std::filesystem::path &aCodeFileName,
                     int aLineNumber, const std::string &aFunctionName);
    ~ChannelException() noexcept override = default;

    ChannelException(ChannelException &&) noexcept        = default;
    ChannelException(const ChannelException &)            = default;
    ChannelException &operator=(const ChannelException &) = delete;
    ChannelException &operator=(ChannelException &&)      = delete;

    [[nodiscard]] ExceptionCode GetCode() const override
    {
        return ExceptionCode::Channel;
    }
};
} // namespace mpp::image

// NOLINTBEGIN --> function like macro, parantheses for "msg",
// bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp...

/// <summary>
/// Checks if Channel is in the valid range
/// </summary>
inline void __checkChannel(const mpp::image::Channel &aChannel, mpp::uint aMaxChannels, const char *aCodeFile,
                           int aLine, const char *aFunction)
{
    if (aChannel.Value() >= aMaxChannels)
    {
        throw mpp::image::ChannelException(aChannel, aMaxChannels, aCodeFile, aLine, aFunction);
    }
}

#define CHANNELEXCEPTION(msg)                                                                                          \
    (mpp::image::ChannelException((std::ostringstream() << msg).str(), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define checkChannel(channel, maxchannel)                                                                              \
    (__checkChannel((channel), (maxchannel), __FILE__, __LINE__, __PRETTY_FUNCTION__))

#define checkChannelT(channel, type)                                                                                   \
    (__checkChannel((channel), mpp::vector_size_v<type>, __FILE__, __LINE__, __PRETTY_FUNCTION__))

// NOLINTEND
