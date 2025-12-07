#pragma once
#include "../dllexport_common.h"
#include "channel.h"
#include <array>
#include <common/defines.h>
#include <common/vector1.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <common/vector4.h>
#include <initializer_list>
#include <ranges>

namespace mpp::image
{

/// <summary>
/// A list of Channel objects, mainly to allow multiple implicit constructors for a short channel list
/// </summary>
template <size_t listSize> class MPPEXPORT_COMMON ChannelList
{
  public:
    ChannelList(std::initializer_list<Channel> aInitList)
    {
        if (aInitList.size() <= listSize)
        {
            std::copy(aInitList.begin(), aInitList.end(), mChannels.begin());
        }
        else
        {
            std::copy(aInitList.begin(), aInitList.begin() + listSize, mChannels.begin());
        }
    }
    ChannelList(std::initializer_list<Axis1D> aInitList)
    {
        if (aInitList.size() <= listSize)
        {
            std::copy(aInitList.begin(), aInitList.end(), mChannels.begin());
        }
        else
        {
            std::copy(aInitList.begin(), aInitList.begin() + listSize, mChannels.begin());
        }
    }
    ChannelList(std::initializer_list<Axis2D> aInitList)
    {
        if (aInitList.size() <= listSize)
        {
            std::copy(aInitList.begin(), aInitList.end(), mChannels.begin());
        }
        else
        {
            std::copy(aInitList.begin(), aInitList.begin() + listSize, mChannels.begin());
        }
    }
    ChannelList(std::initializer_list<Axis3D> aInitList)
    {
        if (aInitList.size() <= listSize)
        {
            std::copy(aInitList.begin(), aInitList.end(), mChannels.begin());
        }
        else
        {
            std::copy(aInitList.begin(), aInitList.begin() + listSize, mChannels.begin());
        }
    }
    ChannelList(std::initializer_list<Axis4D> aInitList)
    {
        if (aInitList.size() <= listSize)
        {
            std::copy(aInitList.begin(), aInitList.end(), mChannels.begin());
        }
        else
        {
            std::copy(aInitList.begin(), aInitList.begin() + listSize, mChannels.begin());
        }
    }

    // avoid implicit use of int* constructor
    ChannelList(int aInt)
        requires(listSize == 1)
    {
        mChannels[0] = aInt;
    }

    // avoid implicit use of uint* constructor
    ChannelList(uint aInt)
        requires(listSize == 1)
    {
        mChannels[0] = aInt;
    }

    ChannelList(int *aIntPtr)
    {
        std::copy(aIntPtr, aIntPtr + listSize, mChannels.begin());
    }

    ChannelList(uint *aIntPtr)
    {
        std::copy(aIntPtr, aIntPtr + listSize, mChannels.begin());
    }

    ChannelList(Channel *aChannelPtr)
    {
        std::copy(aChannelPtr, aChannelPtr + listSize, mChannels.begin());
    }

    ChannelList(Axis1D *aChannelPtr)
    {
        std::copy(aChannelPtr, aChannelPtr + listSize, mChannels.begin());
    }

    ChannelList(Axis2D *aChannelPtr)
    {
        std::copy(aChannelPtr, aChannelPtr + listSize, mChannels.begin());
    }

    ChannelList(Axis3D *aChannelPtr)
    {
        std::copy(aChannelPtr, aChannelPtr + listSize, mChannels.begin());
    }

    ChannelList(Axis4D *aChannelPtr)
    {
        std::copy(aChannelPtr, aChannelPtr + listSize, mChannels.begin());
    }

    ChannelList(const std::ranges::forward_range auto &aRange)
    {
        if (aRange.size() <= listSize)
        {
            std::copy(aRange.begin(), aRange.end(), mChannels.begin());
        }
        else
        {
            std::copy(aRange.begin(), aRange.begin() + listSize, mChannels.begin());
        }
    }

    ~ChannelList() = default;

    ChannelList(const ChannelList &)     = default;
    ChannelList(ChannelList &&) noexcept = default;

    ChannelList &operator=(const ChannelList &)     = default;
    ChannelList &operator=(ChannelList &&) noexcept = default;

    Channel *data()
    {
        return mChannels.data();
    }
    const Channel *data() const
    {
        return mChannels.data();
    }

  private:
    std::array<Channel, listSize> mChannels{0};
};

} // namespace mpp::image