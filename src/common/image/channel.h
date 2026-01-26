#pragma once
#include "pixelTypes.h"
#include <common/defines.h>
#include <common/exception.h>
#include <common/vector1.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <common/vector4.h>
#include <common/vector4A.h>
#include <iostream>

namespace mpp::image
{
enum class RGBA : uint // NOLINT(performance-enum-size)
{
    R     = 0,
    G     = 1,
    B     = 2,
    Alpha = 3
};
enum class BGRA : uint // NOLINT(performance-enum-size)
{
    B     = 0,
    G     = 1,
    R     = 2,
    Alpha = 3
};
enum class LUV : uint // NOLINT(performance-enum-size)
{
    L     = 0,
    U     = 1,
    V     = 2,
    Alpha = 3
};
enum class HSV : uint // NOLINT(performance-enum-size)
{
    H     = 0,
    S     = 1,
    V     = 2,
    Alpha = 3
};
enum class HSL : uint // NOLINT(performance-enum-size)
{
    H     = 0,
    S     = 1,
    L     = 2,
    Alpha = 3
};
enum class YUV : uint // NOLINT(performance-enum-size)
{
    Y     = 0,
    U     = 1,
    V     = 2,
    Alpha = 3
};
enum class YCbCr : uint // NOLINT(performance-enum-size)
{
    Y     = 0,
    Cb    = 1,
    Cr    = 2,
    Alpha = 3
};
enum class YCrCb : uint // NOLINT(performance-enum-size)
{
    Y     = 0,
    Cr    = 1,
    Cb    = 2,
    Alpha = 3
};
enum class CMYK : uint // NOLINT(performance-enum-size)
{
    C = 0,
    M = 1,
    Y = 2,
    K = 3
};
enum class XYZ : uint // NOLINT(performance-enum-size)
{
    X     = 0,
    Y     = 1,
    Z     = 2,
    Alpha = 3
};

/// <summary>
/// A generic interface to image color channels, takes as input any integer or defined channel enum, and converts to
/// Axis1/2/3/4 enum for vector classes. Throws an exception on host code if value exceeds valid range.
/// </summary>
class Channel
{
  public:
    Channel() = default;
    DEVICE_CODE inline Channel(uint aChannel) : mChannel(aChannel) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(int aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(RGBA aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(BGRA aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(LUV aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(HSV aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(HSL aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(YUV aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(YCbCr aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(YCrCb aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(CMYK aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(XYZ aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(Axis1D aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(Axis2D aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(Axis3D aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }
    DEVICE_CODE inline Channel(Axis4D aChannel) : mChannel(static_cast<uint>(aChannel)) // NOLINT
    {
    }

    ~Channel() = default;

    Channel(const Channel &)     = default;
    Channel(Channel &&) noexcept = default;

    Channel &operator=(const Channel &)     = default;
    Channel &operator=(Channel &&) noexcept = default;

    DEVICE_CODE inline operator Axis1D() const // NOLINT
    {
#ifdef IS_HOST_COMPILER
        if (mChannel != 0)
        {
            throw INVALIDARGUMENT(ColorChannel, "Invalid channel value for a 1 channel image: "
                                                    << mChannel << ". Expected a value in range [0].");
        }
#endif
        return static_cast<Axis1D>(mChannel);
    }

    DEVICE_CODE inline operator Axis2D() const // NOLINT
    {
#ifdef IS_HOST_COMPILER
        if (mChannel > 1)
        {
            throw INVALIDARGUMENT(ColorChannel, "Invalid channel value for a 2 channel image: "
                                                    << mChannel << ". Expected a value in range [0..1].");
        }
#endif
        return static_cast<Axis2D>(mChannel);
    }

    DEVICE_CODE inline operator Axis3D() const // NOLINT
    {
#ifdef IS_HOST_COMPILER
        if (mChannel > 2)
        {
            throw INVALIDARGUMENT(ColorChannel, "Invalid channel value for a 3 channel image: "
                                                    << mChannel << ". Expected a value in range [0..2].");
        }
#endif
        return static_cast<Axis3D>(mChannel);
    }

    DEVICE_CODE inline operator Axis4D() const // NOLINT
    {
#ifdef IS_HOST_COMPILER
        if (mChannel > 3)
        {
            throw INVALIDARGUMENT(ColorChannel, "Invalid channel value for a 4 channel image: "
                                                    << mChannel << ". Expected a value in range [0..3].");
        }
#endif
        return static_cast<Axis4D>(mChannel);
    }

    template <typename T> [[nodiscard]] DEVICE_CODE bool IsInRange() const
    {
        static_assert(AlwaysFalse<T>::value,
                      "Calling IsInRange with an unsupported template parameter, use Axis1..4D or VectorX-types only.");
        return false;
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires std::same_as<T, Axis1D>
    {
        return mChannel == 0;
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires std::same_as<T, Axis2D>
    {
        return mChannel <= 1;
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires std::same_as<T, Axis3D>
    {
        return mChannel <= 2;
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires std::same_as<T, Axis4D>
    {
        return mChannel <= 3;
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires SingleChannel<T>
    {
        return IsInRange<Axis1D>();
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires TwoChannel<T>
    {
        return IsInRange<Axis2D>();
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires ThreeChannel<T>
    {
        return IsInRange<Axis3D>();
    }
    template <typename T>
    [[nodiscard]] DEVICE_CODE bool IsInRange() const
        requires FourChannel<T>
    {
        return IsInRange<Axis4D>();
    }

    [[nodiscard]] DEVICE_CODE uint Value() const
    {
        return mChannel;
    }

  private:
    uint mChannel{0};
};

inline std::ostream &operator<<(std::ostream &aOs, const Channel &aChannel)
{
    aOs << aChannel.Value();
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const Channel &aChannel)
{
    aOs << aChannel.Value();
    return aOs;
}

} // namespace mpp::image