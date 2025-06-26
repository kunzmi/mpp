#pragma once
#include <common/moduleEnabler.h>
#if MPP_ENABLE_CUDA_BACKEND

#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <cuda_runtime_api.h>

namespace mpp::image::cuda
{
// allow named config names in template
template <size_t sizeConfigName> struct ConfigNameWrapper
{
    constexpr ConfigNameWrapper(const char (&aConfigName)[sizeConfigName])
    {
        std::copy(aConfigName, aConfigName + sizeConfigName, configName);
    }

    char configName[sizeConfigName];
};

// cuda's dim3 is not constexpr
struct ConstExprDim3
{
    // implicit conversion
    operator dim3() const
    {
        dim3 ret;
        ret.x = x;
        ret.y = y;
        ret.z = z;
        return ret;
    }

    uint x;
    uint y;
    uint z;
};

// Block size for kernel launch
template <ConfigNameWrapper ConfigName, int hardwareMajor = 0, int hardwareMinor = 0> struct ConfigBlockSize
{
    static constexpr ConstExprDim3 value{32, 8, 1};
};
template <> struct ConfigBlockSize<"Default">
{
    static constexpr ConstExprDim3 value{32, 8, 1};
};
template <> struct ConfigBlockSize<"DefaultReductionX">
{
    static constexpr ConstExprDim3 value{32, 8, 1};
};
template <> struct ConfigBlockSize<"DefaultReductionY">
{
    static constexpr ConstExprDim3 value{32, 32, 1};
};
template <> struct ConfigBlockSize<"DefaultReductionYLargeType">
{
    static constexpr ConstExprDim3 value{32, 16, 1};
};

// Warp alignment, the size in bytes a warp should be aligned to
template <ConfigNameWrapper ConfigName, int hardwareMajor = 0, int hardwareMinor = 0> struct ConfigWarpAlignment
{
    static constexpr int value = 64;
};
template <> struct ConfigWarpAlignment<"Default">
{
    static constexpr int value = 64;
};

// The tupel size to use for a given type size
template <ConfigNameWrapper ConfigName, size_t typeSize, int hardwareMajor = 0, int hardwareMinor = 0>
struct ConfigTupelSize
{
    static constexpr size_t value{typeSize == 1 ? 8 : typeSize == 2 ? 4 : typeSize == 4 ? 2 : 1};
};
template <size_t typeSize> struct ConfigTupelSize<"Default", typeSize>
{
    // assuming 64 byte warp alignment
    static constexpr size_t value{typeSize == 1 ? 8 : typeSize == 2 ? 4 : typeSize == 4 ? 2 : 1};
};
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND