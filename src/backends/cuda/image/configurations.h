#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <cuda_runtime_api.h>

namespace opp::cuda::image
{
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

constexpr ConstExprDim3 DefaultBlockSize()
{
    return {32, 8, 1};
}

struct DefaultConfiguration
{
    static constexpr ConstExprDim3 BlockSize{32, 8, 1};
    static constexpr int WarpAlignmentInBytes{64};
};

template <size_t typeSize, int hardwareMajor = 0, int hardwareMinor = 0, int configVersion = 0>
struct KernelConfiguration
{
    static constexpr ConstExprDim3 BlockSize{DefaultConfiguration::BlockSize};
    static constexpr int WarpAlignmentInBytes{DefaultConfiguration::WarpAlignmentInBytes};
    static constexpr size_t TupelSize{typeSize == 1 ? 8 : typeSize == 2 ? 4 : typeSize == 4 ? 2 : 1};
};

// template <size_t typeSize> struct KernelConfiguration<typeSize, 0, 0, 0>
//{
//     static constexpr ConstExprDim3 BlockSize{DefaultConfiguration::BlockSize};
//     static constexpr int WarpAlignmentInBytes{DefaultConfiguration::WarpAlignmentInBytes};
//     static constexpr size_t TupelSize{typeSize == 1 ? 8 : typeSize == 2 ? 4 : typeSize == 4 ? 2 : 1};
// };
} // namespace opp::cuda::image