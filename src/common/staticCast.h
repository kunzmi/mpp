#pragma once

#include "bfloat16.h"
#include "defines.h"
#include "half_fp16.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include <cfloat>
#include <cmath>
#include <numbers>
#include <vector>

namespace opp
{
/// <summary>
/// Our own cast function that does mostly just a static cast. But for floating point to integer casts, behaviour is
/// different depending on the platform: CUDA clamps float-to-int casts to the integer value range and NANs get flushed
/// to zero by default. CPU/C++ code converts float values larger than int.max() to int.min() and NAN usually becomes
/// int.min(), too. So for CPU we must emulate CUDA's behaviour...<para/>
/// Further to note is, that int.max() = 2147483647 and cannot be represented exactly by float32 and will give
/// 2147483648, thus we cannot use the clamp to min/max feature implemented for Vector1..4.</para>
/// For floating point to integer casts, rounding towards zero (or truncation) is applied, other rounding modes must be
/// applied to the floating point value before casting.
/// </summary>
template <typename TFrom, typename TTo> DEVICE_CODE TTo StaticCast(TFrom aValue)
{
    if constexpr (IsBFloat16<TFrom> && IsHalfFp16<TTo>)
    {
        // For HalfFp16 to BFloat, if that makes sense at all, cast over float
        return HalfFp16(static_cast<float>(aValue));
    }
    else if constexpr (IsHalfFp16<TFrom> && IsBFloat16<TTo>)
    {
        // same for the inverse direction
        return BFloat16(static_cast<float>(aValue));
    }
    else
    {
        return static_cast<TTo>(aValue);
    }
}

template <typename TFrom, typename TTo>
DEVICE_CODE TTo StaticCast(TFrom aValue)
    requires RealIntegral<TTo> && RealFloatingPoint<TFrom> && HostCode<TTo>
{
    if (std::isnan(aValue))
    {
        return 0; // flush NAN to 0
    }
    // limit to TTo-max-value
    if (aValue >= static_cast<TFrom>(numeric_limits<TTo>::max()))
    {
        return numeric_limits<TTo>::max();
    }
    // for int everything below int.min() gets automatically clamped to int.min(), but not for the other int types:
    if constexpr (ByteSizeType<TTo> || TwoBytesSizeType<TTo> || std::same_as<TTo, uint>)
    {
        if (aValue < static_cast<TFrom>(numeric_limits<TTo>::lowest()))
        {
            return numeric_limits<TTo>::lowest();
        }
    }
    return static_cast<TTo>(aValue);
}

// It seems there is no intrinsic for (u)short/(s)byte that uses the same "clamp and cast" feature from standard
// static_cast<int>(float). So for float to (u)short/(s)byte we have to use PTX inline so that value range is
// limited to integer.min()/max() and NAN gets flushed to 0, same as for int32
template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, short> && std::same_as<TFrom, float> && CUDA_ONLY<TTo>
{
    short ret;
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.s16.f32 %0, %1;" : "=h"(ret) : "f"(aValue));
#endif
    return ret;
}

template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, ushort> && std::same_as<TFrom, float> && CUDA_ONLY<TTo>
{
    ushort ret;
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.u16.f32 %0, %1;" : "=h"(ret) : "f"(aValue));
#endif
    return ret;
}

template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, sbyte> && std::same_as<TFrom, float> && CUDA_ONLY<TTo>
{
    short ret; // there are no 8-bit registers...
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.s8.f32 %0, %1;" : "=h"(ret) : "f"(aValue));
#endif
    return ret;
}

template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, byte> && std::same_as<TFrom, float> && CUDA_ONLY<TTo>
{
    ushort ret; // there are no 8-bit registers...
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.u8.f32 %0, %1;" : "=h"(ret) : "f"(aValue));
#endif
    return ret;
}
template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, short> && std::same_as<TFrom, double> && CUDA_ONLY<TTo>
{
    short ret;
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.s16.f64 %0, %1;" : "=h"(ret) : "d"(aValue));
#endif
    return ret;
}

template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, ushort> && std::same_as<TFrom, double> && CUDA_ONLY<TTo>
{
    ushort ret;
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.u16.f64 %0, %1;" : "=h"(ret) : "d"(aValue));
#endif
    return ret;
}

template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, sbyte> && std::same_as<TFrom, double> && CUDA_ONLY<TTo>
{
    short ret; // there are no 8-bit registers...
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.s8.f64 %0, %1;" : "=h"(ret) : "d"(aValue));
#endif
    return ret;
}

template <typename TFrom, typename TTo>
DEVICE_ONLY_CODE TTo StaticCast(TFrom aValue)
    requires std::same_as<TTo, byte> && std::same_as<TFrom, double> && CUDA_ONLY<TTo>
{
    ushort ret; // there are no 8-bit registers...
#ifdef IS_CUDA_COMPILER
    asm("cvt.rzi.u8.f64 %0, %1;" : "=h"(ret) : "d"(aValue));
#endif
    return ret;
}
} // namespace opp